import os
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pytest
import yaml

from robot_client.sonic.vision_target import command_color,condition_images,detect_target,HeadProjector
from robot_client.sonic.run import load_config
from robot_client.sonic.launch_sim import prepare_scene
from robot_client.sonic.kinematics import JOINT_NAMES


def scene_image():
    image=np.full((100,160,3),170,np.uint8)
    image[30:50,20:40]=[0,0,220]
    image[30:50,70:90]=[0,200,0]
    image[30:50,120:140]=[0,200,220]
    return image


def test_target_color_conditioning_preserves_target_geometry_and_original_rgb():
    original=scene_image();copy=original.copy()
    images,counts=condition_images({'head':original},'red');result=images['head']
    np.testing.assert_array_equal(original,copy)
    np.testing.assert_array_equal(result[30:50,20:40],original[30:50,20:40])
    assert counts['head']==800
    assert np.all(result[30:50,70:90,0]==result[30:50,70:90,1])
    np.testing.assert_array_equal(result[0:20],original[0:20])
    assert detect_target(result,'red')['bbox_px']==[20,30,40,50]
    with pytest.raises(ValueError,match='not visible'):detect_target(result,'green')


def test_target_selection_rejects_ambiguity_and_handles_bgr_color_order():
    image=scene_image()
    assert command_color('用左手抓红色方块')=='red'
    assert command_color('Pick up the GREEN cube with your left hand')=='green'
    with pytest.raises(ValueError,match='exactly one'):command_color('pick up red or green')
    image[60:80,20:40]=[0,0,220]
    with pytest.raises(ValueError,match='ambiguous'):detect_target(image,'red')


def test_robot_fk_projector_matches_full_scene_camera(tmp_path):
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for actual camera FK')
    root=Path(root);task=yaml.safe_load(Path('robot_client/sonic/tabletop_dex1_synchronized.yaml').read_text())
    config=load_config('robot_client/sonic/vla_cad_synchronized.yaml')
    scene=tmp_path/'scene.xml'
    prepare_scene(root,root/'gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml',scene,task=task)
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
    q=np.zeros(29);q[[3,9]]=.3;q[[18,25]]=.6;q[12:15]=[.03,-.02,.1]
    d.qpos[:7]=[0,0,.75,1,0,0,0]
    for name,value in zip(JOINT_NAMES,q):d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
    mujoco.mj_forward(m,d)
    cam=m.camera('unifolm_head').id
    # Synthetic points are solely a calibration test, not task object poses.
    points=np.array([[.3,.1,.1],[.4,-.1,.15],[.25,0,.05]])
    world=points+[0,0,.75];local=(world-d.cam_xpos[cam])@d.cam_xmat[cam].reshape(3,3)
    f=.5*336/np.tan(np.deg2rad(m.cam_fovy[cam]/2))
    expected=np.c_[f*local[:,0]/-local[:,2]+223.5,-f*local[:,1]/-local[:,2]+167.5]
    p=HeadProjector(root,config,task['camera_mounts']['unifolm_head']);actual,depth=p.project(points,q,(336,448,3))
    np.testing.assert_allclose(actual,expected,atol=.002)
    np.testing.assert_allclose(depth,-local[:,2],atol=3e-6)


def test_visual_selection_never_closes_on_distant_target_and_keeps_proposal_unchanged():
    from robot_client.sonic.vision_target import VisualActionSelector
    from robot_client.sonic.smoke import fixture_action
    from robot_client.sonic.kinematics import G1Kinematics
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for robot FK')
    config=load_config('robot_client/sonic/vla_rgb_selected.yaml')
    task=yaml.safe_load(Path('robot_client/sonic/tabletop_dex1_synchronized.yaml').read_text())
    config['visual_target']['camera_mounts']=task['camera_mounts']
    selector=VisualActionSelector(root,config)
    q=np.zeros(29);q[[3,9]]=.3;q[[18,25]]=.6
    sensor={'body_q':q,'base_quat':[1,0,0,0],'base_position':[0,0,.75]}
    actual=(selector.base(sensor)@selector.physical.forward(q)[0])[:3,3]
    selector.target_world=actual+[.12,0,0];selector.last_seen=1.
    gated=selector.gate_grip(sensor,[0,4.5],np.array([5.5,5.5]),False,1.1)
    assert gated[0]==pytest.approx(config['ready_gripper_value'][0])
    assert selector.blocked_close_packets==1
    selector.target_world=actual.copy()
    assert selector.gate_grip(sensor,[0,4.5],np.array([5.5,5.5]),False,1.1)[0]==0
    assert selector.gate_grip(sensor,[0,4.5],np.array([5.5,5.5]),False,4.)[0]>4
    kin=G1Kinematics(root,tool_offsets=config['tool_offsets'])
    a=fixture_action(kin,q);b={k:v.copy() for k,v in a.items()}
    b['action.left_ee_rpy'][:,:,0]+=.1
    saved={k:v.copy() for k,v in a.items()}
    selected,report=selector.select([b,a],sensor,1.1)
    assert selected is a and report['selected']==1
    for key in a:np.testing.assert_array_equal(a[key],saved[key])


def test_multiview_target_ignores_one_occluded_biased_camera():
    from robot_client.sonic.vision_target import MultiViewProjector
    projector=object.__new__(MultiViewProjector)
    projector.mounts={name:{'fovy':90} for name in ('unifolm_head','unifolm_left_wrist','unifolm_right_wrist')}
    transforms={};images={};target=np.array([.3,0,.1])
    for index,(key,x) in enumerate(zip(('cam_left_high','cam_left_wrist','cam_right_wrist'),(.1,.3,.5))):
        t=np.eye(4);t[:3,3]=[x,0,1];transforms[key]=t
        local=target-t[:3,3];pixel=np.array([168*local[0]/-local[2]+223.5,-168*local[1]/-local[2]+167.5])
        if index==1:pixel[1]+=60  # partial face/occluder gives a bad color center
        px,py=np.rint(pixel).astype(int);image=np.full((336,448,3),170,np.uint8)
        image[py-5:py+6,px-5:px+6]=[0,0,220];images[key]=image
    projector.transforms=lambda q:transforms
    located=projector.locate(images,np.zeros(29),'red',prior_pelvis=target)
    assert set(located['views'])=={'cam_left_high','cam_right_wrist'}
    assert located['rejected_views']==['cam_left_wrist']
    np.testing.assert_allclose(located['center_pelvis'],target,atol=.004)


def test_orientation_constraint_preserves_every_physical_vla_tcp_position():
    from robot_client.sonic.vision_target import VisualActionSelector
    from robot_client.sonic.smoke import fixture_action
    from robot_client.sonic.kinematics import G1Kinematics,pose_matrix
    from scipy.spatial.transform import Rotation
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for robot FK')
    config=load_config('robot_client/sonic/vla_rgb_clearance.yaml')
    task=yaml.safe_load(Path('robot_client/sonic/tabletop_dex1_synchronized.yaml').read_text());config['visual_target']['camera_mounts']=task['camera_mounts']
    selector=VisualActionSelector(root,config);q=np.zeros(29);q[[3,9]]=.3;q[[18,25]]=.6
    kin=G1Kinematics(root,tool_offsets=config['tool_offsets']);action=fixture_action(kin,q)
    quat=Rotation.from_euler('xyz',[.03,-.1,.12]).as_quat()[[3,0,1,2]]
    state={'body_q':q,'base_quat':quat,'base_position':[.03,-.02,.75]}
    result=selector.constrain_orientation(action,state)
    raw=selector.projector.action_points(action,0);actual=selector.projector.action_points(result,0)
    np.testing.assert_allclose(actual,raw,atol=1e-7)
    for pose in result['action.left_ee_rpy'][0]:
        rotation=(selector.base(state)@pose_matrix(pose))[:3,:3]
        assert abs(rotation[2,1])<1e-7
        assert rotation[2,0]==pytest.approx(-np.sin(np.deg2rad(20)),abs=1e-7)
    np.testing.assert_array_equal(result['action.right_ee_rpy'],action['action.right_ee_rpy'])


def test_prefix_is_an_unchanged_vla_segment_and_stops_before_low_tail():
    from robot_client.sonic.vision_target import VisualActionSelector
    from robot_client.sonic.smoke import fixture_action
    from robot_client.sonic.kinematics import G1Kinematics
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for robot FK')
    config=load_config('robot_client/sonic/vla_rgb_clearance.yaml');task=yaml.safe_load(Path('robot_client/sonic/tabletop_dex1_synchronized.yaml').read_text());config['visual_target']['camera_mounts']=task['camera_mounts']
    selector=VisualActionSelector(root,config);q=np.zeros(29);q[[3,9]]=.3;q[[18,25]]=.6
    action=fixture_action(G1Kinematics(root,tool_offsets=config['tool_offsets']),q)
    action['action.left_ee_rpy'][0,:,2]-=np.linspace(0,.10,30)
    state={'body_q':q,'base_quat':[1,0,0,0],'base_position':[0,0,.75]}
    points=selector.projector.action_points(action,0)
    selector.target_world=points[0]+[0,0,.75-.04];selector.last_seen=1.
    chosen,report=selector.select([action],state,1.1);n=report['prefix_frames']
    assert 3<=n<30
    for key in action:np.testing.assert_array_equal(chosen[key],action[key][:,:n])
    end_world=selector.projector.action_points(chosen,0)[-1]+[0,0,.75]
    assert end_world[2]>selector.target_world[2]+.005
