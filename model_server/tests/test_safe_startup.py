import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml
import zmq
from scipy.spatial.transform import Rotation

from robot_client.sonic.arm_servo import ArmReferenceServo
from robot_client.sonic.kinematics import G1Kinematics, JOINT_NAMES, pose_matrix
from robot_client.sonic.launch_sim import prepare_scene, task_video_due
from robot_client.sonic.grasp import enable_task_contacts
from robot_client.sonic.planner import PlannerMapper
from robot_client.sonic.arm_servo import planner_fields
from robot_client.sonic.protocol import UPPER_MUJOCO_IN_ISAAC
from robot_client.sonic.run import startup_upper_body_packet


def test_safe_landing_packet_matches_captured_upper_body_and_does_not_chase_drift():
    q=np.linspace(-.2,.2,29)
    state={'body_q':q.tolist(),'base_quat':[1,0,0,0]}
    config={'base_height':.75,'hand_open':np.zeros((2,7)).tolist()}
    kin=SimpleNamespace(seed_q=lambda q,planner_legs:np.asarray(q).copy())
    planner=PlannerMapper()
    raw,anchor=startup_upper_body_packet(planner,kin,state,config,.02)
    fields=planner_fields(raw)
    np.testing.assert_allclose(fields['upper_body_position'],q[UPPER_MUJOCO_IN_ISAAC])
    np.testing.assert_array_equal(fields['upper_body_velocity'],0)
    state['body_q']=(q+.1).tolist()
    again,same=startup_upper_body_packet(planner,kin,state,config,.02,anchor)
    np.testing.assert_array_equal(same,anchor)
    np.testing.assert_array_equal(planner_fields(again)['upper_body_position'],fields['upper_body_position'])


def test_initialization_video_includes_zero_time_without_duplicate_frames():
    task={'record_from_initialization':True}
    assert task_video_due(task,True,False,0.,-1.,8)
    assert not task_video_due({},True,False,0.,-1.,8)  # Legacy recording starts at release.
    assert not task_video_due(task,True,False,0.,0.,8)
    assert not task_video_due(task,True,False,.1,0.,8)
    assert task_video_due(task,True,False,.125,0.,8)
    assert not task_video_due(task,False,True,1.,0.,8)


@pytest.fixture
def startup_model(tmp_path):
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for real CAD startup clearance')
    root=Path(root)
    task=yaml.safe_load(Path('robot_client/sonic/tabletop_rgb_green_safe.yaml').read_text())
    scene=tmp_path/'scene.xml'
    prepare_scene(root,root/'gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml',scene,task=task)
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    enable_task_contacts(model,task)
    env=SimpleNamespace(mj_model=model,mj_data=data,
        body_joint_index=np.array([model.joint(n).id for n in JOINT_NAMES]),
        robot=SimpleNamespace(MOTOR_KP=np.full(29,40.),MOTOR_KD=np.full(29,1.)),
        unitree_bridge=SimpleNamespace(low_cmd=SimpleNamespace(motor_cmd=[SimpleNamespace(kp=40.,kd=1.)]*29)))
    return root,task,env


def test_startup_is_zero_time_only_and_idle_does_not_lower_arms(startup_model):
    _,task,env=startup_model
    servo=ArmReferenceServo(env,'tcp://127.0.0.1:19997',waist=False,startup_arm_q=task['startup_arm_q'])
    try:
        qa=env.mj_model.jnt_qposadr[env.body_joint_index[15:]]
        np.testing.assert_allclose(env.mj_data.qpos[qa],task['startup_arm_q'])
        real_socket=servo.socket
        raw=PlannerMapper().idle_packet({'base_quat':[1,0,0,0]},.75,.02)
        servo.socket=SimpleNamespace(recv=lambda *args:raw)
        before=env.mj_data.qpos.copy();native=np.arange(29,dtype=float)
        command=servo.torques(native)
        np.testing.assert_array_equal(command[:12],native[:12])
        np.testing.assert_array_equal(env.mj_data.qpos,before)  # Torque feedback never teleports.
        assert servo.reference is None
        q=env.mj_data.qpos[env.mj_model.jnt_qposadr[env.body_joint_index]].copy()
        raw=PlannerMapper().packet(q,np.zeros(29),[0,0,0,.75],{'base_quat':[1,0,0,0]},.02,np.zeros((2,7)))
        command=servo.torques(native)
        assert servo.reference is not None
        np.testing.assert_array_equal(command[:15],native[:15])  # Waist returns to SONIC after handoff.
        servo.socket=real_socket
    finally:servo.close()
    env.mj_data.time=.1
    with pytest.raises(ValueError,match='time zero'):
        invalid=ArmReferenceServo(env,'tcp://127.0.0.1:19997',startup_arm_q=task['startup_arm_q'])


def test_real_mesh_startup_to_ready_path_has_table_clearance(startup_model):
    root,task,env=startup_model;m,d=env.mj_model,env.mj_data
    config=yaml.safe_load(Path('robot_client/sonic/vla_rgb_servo_guarded.yaml').read_text())
    kin=G1Kinematics(root,'pelvis',config['tool_offsets'])
    start=np.zeros(29);start[15:]=task['startup_arm_q']
    # Match run.py: redundant IK is seeded from the configured ready posture,
    # not from the independently chosen high landing posture.
    end=start.copy();end[12:15]=config['ready_waist'];end[15:]=config['ready_arm_seed']
    end=kin.solve([pose_matrix(p) for p in config['ready_pose']],end)
    qa=m.jnt_qposadr[env.body_joint_index];table=m.geom('task_table_top').id
    robot=[i for i in range(m.ngeom) if m.geom_contype[i]==1 and m.geom_bodyid[i]!=0]
    # Nominal standing envelope; dynamic trials independently audit every
    # physics step and must reject excursions beyond the clearance envelope.
    minimum=float('inf')
    for height in (.75,.79,.82):
        for pitch in (-6,0,6):
            for ratio in np.linspace(0,1,41):
                d.qpos[:3]=[0,0,height]
                d.qpos[3:7]=Rotation.from_euler('y',pitch,degrees=True).as_quat()[[3,0,1,2]]
                d.qpos[qa]=start+ratio*(end-start)
                mujoco.mj_forward(m,d)
                table_low=d.geom_xpos[table]-m.geom_size[table]
                table_high=d.geom_xpos[table]+m.geom_size[table]
                for geom in robot:
                    # Conservative world AABBs of the compiled CAD. A positive
                    # gap proves separation without relying on mj_geomDistance,
                    # which returns zero for some separated mesh/box queries.
                    rotation=d.geom_xmat[geom].reshape(3,3)
                    center=d.geom_xpos[geom]+rotation@m.geom_aabb[geom,:3]
                    extent=np.abs(rotation)@m.geom_aabb[geom,3:]
                    gap=np.maximum(0,np.maximum(center-extent-table_high,table_low-center-extent))
                    minimum=min(minimum,float(np.linalg.norm(gap)))
    assert minimum>.015
