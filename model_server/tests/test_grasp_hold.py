from pathlib import Path
import json
import os

import numpy as np
import pytest

from robot_client.sonic.run import load_config
import robot_client.sonic.grasp_hold as module


class RobotKinematicsFixture:
    def __init__(self,*args):pass
    def seed_q(self,q,**kwargs):return np.asarray(q).copy()
    def forward(self,q):
        poses=[np.eye(4),np.eye(4)]
        poses[0][2,3]=q[15]
        poses[1][2,3]=q[22]
        return poses
    def solve(self,targets,q,**kwargs):
        result=np.asarray(q).copy()
        result[15]=targets[0][2,3]
        result[22]=targets[1][2,3]
        return result


@pytest.fixture
def guard(monkeypatch):
    monkeypatch.setattr(module,'G1Kinematics',RobotKinematicsFixture)
    config=load_config(Path(module.__file__).with_name('vla_stable.yaml'))
    return module.GraspHold('unused',config)


def sensor(t,height=.2,forces=(1.,1.),jaws=(.3,.3)):
    q=np.zeros(29)
    q[15]=height
    return {'simulation_time':t,'jaw_force_n':[list(forces),[0,0]],
            'jaw_q':[list(jaws),[0,0]],'body_q':q.tolist(),'base_quat':[1,0,0,0],
            'base_position':[0,0,0]}


@pytest.mark.parametrize('forces,jaws,grip',[
    ((0,0),(.3,.3),2),((1,0),(.3,.3),2),((0,1),(.3,.3),2),
    ((1,1),(.65,.65),0),((1,1),(.3,.3),5.5)])
def test_no_latch_from_air_one_jaw_empty_closure_or_open_command(guard,forces,jaws,grip):
    for t in [0,.1,.2,1]:
        guard.update(sensor(t,forces=forces,jaws=jaws),[grip,5.5],.75)
    assert guard.latched_grip is None
    assert not guard.holding


def test_hold_requires_vla_closure_contact_and_measured_robot_lift(guard):
    guard.waist_reference=np.array([0,.018,.136])
    guard.update(sensor(0),[2,5.5],.75)
    guard.update(sensor(.1),[2,5.5],.75)
    assert guard.latched_grip==2
    grips,_=guard.update(sensor(.2,height=.21),[5.5,5.5],.75)
    np.testing.assert_equal(grips,[2,5.5])
    assert not guard.holding  # No synthesized lift or approach is generated.
    current=sensor(.3,height=.34)
    guard.update(current,[1.8,5.5],.74)
    assert guard.holding
    expected=np.asarray(current['body_q']).copy()
    expected[12:15]=guard.waist_reference
    np.testing.assert_array_equal(guard.hold_q,expected)
    _,done=guard.update(sensor(6.2,height=.34),[5.5,5.5],.74)
    assert not done
    _,done=guard.update(sensor(6.31,height=.34),[5.5,5.5],.74)
    assert done
    assert json.loads(json.dumps({'completed':done,'events':guard.events}))['completed'] is True


def test_lost_support_cannot_be_reported_as_maintained_hold(guard):
    guard.update(sensor(0),[2,5.5],.75)
    guard.update(sensor(.1),[2,5.5],.75)
    guard.update(sensor(.2,height=.34),[2,5.5],.75)
    guard.update(sensor(.3,height=.34,forces=(0,0)),[2,5.5],.75)
    with pytest.raises(RuntimeError,match='support lost'):
        guard.update(sensor(.6,height=.34,forces=(0,0)),[2,5.5],.75)


def test_reset_and_cyclic_config_are_errors(guard,tmp_path):
    guard.update(sensor(2),[2,5.5],.75)
    with pytest.raises(RuntimeError,match='clock reset'):
        guard.update(sensor(0),[2,5.5],.75)
    path=tmp_path/'cycle.yaml'
    path.write_text('base_config: cycle.yaml\n')
    with pytest.raises(ValueError,match='cyclic'):
        load_config(path)


def test_parallel_joint_units_do_not_change_grasp_confirmation(monkeypatch):
    monkeypatch.setattr(module,'G1Kinematics',RobotKinematicsFixture)
    config=load_config(Path(module.__file__).with_name('vla_parallel.yaml'))
    config['grasp_hold_world_frame']=False
    guard=module.GraspHold('unused',config)
    guard.update(sensor(0,jaws=(.025,.025)),[2,5.5],.75)
    guard.update(sensor(.1,jaws=(.025,.025)),[2,5.5],.75)
    assert guard.latched_grip==2
    guard.update(sensor(.2,height=.34,jaws=(.025,.025)),[2,5.5],.75)
    assert guard.holding


def test_force_feedback_only_closes_after_vla_contact_and_resets_on_loss(guard):
    guard.force_target=2.
    # An empty close request cannot invoke a force trim.
    grips,_=guard.update(sensor(0,forces=(0,0)),[2,5.5],.75)
    np.testing.assert_array_equal(grips,[2,5.5])
    guard.update(sensor(.1,forces=(.1,.1)),[2,5.5],.75)
    grips,_=guard.update(sensor(.2,forces=(.1,.1)),[2,5.5],.75)
    assert 0<grips[0]<2
    assert grips[1]==5.5
    for t in np.arange(.3,1.2,.1):grips,_=guard.update(sensor(t,forces=(.1,.1)),[5.5,5.5],.75)
    assert guard.force_correction==pytest.approx(1.5)
    assert grips[0]==pytest.approx(.5)
    correction=guard.force_correction
    guard.update(sensor(1.3,forces=(3,3)),[5.5,5.5],.75)
    assert guard.force_correction==correction
    guard.update(sensor(1.4,forces=(0,0)),[5.5,5.5],.75)
    grips,_=guard.update(sensor(1.7,forces=(0,0)),[5.5,5.5],.75)
    assert guard.force_correction==0
    assert guard.latched_grip is None
    np.testing.assert_array_equal(grips,[5.5,5.5])


def test_force_config_resolves_all_physical_and_execution_settings():
    cfg=load_config(Path(module.__file__).with_name('vla_force.yaml'))
    assert cfg['hand_closed'][0][:2]==[.045,.045]
    assert cfg['grasp_hold']['hold_sim_seconds']==6
    assert cfg['grasp_hold_world_frame'] and cfg['compensate_base_motion']
    assert cfg['base_position_hold'] and cfg['stationary_single_arm']


def make_reflex(monkeypatch):
    monkeypatch.setattr(module,'G1Kinematics',RobotKinematicsFixture)
    config=load_config(Path(module.__file__).with_name('vla_grasp_reflex.yaml'))
    return module.GraspHold('unused',config)


def test_lift_reflex_requires_vla_lift_and_only_adds_bounded_vertical_motion(monkeypatch):
    guard=make_reflex(monkeypatch)
    for t in [0,.1,.2]:guard.update(sensor(t,jaws=(.025,.025)),[2,5.5],.75)
    assert not guard.holding  # Contact and closure alone cannot invoke lift.
    guard.update(sensor(.3,height=.25,jaws=(.025,.025)),[2,5.5],.75)
    assert not guard.holding
    guard.update(sensor(.4,height=.265,jaws=(.025,.025)),[2,5.5],.75)
    event=guard.events[-1]
    assert event['event']=='lift_reflex'
    assert event['vla_hand_lift_m']>=.06
    assert 0<event['assisted_lift_m']<=.06
    initial=guard.world_grasp_pose.copy()
    for t in np.arange(.5,3.4,.1):
        height=guard.hold_q[15]  # Robot follows the previously requested pose.
        guard.update(sensor(t,height=height,forces=(4.5,4.5),jaws=(.025,.025)),[2,5.5],.75)
        np.testing.assert_array_equal(guard.world_grasp_pose[:2,3],initial[:2,3])
        np.testing.assert_array_equal(guard.world_grasp_pose[:3,:3],initial[:3,:3])
        assert .265<=guard.world_grasp_pose[2,3]<=.320000001
    assert guard.reflex_finished
    _,done=guard.update(sensor(guard.hold_started+5.99,height=.32,jaws=(.025,.025)),[2,5.5],.75)
    assert not done
    _,done=guard.update(sensor(guard.hold_started+6.01,height=.32,jaws=(.025,.025)),[2,5.5],.75)
    assert done


def test_reflex_cannot_complete_if_robot_does_not_reach_target(monkeypatch):
    guard=make_reflex(monkeypatch)
    guard.update(sensor(0,jaws=(.025,.025)),[2,5.5],.75)
    guard.update(sensor(.1,jaws=(.025,.025)),[2,5.5],.75)
    guard.update(sensor(.2,height=.265,jaws=(.025,.025)),[2,5.5],.75)
    with pytest.raises(RuntimeError,match='did not reach'):
        guard.update(sensor(8,height=.265,jaws=(.025,.025)),[2,5.5],.75)


def test_world_hold_compensates_robot_odometry_without_new_lift_goal():
    from scipy.spatial.transform import Rotation
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for actual G1 kinematics')
    config=load_config(Path(module.__file__).with_name('vla_grasp_ready.yaml'))
    guard=module.GraspHold(root,config)
    q=np.zeros(29)
    q[[3,9]]=.2
    q[[18,25]]=.6
    q[15]=-.8
    q[16]=.3
    initial={'body_q':q,'base_position':[0,0,.75],'base_quat':[1,0,0,0]}
    guard.hold_anchor=q.copy()
    guard.world_grasp_pose=guard.base_transform(initial)@guard.kin.forward(q)[0]
    quat=Rotation.from_euler('xyz',[.015,-.01,.02]).as_quat()[[3,0,1,2]]
    measured=q.copy()
    measured[13:15]=[.03,-.025]
    disturbed=dict(initial,body_q=measured,base_position=[.01,-.012,.735],base_quat=quat)
    compensated=guard.hold_command(disturbed)
    # The fast arm loop compensates the waist's current tracking error, while
    # the separate waist reference continues to request the captured posture.
    actual=compensated.copy()
    actual[12:15]=measured[12:15]
    measured_world=guard.base_transform(disturbed)@guard.kin.forward(actual)[0]
    np.testing.assert_allclose(measured_world[:3,3],guard.world_grasp_pose[:3,3],atol=.005)
    np.testing.assert_array_equal(compensated[22:],q[22:])
    np.testing.assert_array_equal(compensated[12:15],q[12:15])
