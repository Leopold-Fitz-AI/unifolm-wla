import time
from types import SimpleNamespace

import numpy as np
import pytest

from robot_client.sonic.run import load_config
import robot_client.sonic.grasp_execution as module
from robot_client.sonic.vision_target import VisualActionSelector


class RobotFixture:
    def __init__(self,*args):pass
    def seed_q(self,q,**kwargs):return np.asarray(q,float).copy()
    def forward(self,q):
        result=[np.eye(4),np.eye(4)]
        result[0][:3,3]=q[15:18];result[1][:3,3]=q[22:25]
        return result
    def solve(self,targets,q,**kwargs):
        q=np.asarray(q).copy();q[15:18]=targets[0][:3,3];q[22:25]=targets[1][:3,3];return q


def sensor(t,forces=(0,0)):
    q=np.zeros(29);q[15:18]=[.3,.1,.1];q[22:25]=[.3,-.1,.1]
    return {'simulation_time':t,'body_q':q.tolist(),'base_quat':[1,0,0,0],
            'base_position':[0,0,0],'jaw_force_n':[list(forces),[0,0]]}


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(module,'G1Kinematics',RobotFixture)
    config=load_config('robot_client/sonic/vla_rgb_execution.yaml')
    executor=module.GraspExecution('unused',config)
    selector=SimpleNamespace(hand=0,target_world=np.array([.31,.1,.105]),last_seen=time.monotonic(),max_age=2,
                             base=VisualActionSelector.base)
    return executor,selector


def test_executor_requires_correct_visual_alignment_and_stability(setup):
    executor,selector=setup
    s=sensor(0);selector.target_world=[.45,.1,.1]
    assert not executor.consider(s,selector,[4.5,4.5],[0,0,.13])
    selector.target_world=np.array([.31,.1,.105])
    assert not executor.consider(s,selector,[4.5,4.5],[0,0,.13])
    s['simulation_time']=.11
    assert executor.consider(s,selector,[4.5,4.5],[0,0,.13])
    np.testing.assert_array_equal(executor.world_pose[:3,3],[.3,.1,.1])  # actual achieved pose, not the visual goal
    command,grip,done=executor.command(s)
    assert not done and grip[0]==pytest.approx(4.5)
    s['simulation_time']=1.0
    with pytest.raises(RuntimeError,match='bilateral'):
        s['simulation_time']=4.2;executor.command(s)


def test_loaded_close_then_bounded_lift_and_measured_hold(setup):
    executor,selector=setup;s=sensor(0)
    executor.consider(s,selector,[4.5,4.5],[0,0,.13]);s['simulation_time']=.11
    executor.consider(s,selector,[4.5,4.5],[0,0,.13])
    positions=[];done=False
    for step in range(1,900):
        s['simulation_time']=.11+step*.02;s['jaw_force_n']=[[2,2],[0,0]]
        command,grip,done=executor.command(s);positions.append(command[17])
        q=np.array(s['body_q']);q[15:18]+=.5*(command[15:18]-q[15:18]);s['body_q']=q.tolist()
        if done:break
    assert done and executor.phase=='hold'
    assert [e['phase'] for e in executor.events]==['close','lift','hold']
    assert max(np.diff(positions))/.02<=.040001
    assert max(positions)==pytest.approx(.24,abs=1e-6)
    np.testing.assert_allclose(command[15:17],[.3,.1])
    assert grip[0]==0
    assert s['simulation_time']-executor.hold_started>=6


def test_lift_never_passes_if_measured_hand_does_not_follow(setup):
    executor,selector=setup;s=sensor(0)
    executor.consider(s,selector,[4.5,4.5],[0,0,.13]);s['simulation_time']=.11
    executor.consider(s,selector,[4.5,4.5],[0,0,.13]);s['jaw_force_n']=[[2,2],[0,0]]
    with pytest.raises(RuntimeError,match='failed to reach'):
        for step in range(900):
            s['simulation_time']=.12+step*.02;executor.command(s)
    assert not executor.done


def test_visual_residual_is_bounded_and_requires_fresh_rgb(setup):
    executor,selector=setup;executor.options['visual_residual_m']=.10
    selector.safe_open=np.array([4.5,4.5]);selector.target_world=np.array([.36,.1,.10])
    s=sensor(0)
    assert executor.consider(s,selector,[4.5,4.5],[0,0,.13])
    assert executor.phase=='align'
    previous=executor.world_pose[:3,3].copy()
    for i in range(1,20):
        s['simulation_time']=i*.02
        executor.command(s)
        current=executor.world_pose[:3,3].copy()
        assert np.linalg.norm(current-previous)<=.020001*.02
        previous=current
    assert executor.phase=='align'  # measured hand has not followed the command
    selector.last_seen=time.monotonic()-3
    with pytest.raises(RuntimeError,match='stale'):executor.command(s)


def test_visual_residual_rejects_large_target_shift(setup):
    executor,selector=setup;executor.options['visual_residual_m']=.1
    selector.safe_open=np.array([4.5,4.5]);selector.target_world=np.array([.36,.1,.1])
    s=sensor(0);executor.consider(s,selector,[4.5,4.5],[0,0,.13])
    selector.target_world=np.array([.5,.1,.1])
    with pytest.raises(RuntimeError,match='workspace'):executor.command(s)


def test_clearance_approach_reaches_measured_alignment_despite_robot_sag(setup):
    executor,selector=setup
    executor.options.update(visual_residual_m=.12,clearance_approach=True,tracking_bias_limit_m=.015,
                            alignment_height_m=.014,alignment_timeout_seconds=20)
    selector.target_world=np.array([.36,.1,.10]);s=sensor(0)
    assert executor.consider(s,selector,[4.5,4.5],[0,0,.13])
    max_z=0
    for step in range(1,1001):
        s['simulation_time']=step*.02
        command,grip,_=executor.command(s)
        max_z=max(max_z,command[17])
        if executor.phase=='close':break
        assert grip[0]==executor.config['gripper_open_value'][0]
        q=np.array(s['body_q']);q[15:18]=command[15:18]-[0,0,.008];s['body_q']=q.tolist()
    assert executor.phase=='close' and max_z>=.16
    assert [e['phase'] for e in executor.events]==['align','align_hover','align_descend','close']
    assert np.linalg.norm(executor.tracking_bias)<=.015001
    assert np.linalg.norm(np.asarray(s['body_q'])[15:17]-selector.target_world[:2])<.01


def precision_executor(setup):
    executor,selector=setup
    config=load_config('robot_client/sonic/vla_rgb_precision.yaml')
    executor=module.GraspExecution('unused',config)
    selector.target_world=np.array([.3,.1,.092])
    s=sensor(0)
    executor.consider(s,selector,[5.5,5.5],[0,0,.13])
    executor.align_stage='descend'
    # Fixture FK has identity orientation; keep the desired rotation matched.
    executor.world_pose[:3,:3]=np.eye(3)
    return executor,selector,s


def test_precision_alignment_waits_for_motion_to_settle(setup):
    executor,selector,s=precision_executor(setup)
    for i in range(1,51):
        s['simulation_time']=i*.02
        s['body_q'][15]=.3+(.003 if i%2 else -.003)
        executor.command(s)
        assert executor.phase=='align'  # Always inside the old position gate.
    for i in range(51,100):
        s['simulation_time']=i*.02;s['body_q'][15]=.3
        executor.command(s)
        if executor.phase=='close':break
    assert executor.phase=='close'
    assert executor.events[-1]['measured_speed_mps']<=.012


def test_precision_alignment_requires_orientation_and_target_height(setup):
    executor,selector,s=precision_executor(setup)
    executor.world_pose[:3,:3]=module.Rotation.from_euler('y',.2).as_matrix()
    for i in range(1,40):
        s['simulation_time']=i*.02;executor.command(s)
    assert executor.phase=='align'
    executor.world_pose[:3,:3]=np.eye(3)
    s['body_q'][17]=.107  # Old gate accepts this shallow 15 mm pinch.
    for i in range(40,80):
        s['simulation_time']=i*.02;executor.command(s)
    assert executor.phase=='align'


def test_final_support_loss_cannot_be_reported_as_complete(setup):
    executor,selector=setup;s=sensor(0)
    executor.consider(s,selector,[4.5,4.5],[0,0,.13]);s['simulation_time']=.11
    executor.consider(s,selector,[4.5,4.5],[0,0,.13])
    executor.phase='hold';executor.hold_started=0
    s['simulation_time']=6.01
    assert not executor.command(s)[2]
    s['simulation_time']=6.22
    with pytest.raises(RuntimeError,match='support lost'):executor.command(s)


@pytest.mark.parametrize('key,value',[('speed_mps',float('nan')),('window_seconds',0),('position_tolerance_m',.1)])
def test_invalid_precision_gate_rejected(setup,key,value):
    config=load_config('robot_client/sonic/vla_rgb_precision.yaml')
    config['grasp_execution']['alignment_settle'][key]=value
    with pytest.raises(ValueError,match='alignment_settle'):module.GraspExecution('unused',config)


def test_descent_lock_ignores_small_visual_bias_but_rejects_motion_and_staleness(setup):
    executor,selector,s=precision_executor(setup)
    executor.descent_target=selector.target_world.copy()
    selector.target_world=selector.target_world+np.array([.006,0,0])
    executor.command(s)
    assert executor.alignment_status['goal_error_m'][0]==pytest.approx(0)
    selector.target_world=executor.descent_target+np.array([.016,0,0])
    with pytest.raises(RuntimeError,match='moved after descent'):executor.command(s)
    selector.target_world=executor.descent_target.copy();selector.last_seen=time.monotonic()-3
    with pytest.raises(RuntimeError,match='stale'):executor.command(s)


def simulate_precision_descent(setup,settle_only,offset_m=0.,shift_target=False):
    """A delayed robot, not instantaneous command tracking, exposes windup."""
    _,selector=setup
    config=load_config('robot_client/sonic/vla_rgb_precision_fast.yaml')
    config['grasp_execution'].update(clearance_approach=False,tracking_bias_settle_only=settle_only,
                                    lock_descent_target=not shift_target)
    executor=module.GraspExecution('unused',config)
    selector.target_world=np.array([.3,.1,.10])
    s=sensor(0);s['body_q'][17]=.16
    assert executor.consider(s,selector,[5.5,5.5],[0,0,.13])
    executor.world_pose[:3,:3]=np.eye(3)
    positions=[];commands=[]
    for i in range(1,1001):
        s['simulation_time']=i*.02
        if shift_target and i==75:selector.target_world[0]+=.008
        selector.last_seen=time.monotonic()
        command,grip,_=executor.command(s)
        positions.append(np.array(s['body_q'][15:18]));commands.append(command[15:18].copy())
        if executor.phase=='close':break
        assert grip[0]==5.5
        # 300 ms actuator lag plus optional persistent height tracking error.
        actual=np.array(s['body_q'][15:18])
        actual+=(.02/.3)*(command[15:18]+[0,0,offset_m]-actual)
        s['body_q'][15:18]=actual.tolist()
    assert executor.phase=='close'
    return executor,np.asarray(positions),np.asarray(commands),s['simulation_time']


def test_arrival_gated_bias_reduces_transit_overshoot_without_relaxing_precision(setup):
    slow,slow_positions,_,slow_time=simulate_precision_descent(setup,False)
    fast,fast_positions,commands,fast_time=simulate_precision_descent(setup,True)
    assert fast_time<slow_time-.5
    assert max(0,.110-fast_positions[:,2].min())<max(0,.110-slow_positions[:,2].min())
    # The final packet switches to the existing measured-pose close anchor;
    # all preceding packets are the rate-limited visual alignment trajectory.
    assert np.max(np.linalg.norm(np.diff(commands[:-1],axis=0),axis=1))<=.020001*.02
    assert fast.settle==slow.settle
    assert fast.events[-1]['measured_speed_mps']<=.012
    np.testing.assert_allclose(fast_positions[-1],[.3,.1,.110],atol=.004)


def test_arrival_gated_bias_still_compensates_static_error_and_changed_goal(setup):
    executor,positions,_,_=simulate_precision_descent(setup,True,offset_m=.008,shift_target=True)
    # More than the 2 mm arrival gate is needed: comparing against the raw goal
    # would strand this controller before it compensates the physical offset.
    assert executor.tracking_bias[2]<-.003
    assert np.linalg.norm(executor.tracking_bias)<=.015001
    np.testing.assert_allclose(positions[-1],[.308,.1,.110],atol=.004)
