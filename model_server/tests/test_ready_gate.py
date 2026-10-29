import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_client.sonic.adapter import pose_to_rpy
from robot_client.sonic.kinematics import G1Kinematics
from robot_client.sonic.ready import ReadyGate


class StaticReadyKinematics:
    """Keep task-independent posture perfect to isolate robot odometry gates."""
    def forward(self,q):
        return [np.eye(4),np.eye(4)]

    def solve(self,targets,seed,**kwargs):
        return seed.copy()


def orientation_gate(**options):
    return ReadyGate(StaticReadyKinematics(),{
        'ready_pose':[[0.]*6]*2,
        'ready_gate':{'yaw_tolerance_rad':np.deg2rad(2.),
                      'angular_speed_tolerance_rps':.05,'speed_tolerance_mps':.02,
                      'stable_sim_seconds':.5,**options}})


def ready_sensor(time,yaw=0.,pitch=0.):
    quat=Rotation.from_euler('xyz',[0,pitch,yaw]).as_quat()
    return {'simulation_time':time,'base_position':[0,0,.75],
            'initial_base_position':[0,0,.75],'base_quat':quat[[3,0,1,2]].tolist()}


def test_orientation_gate_waits_for_existing_heading_without_changing_commands():
    gate=orientation_gate();q=np.zeros(29)
    for time in np.arange(0,.8,.1):
        command=gate.update(q,q,ready_sensor(time,yaw=np.deg2rad(5),pitch=.2),target_base_yaw=0.)
        np.testing.assert_array_equal(command,q)
    assert not gate.completed
    assert gate.metrics['base_yaw_error_rad']==pytest.approx(np.deg2rad(5))
    # The gate is a check, not a zero-tilt posture controller.
    for time in np.arange(.8,1.5,.1):
        command=gate.update(q,q,ready_sensor(time,yaw=np.deg2rad(5),pitch=.2),target_base_yaw=np.deg2rad(5))
    assert gate.completed
    assert gate.metrics['base_angular_speed_rps']==pytest.approx(0.)
    assert gate.metrics['angular_speed_tolerance_rps']==.05
    np.testing.assert_array_equal(command,q)


def test_yaw_error_wraps_across_pi():
    gate=orientation_gate();q=np.zeros(29)
    for time in np.arange(0,.8,.1):
        gate.update(q,q,ready_sensor(time,yaw=-np.pi+.01),target_base_yaw=np.pi-.01)
    assert gate.completed
    assert gate.metrics['base_yaw_error_rad']==pytest.approx(.02)


def test_full_quaternion_speed_detects_pitch_motion_with_stationary_yaw():
    gate=orientation_gate();q=np.zeros(29)
    for time in np.arange(0,1.1,.1):
        gate.update(q,q,ready_sensor(time,pitch=.1*time),target_base_yaw=0.)
    assert not gate.completed
    assert gate.metrics['base_yaw_error_rad']==0.
    assert gate.metrics['base_angular_speed_rps']==pytest.approx(.1)
    for time in np.arange(1.1,1.8,.1):
        gate.update(q,q,ready_sensor(time,pitch=.1),target_base_yaw=0.)
    assert gate.completed


def test_duplicate_timestamps_do_not_manufacture_orientation_stability():
    gate=orientation_gate();q=np.zeros(29)
    for time in [0.,.1]+[.1]*20:
        gate.update(q,q,ready_sensor(time),target_base_yaw=0.)
    assert not gate.completed and gate.since==pytest.approx(.1)
    gate.update(q,q,ready_sensor(.1,yaw=.01),target_base_yaw=0.)
    assert not gate.completed and gate.since is None
    assert gate.metrics['base_angular_speed_rps'] is None
    for time in [.2,.3,.4,.5,.6,.7,.81]:
        gate.update(q,q,ready_sensor(time,yaw=.01),target_base_yaw=0.)
    assert gate.completed
    gate.update(q,q,ready_sensor(.81,yaw=.01),target_base_yaw=0.)
    assert gate.completed


def test_clock_rollback_restarts_the_whole_stability_window():
    gate=orientation_gate();q=np.zeros(29)
    for time in [0.,.1,.2,.3,.4,.5,.6,.7]:
        gate.update(q,q,ready_sensor(time),target_base_yaw=0.)
    assert gate.completed
    gate.update(q,q,ready_sensor(.1),target_base_yaw=0.)
    assert not gate.completed and gate.since is None
    assert gate.metrics['clock_restarted']
    assert gate.metrics['base_speed_mps'] is None
    for time in [.2,.3,.4,.5,.6]:
        gate.update(q,q,ready_sensor(time),target_base_yaw=0.)
    assert not gate.completed
    gate.update(q,q,ready_sensor(.71),target_base_yaw=0.)
    assert gate.completed


@pytest.mark.parametrize('quaternion',[None,[0,0,0,0],[2,0,0,0],[1,np.nan,0,0]])
def test_orientation_gate_rejects_missing_or_invalid_quaternion(quaternion):
    gate=orientation_gate();sensor=ready_sensor(0.)
    if quaternion is None:sensor.pop('base_quat')
    else:sensor['base_quat']=quaternion
    with pytest.raises(ValueError,match='base_quat'):
        gate.update(np.zeros(29),np.zeros(29),sensor,target_base_yaw=0.)


@pytest.mark.parametrize('target',[None,np.nan,np.inf])
def test_yaw_gate_requires_explicit_finite_target(target):
    with pytest.raises(ValueError,match='target_base_yaw'):
        orientation_gate().update(np.zeros(29),np.zeros(29),ready_sensor(0.),target_base_yaw=target)


def test_legacy_ready_configuration_does_not_require_orientation_feedback():
    gate=orientation_gate(yaw_tolerance_rad=None,angular_speed_tolerance_rps=None)
    for time in [0.,.1,.2,.3,.4,.5,.6,.7]:
        sensor=ready_sensor(time);sensor.pop('base_quat')
        gate.update(np.zeros(29),np.zeros(29),sensor)
    assert gate.completed
    assert gate.metrics['base_yaw_rad'] is None


def test_ready_gate_requires_measured_stability_and_accepts_repeated_fresh_samples():
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for actual G1 kinematics')
    kin=G1Kinematics(Path(root))
    q=np.zeros(29)
    q[[3,9]]=.2
    q[[18,25]]=.6
    config={'ready_pose':[pose_to_rpy(t) for t in kin.forward(q)],'ready_gate':{'enabled':True}}
    gate=ReadyGate(kin,config)
    sensor={'simulation_time':0.,'base_position':[0,0,.75],'initial_base_position':[0,0,.75]}
    for t in [0,.1,.1,.2,.2,.3,.4,.5,.61]:
        sensor['simulation_time']=t
        command=gate.update(q,q,sensor)
    assert gate.completed
    np.testing.assert_allclose(command,q,atol=1e-5)
    # Being at the ready joint pose while the base moves is insufficient.
    sensor.update(simulation_time=.7,base_position=[.1,0,.75])
    gate.update(q,q,sensor)
    assert not gate.completed
    assert gate.metrics['base_error_m']==pytest.approx(.1)
    sensor.update(simulation_time=.8,base_position=[0,0,.75])
    wrong_waist=q.copy()
    wrong_waist[12]=.2
    gate.update(wrong_waist,q,sensor)
    assert not gate.completed
    assert gate.metrics['waist_error_rad']==pytest.approx(.2)


def test_recorded_reset_seed_keeps_ready_nullspace_consistent():
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for actual G1 kinematics')
    kin=G1Kinematics(Path(root))
    target=np.zeros(29);target[[3,9]]=.2;target[[18,25]]=.6
    config={'ready_pose':[pose_to_rpy(t) for t in kin.forward(target)],
            'ready_arm_seed':target[15:].tolist(),'ready_gate':{'enabled':True}}
    sensor={'simulation_time':0.,'base_position':[0,0,.75],'initial_base_position':[0,0,.75]}
    a=target.copy();a[15]+=.05
    b=target.copy();b[15]-=.05
    first=ReadyGate(kin,config).update(a,target,sensor)
    second=ReadyGate(kin,config).update(b,target,sensor)
    np.testing.assert_allclose(first[15:],second[15:],atol=1e-5)
    np.testing.assert_allclose(first[15:],target[15:],atol=1e-5)
