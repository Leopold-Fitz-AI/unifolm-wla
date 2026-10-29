from pathlib import Path
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_client.sonic.execution import StationaryArmExecution, instruction_hand
from robot_client.sonic.kinematics import G1Kinematics
from robot_client.sonic.run import load_config
from robot_client.sonic.smoke import fixture_action
from robot_client.sonic.adapter import ActionAdapter


@pytest.fixture
def sonic_root():
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for actual G1 kinematics')
    return Path(root)


@pytest.mark.parametrize('instruction,hand',[('pick up red with your left hand',0),('用右手拿起黄块',1)])
def test_hand_scope(instruction,hand):
    assert instruction_hand(instruction)==hand


@pytest.mark.parametrize('instruction',['pick up the cube','left hand and right hand'])
def test_ambiguous_scope_rejected(instruction):
    with pytest.raises(ValueError,match='explicit'):
        instruction_hand(instruction)


@pytest.mark.parametrize('hand',[0,1])
def test_stationary_scope_preserves_active_cartesian_goals_and_safe_ik(sonic_root,hand):
    kin=G1Kinematics(sonic_root)
    q=np.zeros(29)
    q[[0,4,6,10]]=-.1
    q[[3,9]]=.2
    q[[18,25]]=.5
    action=fixture_action(kin,q)
    original={k:v.copy() for k,v in action.items()}
    side=('left','right')[hand]
    other=('right','left')[hand]
    executor=StationaryArmExecution(kin,q,[.7,.8],f'pick up cube with your {side} hand',.75)
    constrained=executor.constrain(action)
    np.testing.assert_array_equal(constrained[f'action.{side}_ee_rpy'],action[f'action.{side}_ee_rpy'])
    np.testing.assert_array_equal(constrained[f'action.{side}_gripper'],action[f'action.{side}_gripper'])
    for k in action:np.testing.assert_array_equal(action[k],original[k])
    np.testing.assert_allclose(constrained['action.lower_body'][0,:,12:15],np.tile(q[12:15],(30,1)))
    np.testing.assert_allclose(constrained['action.base_command'][0],np.tile([0,0,0,.75],(30,1)))
    np.testing.assert_allclose(constrained[f'action.{other}_gripper'],[.7,.8][1-hand])
    adapter=ActionAdapter(kin)
    adapter.planner_mode=True
    ref=executor.freeze_unused_joints(adapter.convert(constrained,q,[1,0,0,0],[.7,.8],0))
    start=15+7*(1-hand)
    np.testing.assert_array_equal(ref.q[:,start:start+7],np.tile(q[start:start+7],(len(ref.q),1)))
    assert ref.max_ik_position_error<.005

    # A lagging waist must not translate the selected learned EE goal. The
    # arm command compensates measured waist angles, without error integration.
    measured=q.copy()
    measured[13:15]=[.04,-.05]
    compensated=executor.compensate_waist(q,measured)
    actual=compensated.copy()
    actual[:15]=measured[:15]
    expected=kin.forward(q)[hand]
    np.testing.assert_allclose(kin.forward(actual)[hand][:3,3],expected[:3,3],atol=.005)
    np.testing.assert_array_equal(compensated[:15],q[:15])
    source=np.eye(4)
    source[2,3]=.75
    current=source.copy()
    current[:3,:3]=Rotation.from_euler('z',.02).as_matrix()
    current[:3,3]+=[.01,-.015,.01]
    transferred=executor.compensate_waist(q,measured,source,current)
    actual=transferred.copy()
    actual[:15]=measured[:15]
    desired_world=source@kin.forward(q)[hand]
    actual_world=current@kin.forward(actual)[hand]
    np.testing.assert_allclose(actual_world[:3,3],desired_world[:3,3],atol=.005)


def test_stationary_config_retains_tactile_hold():
    cfg=load_config(Path(__file__).parents[2]/'robot_client/sonic/vla_stationary.yaml')
    assert cfg['stationary_single_arm'] and cfg['grasp_hold']['enabled']


def test_prepared_waist_reference_is_continuous_through_vla_handoff(sonic_root):
    kin=G1Kinematics(sonic_root)
    q=np.zeros(29)
    q[[3,9]]=.2
    q[[18,25]]=.6
    q[14]=.03  # Native balance need not equal its commanded waist posture.
    waist=np.array([0,0,.136])
    action=fixture_action(kin,q)
    executor=StationaryArmExecution(kin,q,[.7,.8],'pick up red with your left hand',.75,waist)
    adapter=ActionAdapter(kin)
    adapter.planner_mode=True
    adapter.limit_nominal_velocity=False
    constrained=executor.constrain(action)
    reference=executor.freeze_unused_joints(adapter.convert(constrained,q,[1,0,0,0],[.7,.8],0))
    np.testing.assert_array_equal(reference.q[:,12:15],np.tile(waist,(len(reference.q),1)))
    compensated=executor.compensate_waist(reference.q[0],q)
    np.testing.assert_array_equal(compensated[12:15],waist)
    actual=compensated.copy()
    actual[12:15]=q[12:15]
    expected=action['action.left_ee_rpy'][0,0,:3]
    np.testing.assert_allclose(kin.forward(actual)[0][:3,3],expected,atol=.005)
