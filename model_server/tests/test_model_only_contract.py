"""A model-only task policy must not silently inherit grasp interventions."""
import asyncio
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_client.sonic import run as client
from robot_client.sonic.kinematics import G1Kinematics


def test_calibrated_config_keeps_model_task_ownership_and_records_mechanical_mapping():
    config=client.load_config(Path(client.__file__).with_name('vla_model_only_calibrated.yaml'))
    contract=client.execution_contract(config)
    assert contract['mode']=='model_only' and contract['task_assists_enabled']==[]
    np.testing.assert_allclose(contract['tool_offsets'],[[.105,0,0,0,0,0]]*2)
    assert contract['gripper_mapping']['open_motor_value']==[5.4,5.4]
    assert contract['gripper_mapping']['closed_motor_value']==[0,0]
    assert contract['limits']['planner']['minimum_walking_speed_mps']==.1
    assert not contract['base_motion_compensation']
    assert contract['calibration_validation']['motor_closed_rad']==0
    config['tool_offsets'][0][0]=.12
    with pytest.raises(ValueError,match='verified Dex1 calibration'):
        client.execution_contract(config)


@pytest.mark.parametrize('override',[
    {'visual_target':{'candidate_count':8}},
    {'grasp_execution':{'hold_seconds':6}},
    {'grasp_hold':{'enabled':True}},
    {'grasp_lift_reflex':{'target_lift_m':.14}},
    {'grasp_hold_force':{'target_n':4}},
    {'stationary_single_arm':True},
    {'base_position_hold':True},
    {'align_first_ee_frame':True},
    {'cartesian_tracking':True},
    {'action_time_scale':1.5},
    {'encoder_generation_prompt':True},
])
def test_model_only_rejects_inherited_interventions(tmp_path,override):
    parent=tmp_path/'parent.yaml'
    parent.write_text(yaml.safe_dump(dict(base_config=str(Path(client.__file__).with_name('sim.yaml')),**override)))
    child=tmp_path/'child.yaml'
    child.write_text(yaml.safe_dump({'base_config':'parent.yaml','execution_mode':'model_only'}))
    with pytest.raises(ValueError,match='model_only'):
        client.load_config(child)


def test_model_only_revalidates_after_callers_override_loaded_config(monkeypatch):
    config=client.load_config(Path(client.__file__).with_name('sim.yaml'))
    config.update(execution_mode='model_only',stationary_single_arm=True)
    def unexpected_construction(*args,**kwargs):
        raise AssertionError('must reject before creating execution resources')
    monkeypatch.setattr(client,'G1Kinematics',unexpected_construction)
    with pytest.raises(ValueError,match='stationary_single_arm'):
        asyncio.run(client.run(config,Path('unused'),.1))


def test_robot_feedback_does_not_require_or_expose_tactile_task_control():
    config={'execution_mode':'model_only','sonic_mode':'planner',
            'grasp_hold':{'enabled':False},'ready_gate':{'enabled':True},
            'robot_sensor_endpoint':'tcp://127.0.0.1:15558'}
    contract=client.execution_contract(config)
    assert contract['validated'] and contract['task_assists_enabled']==[]
    assert contract['official_wbc_client'] is False
    assert contract['leg_reference_source']=='sonic_planner'
    sensor={'simulation_time':1.,'body_q':[0.]*29,'base_position':[0,0,.75],
            'base_quat':[1,0,0,0],'initial_base_position':[0,0,.75],
            'jaw_force_n':[[8,8],[0,0]],'objects':{'target_pose':[1,2,3]},
            'jaw_contacts':[0,1]}
    feedback=client.robot_only_feedback(sensor)
    assert set(feedback)==set(contract['robot_feedback_used'])
    np.testing.assert_allclose(client.robot_base_transform(feedback)[:3,3],[0,0,.75])
    del config['robot_sensor_endpoint']
    with pytest.raises(ValueError,match='robot_sensor_endpoint'):
        client.execution_contract(config)


@pytest.mark.parametrize('change',[
    {'compensate_base_motion':True},
    {'sonic_mode':'joint','waist_tracking_compensation':True},
])
def test_unsupported_robot_frame_compensation_fails_closed(change):
    config={'execution_mode':'model_only','sonic_mode':'planner',
            'robot_sensor_endpoint':'tcp://127.0.0.1:15558',**change}
    with pytest.raises(ValueError,match='model_only'):
        client.execution_contract(config)


def test_external_safety_signal_stops_without_reading_task_data(tmp_path):
    stop_file=tmp_path/'safety-stop.json'
    config={'execution_mode':'model_only','safety_stop_file':str(stop_file)}
    client.emergency_stop_check(config)
    # Even an incomplete/non-JSON supervisor write must trigger the stop.
    stop_file.write_text('incomplete supervisor message')
    with pytest.raises(RuntimeError,match='external safety stop'):
        client.emergency_stop_check(config)


@pytest.mark.parametrize('height,tilt,stops',[
    (.75,0.,False),(.501,24.9,False),(.5,0.,True),(.49,0.,True),(.75,25.1,True),
])
def test_model_only_robot_pose_safety_bounds(height,tilt,stops):
    angle=np.deg2rad(tilt)/2
    feedback={'base_position':[0,0,height],
              'base_quat':[np.cos(angle),np.sin(angle),0,0]}
    config={'execution_mode':'model_only'}
    if stops:
        with pytest.raises(RuntimeError,match='robot pose safety stop'):
            client.emergency_stop_check(config,feedback)
    else:client.emergency_stop_check(config,feedback)
    # Preserve the legacy executor's behavior unless a supervisor stop exists.
    client.emergency_stop_check({'execution_mode':'legacy'},feedback)


def test_two_arm_compensation_preserves_model_goals_and_waist():
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for real G1 kinematics')
    kin=G1Kinematics(Path(root))
    nominal=np.zeros(29)
    nominal[[0,4,6,10]]=-.1;nominal[[3,9]]=.2;nominal[[18,25]]=.5
    nominal[12:15]=[.03,.02,.1]
    measured=nominal.copy();measured[12:15]=[0,0,0]
    before=nominal.copy();targets=kin.forward(nominal)
    result=client.compensate_model_targets(kin,nominal,measured)
    np.testing.assert_array_equal(nominal,before)
    np.testing.assert_array_equal(result[:15],nominal[:15])
    assert np.linalg.norm(result[15:22]-nominal[15:22])>1e-3
    assert np.linalg.norm(result[22:]-nominal[22:])>1e-3
    physical=result.copy();physical[:15]=measured[:15]
    actual=kin.forward(physical)
    for goal,pose in zip(targets,actual):
        assert np.linalg.norm(goal[:3,3]-pose[:3,3])<.01


def test_model_only_loop_keeps_both_model_grippers_and_reports_provenance(tmp_path,monkeypatch):
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for real G1/socket integration')
    fixture=importlib.import_module('robot_client.sonic.smoke')
    original=fixture.fixture_action
    def two_grippers(kin,q0,frames=30):
        action=original(kin,q0,frames)
        action['action.left_gripper'][0,:,0]=np.linspace(1.,4.,frames)
        action['action.right_gripper'][0,:,0]=np.linspace(4.,1.,frames)
        return action
    monkeypatch.setattr(fixture,'fixture_action',two_grippers)
    report=asyncio.run(fixture.smoke(Path(root),seconds=2.5,config_overrides={
        'execution_mode':'model_only','trace_control':True,'trace_dir':str(tmp_path/'actions')}))
    assert report['execution_mode']=='model_only'
    assert report['policy_source']==client.MODEL_ONLY_SOURCE
    assert report['execution_contract']['validated']
    assert report['execution_contract']['task_assists_enabled']==[]
    assert report['grasp_execution'] is None and report['grasp_hold'] is None
    assert report['visual_selection'] is None and not report['stationary_single_arm']
    rows=[json.loads(line) for line in (tmp_path/'control.jsonl').read_text().splitlines()]
    assert {r['phase'] for r in rows}=={'vla'}
    assert all(r['accepted_model_chunks']>0 for r in rows)
    grips=np.asarray([r['command_grippers'] for r in rows])
    assert np.ptp(grips[:,0])>1 and np.ptp(grips[:,1])>1
    assert np.max(np.abs(grips[:,0]-grips[:,1]))>1


def test_aborted_model_only_loop_retains_validated_provenance(monkeypatch):
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for real G1/socket integration')
    fixture=importlib.import_module('robot_client.sonic.smoke')
    original=fixture.fixture_action
    def bad_action(kin,q0,frames=30):
        action=original(kin,q0,frames)
        action['action.left_gripper'][0,0,0]=99
        return action
    monkeypatch.setattr(fixture,'fixture_action',bad_action)
    with pytest.raises(ValueError,match='gripper action outside') as error:
        asyncio.run(fixture.smoke(Path(root),seconds=2.,config_overrides={'execution_mode':'model_only'}))
    report=error.value.client_report
    assert report['execution_contract']['validated']
    assert report['execution_contract']['task_assists_enabled']==[]
    assert report['execution_mode']=='model_only'
    assert report['policy_source']==client.MODEL_ONLY_SOURCE
    assert report['model_requests']>0


def test_external_stop_in_model_only_loop_preserves_failure_report(tmp_path):
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for real G1/socket integration')
    fixture=importlib.import_module('robot_client.sonic.smoke')
    stop_file=tmp_path/'safety-stop.json';stop_file.write_text('{}')
    with pytest.raises(RuntimeError,match='external safety stop') as error:
        asyncio.run(fixture.smoke(Path(root),seconds=.5,config_overrides={
            'execution_mode':'model_only','safety_stop_file':str(stop_file)}))
    report=error.value.client_report
    assert report['execution_mode']=='model_only'
    assert report['execution_contract']['limits']['safety_stop_file']==str(stop_file)
    assert report['packets']==0 and report['model_requests']==0
