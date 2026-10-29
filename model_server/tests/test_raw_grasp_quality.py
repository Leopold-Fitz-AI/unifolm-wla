import json
import subprocess
import sys

import numpy as np
import pytest
import yaml

from robot_client.sonic.raw_grasp_quality import audit_trial, measure_samples


def samples(seconds=6.):
    return [{'simulation_time': round(i*.02, 8), 'suspension_enabled': False,
             'base_height': .79, 'base_tilt_deg': 0.,
             'hands': {'left': {'tcp_position': [0, 0, 1.],
                                'wrist_rotation': np.eye(3).ravel().tolist()}},
             'objects': {'cube': {'position': [0, 0, 1.], 'bottom_above_table': .12,
                                   'jaw_contacts': {'left': [0, 1]}, 'table_contact': False},
                         'other': {'position': [.2, 0, .85], 'bottom_above_table': 0.,
                                   'jaw_contacts': {'left': []}, 'table_contact': True}}}
            for i in range(round(seconds/.02)+1)]


def measure(rows, **kwargs):
    return measure_samples(rows, 'cube', 'left', evaluation_start=kwargs.pop('evaluation_start', 0.), **kwargs)


def preparation_samples():
    rows=samples(.18)
    for row in rows:
        row['simulation_time']=round(row['simulation_time']-.2,8)
        obj=row['objects']['cube']
        obj.update(position=[0,0,.85],bottom_above_table=0.,
                   jaw_contacts={'left':[]},table_contact=True)
    return rows


def test_rigid_common_motion_has_no_relative_slip():
    rows = samples()
    for i, row in enumerate(rows):
        row['hands']['left']['tcp_position'][0] = i*.001
        row['objects']['cube']['position'][0] = i*.001
    report = measure(rows)
    assert report['two_second_support_observed']
    assert report['longest_supported_seconds'] == 6
    assert report['first_support_window_passed']
    assert report['first_support_window']['maximum_local_slip_m'] == 0


def test_first_support_window_cannot_be_replaced_by_later_success():
    rows = samples(8.)
    for row in rows:
        if .2 <= row['simulation_time'] <= .4:
            row['objects']['cube']['jaw_contacts']['left'] = []
    report = measure(rows)
    assert len(report['support_segments']) == 2
    assert report['six_second_support_segment_observed']
    assert report['first_support_start'] == 0
    assert not report['first_support_window_passed']
    assert 'bilateral_support_lost' in report['reasons']


def test_startup_grasp_does_not_count_as_model_execution():
    rows = samples(8.)
    for row in rows:
        if row['simulation_time'] >= 2:
            row['objects']['cube']['jaw_contacts']['left'] = []
    report = measure(rows, evaluation_start=2.)
    assert not report['two_second_support_observed']
    assert report['first_support_window'] is None
    assert report['maximum_supported_height_m'] is None


def test_observation_gap_breaks_support_and_rejects_quality():
    rows = [r for r in samples() if not 1.9 <= r['simulation_time'] <= 2.1]
    report = measure(rows)
    assert len(report['support_segments']) == 2
    assert report['longest_supported_seconds'] < 4
    assert not report['first_support_window_passed']
    assert 'incomplete_hold_coverage' in report['reasons']


def test_stream_ending_just_before_six_seconds_is_incomplete():
    report = measure(samples()[:-1])
    assert report['first_support_window']['maximum_sample_gap_seconds'] < .06
    assert not report['first_support_window_passed']
    assert 'recording_ends_before_requested_window' in report['reasons']


def test_slow_slip_passes_support_but_fails_quality():
    rows = samples()
    for i, row in enumerate(rows):
        row['objects']['cube']['position'][2] -= .012*i/300
    report = measure(rows)
    assert report['two_second_support_observed']
    assert report['first_support_window']['maximum_local_slip_m'] == pytest.approx(.012)
    assert not report['first_support_window_passed']


def test_clock_reset_is_not_a_new_attempt():
    rows = samples()
    rows[-1]['simulation_time'] = 0
    report = measure(rows)
    assert report['simulation_clock_reset']
    assert not report['physics_integrity_passed']
    assert report['longest_supported_seconds'] is None
    assert report['first_support_window'] is None


@pytest.mark.parametrize('field,value', [('suspension_enabled', True), ('base_height', .4), ('base_tilt_deg', 30)])
def test_external_support_and_unsafe_robot_pose_do_not_count(field, value):
    rows = samples()
    for row in rows:
        row[field] = value
    assert not measure(rows)['two_second_support_observed']


@pytest.fixture
def trial(tmp_path):
    (tmp_path/'task.yaml').write_text(yaml.safe_dump({
        'target': 'cube', 'required_hand': 'left', 'lift_height': .08,
        'robot_contacts_from_start': True}))
    (tmp_path/'client-config.yaml').write_text(yaml.safe_dump({'execution_mode': 'model_only'}))
    client = {'status': 'passed', 'chunks': 1, 'execution_mode': 'model_only',
              'policy_source': 'unifolm_model_only_sonic_adapter',
              'execution_contract': {'mode': 'model_only', 'validated': True,
                                     'task_assists_enabled': [], 'official_wbc_client': False}}
    (tmp_path/'client.json').write_text(json.dumps(client))
    collision = {'passed': True, 'enabled_from_start': True, 'complete_step_coverage': True,
                 'collider_manifest_valid': True, 'checked_physics_steps': 1201,
                 'required_robot_collision_geoms': 45, 'disabled_steps': 0,
                 'invalid_contacts': 0, 'clock_reset': False, 'maximum_penetration_m': 0.}
    # The legacy task scorer's success value is deliberately not authoritative.
    (tmp_path/'task-report.json').write_text(json.dumps({'success': False,
        'simulation_reset': False, 'table_collision_audit': collision}))
    (tmp_path/'physics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in preparation_samples()+samples()))
    (tmp_path/'control.jsonl').write_text(json.dumps({'phase': 'vla', 'simulation_time': 0.})+'\n')
    actions = tmp_path/'actions'; actions.mkdir()
    reply = {'action.left_ee_rpy': np.zeros((1, 30, 6)), 'action.right_ee_rpy': np.zeros((1, 30, 6)),
             'action.left_gripper': np.zeros((1, 30, 1)), 'action.right_gripper': np.zeros((1, 30, 1)),
             'action.lower_body': np.zeros((1, 30, 15)), 'action.base_command': np.zeros((1, 30, 4))}
    np.savez(actions/'reply-001.npz', **reply,camera_capture_simulation_time=-.02)
    return tmp_path


def change_json(path, name, **kwargs):
    file = path/name
    result = json.loads(file.read_text()); result.update(kwargs)
    file.write_text(json.dumps(result))


def test_unassisted_physics_can_pass_without_grasp_execution_events(trial):
    report = audit_trial(trial)
    assert report['execution_completed'] and report['task_passed'] and report['quality_passed']
    assert report['model_evidence']['raw_model_replies'] == 1
    assert not report['model_evidence']['execution_contract']['official_wbc_client']
    assert report['preparation_evidence']['verified']


@pytest.mark.parametrize('grasp,lift',[(True,False),(False,True),(True,True)])
def test_preparation_grasp_or_lift_cannot_be_attributed_to_model(trial,grasp,lift):
    rows=preparation_samples()+samples()
    obj=rows[2]['objects']['cube']
    if grasp:obj['jaw_contacts']['left']=[0,1]
    if lift:obj['bottom_above_table']=.1
    (trial/'physics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=audit_trial(trial)
    assert report['two_second_support_observed'] and report['first_support_window_passed']
    assert not report['task_passed'] and not report['quality_passed']
    assert not report['preparation_evidence']['verified']
    if grasp:assert 'target_already_grasped_during_preparation' in report['reasons']
    if lift:assert 'target_already_lifted_during_preparation' in report['reasons']


def test_grasp_during_unlogged_reply_wait_is_not_attributable(trial):
    (trial/'control.jsonl').write_text(json.dumps({'phase':'vla','simulation_time':.1})+'\n')
    rows=preparation_samples()+samples(6.2)
    (trial/'physics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=audit_trial(trial)
    assert report['first_support_window_passed']
    assert not report['task_passed'] and not report['quality_passed']
    assert 'target_grasp_before_first_model_command_not_attributable' in report['reasons']


@pytest.mark.parametrize('failure',['missing','gap','no_table_contact'])
def test_preparation_baseline_requires_complete_physics_evidence(trial,failure):
    before=preparation_samples()
    if failure=='missing':before=[]
    if failure=='gap':before=[r for r in before if not -.12<=r['simulation_time']<=-.06]
    if failure=='no_table_contact':before[-1]['objects']['cube']['table_contact']=False
    (trial/'physics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in before+samples()))
    report=audit_trial(trial)
    assert not report['preparation_evidence']['verified']
    assert not report['task_passed'] and not report['quality_passed']


@pytest.mark.parametrize('captured',[None,float('nan'),.1])
def test_first_model_observation_clock_must_be_known_and_precede_execution(trial,captured):
    path=trial/'actions/reply-001.npz'
    with np.load(path) as original:
        reply={k:original[k] for k in original.files if k!='camera_capture_simulation_time'}
    if captured is not None:reply['camera_capture_simulation_time']=captured
    np.savez(path,**reply)
    report=audit_trial(trial)
    assert not report['model_evidence']['verified']
    assert not report['task_passed'] and not report['quality_passed']


def test_normal_time_budget_completion_does_not_mean_task_success(trial):
    rows = samples()
    for row in rows:
        row['objects']['cube']['jaw_contacts']['left'] = []
    (trial/'physics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report = audit_trial(trial)
    assert report['execution_completed']
    assert not report['task_passed'] and not report['quality_passed']


def test_safety_stop_preserves_observation_but_is_not_complete_task(trial):
    change_json(trial, 'client.json', status='failed', error='safety stop')
    report = audit_trial(trial)
    assert report['two_second_support_observed']
    assert not report['execution_completed'] and not report['task_passed']


def test_unexecuted_model_reply_is_not_a_model_grasp(trial):
    change_json(trial, 'client.json', chunks=0)
    report = audit_trial(trial)
    assert not report['model_evidence']['verified'] and not report['task_passed']


@pytest.mark.parametrize('failure', ['missing_reply', 'assisted_config', 'assisted_phase', 'missing_clock'])
def test_missing_or_assisted_model_provenance_cannot_pass(trial, failure):
    if failure == 'missing_reply':
        (trial/'actions/reply-001.npz').unlink()
    elif failure == 'assisted_config':
        (trial/'client-config.yaml').write_text(yaml.safe_dump({
            'execution_mode': 'model_only', 'grasp_execution': {'hold_seconds': 6}}))
    elif failure == 'assisted_phase':
        (trial/'control.jsonl').write_text(json.dumps({'phase': 'rgb_tactile_hold', 'simulation_time': 0})+'\n')
    else:
        (trial/'control.jsonl').write_text(json.dumps({'phase': 'vla'})+'\n')
    report = audit_trial(trial)
    assert not report['model_evidence']['verified']
    assert not report['task_passed'] and not report['quality_passed']
    if failure in ('assisted_phase', 'missing_clock'):
        assert not report['two_second_support_observed']


def test_failed_collision_and_non_target_push_are_global_failures(trial):
    physical = json.loads((trial/'task-report.json').read_text())
    physical['table_collision_audit']['passed'] = False
    (trial/'task-report.json').write_text(json.dumps(physical))
    rows = samples(); rows[-1]['objects']['other']['position'][0] += .02
    (trial/'physics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report = audit_trial(trial)
    assert report['first_support_window_passed']
    assert not report['quality_passed']
    assert 'full_episode_table_collision_audit_failed' in report['reasons']
    assert 'non_target_object_disturbed' in report['reasons']


def test_wrong_object_lift_anywhere_invalidates_episode(trial):
    rows = samples(); rows[-1]['objects']['other']['bottom_above_table'] = .1
    (trial/'physics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report = audit_trial(trial)
    assert report['wrong_object_lifted'] and not report['task_passed']


@pytest.mark.parametrize('depth', [.01, float('nan')])
def test_inconsistent_passed_collision_flag_is_not_sufficient(trial, depth):
    physical = json.loads((trial/'task-report.json').read_text())
    physical['table_collision_audit']['maximum_penetration_m'] = depth
    (trial/'task-report.json').write_text(json.dumps(physical))
    report = audit_trial(trial)
    assert not report['quality_passed']
    json.dumps(report, allow_nan=False)


def test_missing_trial_retained_as_failure(tmp_path):
    report = audit_trial(tmp_path/'missing-trial')
    assert not report['task_passed'] and not report['quality_passed']
    assert report['longest_supported_seconds'] is None


def test_suite_keeps_all_planned_missing_trials_in_denominator(tmp_path):
    (tmp_path/'suite.json').write_text(json.dumps({'repeats_planned': 3, 'cases': []}))
    output = tmp_path/'audit.json'
    result = subprocess.run([sys.executable, '-m', 'robot_client.sonic.raw_grasp_quality',
                             str(tmp_path), '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 1
    report = json.loads(output.read_text())
    assert report['total'] == 3 and report['task_passed'] == report['quality_passed'] == 0
