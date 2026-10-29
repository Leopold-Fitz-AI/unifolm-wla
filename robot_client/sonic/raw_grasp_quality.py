"""Offline measurement of model-only grasping, without a scripted hold event.

Simulator object poses are read only by this evaluator, never by the policy.
The first supported lift after the first logged model command fixes the quality
window. Later, more favorable windows do not replace it. A normally completed
client time budget is reported separately from physical task/quality success.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .grasp_quality import audit_samples


def _invalid_json_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def _json(text):
    return json.loads(text, parse_constant=_invalid_json_constant)


def _empty_measurements(reason):
    return {
        'physics_integrity_passed': False, 'reasons': [reason],
        'support_segments': [], 'longest_supported_seconds': None,
        'two_second_support_observed': False,
        'six_second_support_segment_observed': False,
        'first_support_start': None, 'first_support_window': None,
        'first_support_window_passed': False,
        'maximum_supported_height_m': None,
        'non_target_max_xy_displacement_m': {},
        'wrong_object_lifted': False, 'simulation_clock_reset': False,
    }


def _pose(row, target, hand):
    obj, robot = row['objects'][target], row['hands'][hand]
    p = np.asarray(obj['position'], float)
    tcp = np.asarray(robot['tcp_position'], float)
    rotation = np.asarray(robot['wrist_rotation'], float).reshape(3, 3)
    if p.shape != (3,) or tcp.shape != (3,) or not np.isfinite(
            np.r_[p, tcp, rotation.ravel(), obj['bottom_above_table'],
                  row['base_height'], row['base_tilt_deg']]).all():
        raise ValueError('invalid/nonfinite physics pose')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) < .999:
        raise ValueError('invalid wrist rotation')
    return rotation.T @ (p-tcp)


def measure_samples(samples, target, hand, *, evaluation_start, hold_seconds=6.,
                    min_height_m=.08, max_slip_m=.005, max_gap_seconds=.06,
                    max_non_target_m=.01):
    """Measure all supported segments and the predeclared first support window.

    Segment durations use observed endpoints, not inferred time past a last
    sample. Unknown intervals break continuity. Invalid clocks invalidate the
    measurements instead of allowing a reset to manufacture a new successful
    segment.
    """
    if hand not in ('left', 'right') or not np.isfinite([
            evaluation_start, hold_seconds, min_height_m, max_slip_m,
            max_gap_seconds, max_non_target_m]).all() or min(
                hold_seconds, min_height_m, max_slip_m, max_gap_seconds,
                max_non_target_m) <= 0:
        raise ValueError('invalid raw grasp audit bounds')
    if not samples:
        return _empty_measurements('missing_physics_samples')
    times = np.asarray([r['simulation_time'] for r in samples], float)
    if not np.isfinite(times).all():
        return _empty_measurements('invalid_physics_clock')
    if np.any(np.diff(times) <= 0):
        result = _empty_measurements('nonmonotonic_physics_clock')
        result['simulation_clock_reset'] = True
        return result

    objects = set(samples[0]['objects'])
    if target not in objects:
        raise ValueError('target missing from physics evidence')
    origins = {name: np.asarray(samples[0]['objects'][name]['position'][:2], float)
               for name in objects if name != target}
    disturbance = {name: 0. for name in origins}
    segments, active = [], []
    first_start = None
    wrong_object = False
    max_supported_height = None
    previous_time = None

    def finish():
        nonlocal active
        if not active:
            return
        start, end = active[0][0], active[-1][0]
        local = np.asarray([p for _, p in active])
        segments.append({
            'start': start, 'end': end, 'duration_seconds': end-start,
            'samples': len(active),
            'maximum_local_slip_m': float(np.linalg.norm(local-local[0], axis=1).max()),
        })
        active = []

    for row in samples:
        now = float(row['simulation_time'])
        if set(row['objects']) != objects:
            raise ValueError('object inventory changed in physics evidence')
        local = _pose(row, target, hand)
        for name, origin in origins.items():
            obj = row['objects'][name]
            xy = np.asarray(obj['position'][:2], float)
            height = float(obj['bottom_above_table'])
            if xy.shape != (2,) or not np.isfinite(np.r_[xy, height, origin]).all():
                raise ValueError('invalid non-target physics pose')
            disturbance[name] = max(disturbance[name], float(np.linalg.norm(xy-origin)))
            wrong_object |= height >= min_height_m
        if previous_time is not None and now-previous_time > max_gap_seconds:
            finish()
        previous_time = now
        obj = row['objects'][target]
        supported = (now >= evaluation_start and not row['suspension_enabled']
                     and row['base_height'] > .5 and row['base_tilt_deg'] < 25
                     and obj['bottom_above_table'] >= min_height_m
                     and {0, 1}.issubset(obj['jaw_contacts'][hand])
                     and not obj['table_contact'])
        if supported:
            if first_start is None:
                first_start = now
            active.append((now, local))
            height = float(obj['bottom_above_table'])
            max_supported_height = height if max_supported_height is None else max(max_supported_height, height)
        else:
            finish()
    finish()
    longest = max((s['duration_seconds'] for s in segments), default=0.)
    reasons = []
    if first_start is None:
        window = None
        reasons.append('no_supported_lift_after_model_execution')
    else:
        window = audit_samples(samples, first_start, hold_seconds, target, hand,
                               max_slip_m=max_slip_m, min_height_m=min_height_m,
                               max_gap_seconds=max_gap_seconds)
        # A truncated stream just short of six seconds must not be accepted
        # merely because its end is inside the sampling-gap tolerance.
        if times[-1] < first_start+hold_seconds-1e-8:
            window['reasons'].append('recording_ends_before_requested_window')
        in_window = [r for r in samples if first_start <= r['simulation_time'] <= first_start+hold_seconds]
        if any(r['base_height'] <= .5 or r['base_tilt_deg'] >= 25 for r in in_window):
            window['reasons'].append('robot_pose_unsafe_during_window')
        window['reasons'] = list(dict.fromkeys(window['reasons']))
        window['quality_passed'] = not window['reasons']
        reasons.extend(window['reasons'])
    if wrong_object:
        reasons.append('wrong_object_lifted')
    if any(v > max_non_target_m for v in disturbance.values()):
        reasons.append('non_target_object_disturbed')
    if longest < 2.-1e-8:
        reasons.append('no_complete_two_second_support')
    return {
        'physics_integrity_passed': True, 'reasons': list(dict.fromkeys(reasons)),
        'evaluation_start': float(evaluation_start), 'support_segments': segments,
        'longest_supported_seconds': float(longest),
        'two_second_support_observed': bool(longest >= 2.-1e-8),
        'six_second_support_segment_observed': bool(longest >= hold_seconds-1e-8),
        'first_support_start': first_start, 'first_support_window': window,
        'first_support_window_passed': bool(window and window['quality_passed']),
        'maximum_supported_height_m': max_supported_height,
        'maximum_recording_gap_seconds': float(np.diff(times).max()) if len(times) > 1 else None,
        'non_target_max_xy_displacement_m': disturbance,
        'wrong_object_lifted': bool(wrong_object), 'simulation_clock_reset': False,
    }


def preparation_evidence(samples, target, *, observation_time, execution_start, min_height_m=.08,
                         max_gap_seconds=.06):
    """Require an untouched tabletop baseline before the first model observation.

    Control traces are downsampled, so their first row can follow the first
    model action. The camera timestamp proves what preceded inference; a
    grasp in the unlogged interval before control evidence is conservatively
    rejected as ambiguous. This is an offline provenance check only.
    """
    result={'verified':False,'model_observation_time':observation_time,
            'samples':0,'last_sample_time':None,'reasons':[]}
    if observation_time is None or not np.isfinite(observation_time):
        result['reasons']=['model_observation_clock_unavailable']
        return result
    rows=[row for row in samples if row['simulation_time']<=observation_time+1e-8]
    result['samples']=len(rows)
    if not rows:
        result['reasons']=['missing_preparation_physics']
        return result
    times=np.asarray([row['simulation_time'] for row in rows],float)
    result['last_sample_time']=float(times[-1])
    if (not np.isfinite(times).all() or np.any(np.diff(times)<=0)
        or max(observation_time-times[-1],np.diff(times).max(initial=0.))>max_gap_seconds):
        result['reasons'].append('incomplete_preparation_physics')
    for row in rows:
        obj=row['objects'][target]
        if any({0,1}.issubset(jaws) for jaws in obj['jaw_contacts'].values()):
            result['reasons'].append('target_already_grasped_during_preparation')
        if obj['bottom_above_table']>=min_height_m:
            result['reasons'].append('target_already_lifted_during_preparation')
    if not rows[-1]['objects'][target]['table_contact']:
        result['reasons'].append('missing_target_tabletop_baseline')
    if execution_start is None or not np.isfinite(execution_start):
        result['reasons'].append('model_execution_clock_unavailable')
    else:
        unlogged=[row for row in samples if observation_time<row['simulation_time']<execution_start]
        if any(row['objects'][target]['bottom_above_table']>=min_height_m or
               any({0,1}.issubset(jaws) for jaws in row['objects'][target]['jaw_contacts'].values())
               for row in unlogged):
            result['reasons'].append('target_grasp_before_first_model_command_not_attributable')
    result['reasons']=list(dict.fromkeys(result['reasons']))
    result['verified']=not result['reasons']
    return result


def model_evidence(path, config, client):
    """Check independent persisted evidence; this does not certify an official WBC."""
    reasons = []
    contract = client.get('execution_contract', {})
    if config.get('execution_mode') != 'model_only' or client.get('execution_mode') != 'model_only':
        reasons.append('model_only_mode_not_declared')
    if (not contract.get('validated') or contract.get('mode') != 'model_only'
            or contract.get('task_assists_enabled') != []):
        reasons.append('model_only_contract_not_verified')
    active = [key for key in ('visual_target', 'grasp_execution', 'stationary_single_arm',
              'base_position_hold', 'cartesian_tracking', 'align_first_ee_frame') if config.get(key)]
    if config.get('grasp_hold', {}).get('enabled', False):
        active.append('grasp_hold')
    if config.get('action_time_scale', 1.) != 1.:
        active.append('action_time_scale')
    if active:
        reasons.append('task_assist_or_action_retiming_configured')
    if any(client.get(k) for k in ('visual_selection', 'grasp_execution', 'grasp_hold')):
        reasons.append('task_assist_reported')

    controls = [_json(line) for line in (path/'control.jsonl').open() if line.strip()]
    if not controls or any(r.get('phase') != 'vla' for r in controls):
        reasons.append('missing_or_non_model_control_phase')
    command_times = np.asarray([r['simulation_time'] for r in controls
                               if r.get('phase') == 'vla' and 'simulation_time' in r], float)
    if (len(command_times) != len(controls) or not len(command_times)
            or not np.isfinite(command_times).all() or np.any(np.diff(command_times) < 0)):
        reasons.append('missing_or_invalid_model_execution_clock')
        start = None
    else:
        start = float(command_times[0])
    replies = sorted((path/'actions').glob('reply-*.npz'))
    required = {'action.left_ee_rpy': 6, 'action.right_ee_rpy': 6,
                'action.left_gripper': 1, 'action.right_gripper': 1,
                'action.lower_body': 15, 'action.base_command': 4}
    valid_replies = 0
    observation_times=[]
    for reply in replies:
        with np.load(reply, allow_pickle=False) as data:
            if any(k.startswith('executed_') for k in data.files):
                reasons.append('rewritten_model_reply_recorded')
            lengths = []
            for key, width in required.items():
                value = data[key]
                if value.ndim == 3 and value.shape[0] == 1:
                    value = value[0]
                if value.ndim != 2 or value.shape[1] != width or not len(value) or not np.isfinite(value).all():
                    raise ValueError(f'invalid raw model reply {reply.name}: {key}')
                lengths.append(len(value))
            if len(set(lengths)) != 1:
                raise ValueError('raw model reply action horizons differ')
            captured=data['camera_capture_simulation_time'] if 'camera_capture_simulation_time' in data else None
            if captured is None or captured.shape!=() or not np.isfinite(captured):
                reasons.append('missing_or_invalid_model_observation_clock')
            else:
                observation_times.append(float(captured))
            valid_replies += 1
    if not valid_replies:
        reasons.append('missing_raw_model_reply')
    chunks = client.get('chunks', 0)
    if not isinstance(chunks, int) or isinstance(chunks, bool) or not 0 < chunks <= valid_replies:
        reasons.append('executed_model_chunk_count_not_verified')
    if list((path/'actions').glob('selection-*.json')) or list((path/'actions').glob('candidates-*.npz')):
        reasons.append('task_candidate_selection_recorded')
    first_observation=min(observation_times) if observation_times else None
    if first_observation is not None and start is not None and first_observation>start+1e-8:
        reasons.append('model_observation_after_execution_start')
    return {'verified': not reasons, 'reasons': list(dict.fromkeys(reasons)),
            'active_config_fields': active, 'raw_model_replies': valid_replies,
            'control_samples': len(controls), 'first_model_command_time': start,
            'first_model_observation_time':first_observation,
            'execution_contract': contract, 'policy_source': client.get('policy_source')}


def audit_trial(path, *, hold_seconds=6., max_slip_m=.005):
    path = Path(path)
    result = dict(trial=str(path), task_passed=False, quality_passed=False,
                  execution_completed=False, two_second_support_observed=False,
                  first_support_window=None, longest_supported_seconds=None)
    try:
        task = yaml.safe_load((path/'task.yaml').read_text())
        config = yaml.safe_load((path/'client-config.yaml').read_text())
        physical = _json((path/'task-report.json').read_text())
        client = _json((path/'client.json').read_text())
        samples = [_json(line) for line in (path/'physics.jsonl').open() if line.strip()]
        evidence = model_evidence(path, config, client)
        result['model_evidence'] = evidence
        # A missing execution timestamp cannot attribute a lift to the model;
        # in particular it must not count the preparation motion as a grasp.
        start = evidence['first_model_command_time']
        measures = (measure_samples(samples, task['target'], task['required_hand'],
                                   evaluation_start=start,
                                   hold_seconds=hold_seconds, max_slip_m=max_slip_m,
                                   min_height_m=task['lift_height']) if start is not None
                    else _empty_measurements('model_execution_start_unavailable'))
        result.update(measures)
        preparation=preparation_evidence(samples,task['target'],
            observation_time=evidence['first_model_observation_time'],execution_start=start,
            min_height_m=task['lift_height'])
        result['preparation_evidence']=preparation
        reasons = list(measures['reasons']) + evidence['reasons'] + preparation['reasons']
        collision = physical.get('table_collision_audit', {})
        penetration = collision.get('maximum_penetration_m', float('nan'))
        penetration_limit = float(task.get('table_penetration_limit_m', .001))
        collision_verified = bool(task.get('robot_contacts_from_start') and collision.get('passed')
            and collision.get('enabled_from_start') and collision.get('complete_step_coverage')
            and collision.get('collider_manifest_valid') and collision.get('checked_physics_steps', 0) > 0
            and collision.get('required_robot_collision_geoms', 0) > 0
            and collision.get('disabled_steps') == 0 and collision.get('invalid_contacts') == 0
            and not collision.get('clock_reset', True)
            and np.isfinite([penetration, penetration_limit]).all()
            and 0 <= penetration <= penetration_limit <= .005)
        if not collision_verified:
            reasons.append('full_episode_table_collision_audit_failed')
        reset = bool(physical.get('simulation_reset') or collision.get('clock_reset')
                     or measures['simulation_clock_reset'])
        if reset:
            reasons.append('simulation_reset')
        completed = client.get('status') == 'passed'
        if not completed:
            reasons.append('client_execution_not_completed')
        global_integrity = bool(measures['physics_integrity_passed'] and evidence['verified']
            and preparation['verified']
            and collision_verified and not reset and not measures['wrong_object_lifted']
            and not any(x > .01 for x in measures['non_target_max_xy_displacement_m'].values()))
        result.update(execution_completed=completed, client_status=client.get('status'),
                      client_error=client.get('error'), simulation_reset=reset,
                      table_collision_audit=collision, episode_integrity_passed=global_integrity,
                      requested_hold_seconds=hold_seconds, max_slip_limit_m=max_slip_m,
                      task_passed=bool(completed and global_integrity and measures['two_second_support_observed']),
                      quality_passed=bool(completed and global_integrity and measures['first_support_window_passed']),
                      reasons=list(dict.fromkeys(reasons)))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result.update(task_passed=False, quality_passed=False,
                      reasons=['missing_or_invalid_evidence'], error=str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--hold-seconds', type=float, default=6.)
    parser.add_argument('--max-slip-mm', type=float, default=5.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not np.isfinite([args.hold_seconds, args.max_slip_mm]).all() or min(args.hold_seconds, args.max_slip_mm) <= 0:
        parser.error('hold-seconds and max-slip-mm must be positive and finite')
    if (args.path/'suite.json').exists():
        suite = _json((args.path/'suite.json').read_text())
        planned = suite['repeats_planned']
        if not isinstance(planned, int) or isinstance(planned, bool) or planned < 1:
            parser.error('suite repeats_planned must be a positive integer')
        paths = [args.path/f'trial-{i:03d}' for i in range(1, planned+1)]
    else:
        paths = [args.path]
    cases = [audit_trial(p, hold_seconds=args.hold_seconds, max_slip_m=args.max_slip_mm/1000) for p in paths]
    result = {'task_passed': sum(c['task_passed'] for c in cases),
              'quality_passed': sum(c['quality_passed'] for c in cases), 'total': len(cases),
              'hold_seconds': args.hold_seconds, 'max_slip_mm': args.max_slip_mm, 'cases': cases}
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')
    raise SystemExit(0 if result['quality_passed'] == result['total'] else 1)


if __name__ == '__main__':
    main()
