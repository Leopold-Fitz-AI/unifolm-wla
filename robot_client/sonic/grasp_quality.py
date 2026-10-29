"""Offline grasp quality audit. Simulator truth never enters the controller.

Run against a trial or a grasp_repeat suite. Quality is stricter than task
completion: the entire requested hold must remain supported and below a slip
limit, with no missing interval hidden by sparse or truncated recordings.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def audit_samples(samples, start, duration, target, hand, *, max_slip_m=.005,
                  min_height_m=.08, max_gap_seconds=.06):
    if (hand not in ('left','right') or not np.isfinite(
            [start,duration,max_slip_m,min_height_m,max_gap_seconds]).all()
            or min(duration,max_slip_m,max_gap_seconds)<=0):
        raise ValueError('invalid quality audit bounds')
    times=np.asarray([r['simulation_time'] for r in samples],float)
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('physics clock must be finite and strictly increasing')
    rows=[r for r in samples if start<=r['simulation_time']<=start+duration+1e-8]
    if len(rows)<2:
        return {'quality_passed':False,'reasons':['missing_hold_samples'],'hold_samples':len(rows)}
    t=np.asarray([r['simulation_time'] for r in rows])
    positions=[];tcp=[];heights=[];bilateral=[];table=[]
    for r in rows:
        obj=r['objects'][target];robot=r['hands'][hand]
        rotation=np.asarray(robot['wrist_rotation'],float).reshape(3,3)
        p=np.asarray(robot['tcp_position'],float)
        position=np.asarray(obj['position'],float)
        if not np.isfinite(np.r_[rotation.ravel(),p,position,obj['bottom_above_table']]).all():
            raise ValueError('nonfinite physics pose')
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-4) or np.linalg.det(rotation)<.999:
            raise ValueError('invalid wrist rotation')
        positions.append(rotation.T@(position-p));tcp.append(p)
        heights.append(obj['bottom_above_table'])
        bilateral.append({0,1}.issubset(obj['jaw_contacts'][hand]))
        table.append(bool(obj['table_contact']))
    local=np.asarray(positions);tcp=np.asarray(tcp)
    slip=np.linalg.norm(local-local[0],axis=1)
    # Include both ends in coverage; do not claim 6 s from only 2 s of samples.
    gap=float(max(t[0]-start,start+duration-t[-1],np.max(np.diff(t))))
    reasons=[]
    if gap>max_gap_seconds:reasons.append('incomplete_hold_coverage')
    if not all(bilateral):reasons.append('bilateral_support_lost')
    if any(table):reasons.append('table_support')
    if any(r['suspension_enabled'] for r in rows):reasons.append('suspension_support')
    if min(heights)<min_height_m:reasons.append('insufficient_height')
    if float(slip.max())>max_slip_m:reasons.append('excessive_slip')
    for r in rows:
        if any(o['bottom_above_table']>=min_height_m for name,o in r['objects'].items() if name!=target):
            reasons.append('wrong_object_lifted');break
    return {'quality_passed':not reasons,'reasons':reasons,'hold_samples':len(rows),
            'requested_hold_seconds':duration,'observed_span_seconds':float(t[-1]-t[0]),
            'maximum_sample_gap_seconds':gap,'maximum_local_slip_m':float(slip.max()),
            'final_local_slip_m':float(slip[-1]),'local_displacement_m':(local[-1]-local[0]).tolist(),
            'tcp_range_m':np.ptp(tcp,axis=0).tolist(),
            'minimum_bottom_above_table_m':float(min(heights)),
            'bilateral_contact_fraction':float(np.mean(bilateral)),
            'max_slip_limit_m':max_slip_m}


def audit_trial(path, max_slip_m=.005):
    path=Path(path)
    try:
        task=yaml.safe_load((path/'task.yaml').read_text())
        config=yaml.safe_load((path/'client-config.yaml').read_text())
        execution=json.loads((path/'grasp-execution.json').read_text())
        physical=json.loads((path/'task-report.json').read_text())
        client=json.loads((path/'client.json').read_text())
        holds=[e for e in execution['events'] if e['phase']=='hold']
        if len(holds)!=1:raise ValueError('expected one hold event')
        samples=[json.loads(line) for line in (path/'physics.jsonl').open() if line.strip()]
        report=audit_samples(samples,holds[0]['simulation_time'],config['grasp_execution']['hold_seconds'],
                             task['target'],task['required_hand'],max_slip_m=max_slip_m,
                             min_height_m=task['lift_height'])
        complete=bool(execution['completed'] and physical['success'] and client['status']=='passed')
        report['task_completed']=complete
        if task.get('robot_contacts_from_start',False):
            collision=physical.get('table_collision_audit',{})
            report['table_collision_audit']=collision
            if not collision.get('passed',False):
                report['reasons'].append('full_episode_table_collision_audit_failed')
                report['quality_passed']=False
            disturbances={}
            for name in samples[0]['objects']:
                if name==task['target']:continue
                origin=np.asarray(samples[0]['objects'][name]['position'][:2],float)
                disturbances[name]=max(float(np.linalg.norm(np.asarray(s['objects'][name]['position'][:2])-origin)) for s in samples)
            report['non_target_max_xy_displacement_m']=disturbances
            if any(d>.01 for d in disturbances.values()):
                report['reasons'].append('non_target_object_disturbed')
                report['quality_passed']=False
        if not complete:
            report['reasons'].append('task_not_completed');report['quality_passed']=False
        report['phase_times']={e['phase']:e['simulation_time'] for e in execution['events']}
    except (OSError,ValueError,KeyError,TypeError) as error:
        report={'quality_passed':False,'reasons':['missing_or_invalid_evidence'],'error':str(error)}
    return dict(trial=str(path),**report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path)
    parser.add_argument('--max-slip-mm',type=float,default=5.)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not np.isfinite(args.max_slip_mm) or args.max_slip_mm<=0:parser.error('max-slip-mm must be positive')
    if (args.path/'suite.json').is_file():
        suite=json.loads((args.path/'suite.json').read_text())
        # Enumerate planned trials, so absent/aborted trials remain failures.
        paths=[args.path/f'trial-{i:03d}' for i in range(1,suite['repeats_planned']+1)]
    else:paths=[args.path]
    cases=[audit_trial(p,args.max_slip_mm/1000) for p in paths]
    result={'passed':sum(c['quality_passed'] for c in cases),'total':len(cases),
            'max_slip_mm':args.max_slip_mm,'cases':cases}
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:stream.write(text)
    print(text,end='')
    raise SystemExit(0 if result['passed']==result['total'] else 1)


if __name__=='__main__':main()
