"""Repeat isolated frozen-VLA tabletop trials; retain every success and failure.

The model server must already be running. Each repetition restarts simulation
and SONIC; task poses are supplied only to scene generation and scoring.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal

import numpy as np
import yaml

from .run import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--simulation', action='store_true', required=True)
    parser.add_argument('--task', type=Path, default=Path(__file__).with_name('tabletop_vla.yaml'))
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('vla_stable.yaml'))
    parser.add_argument('--arm-tracking', choices=('sonic', 'joint_pd', 'arms_pd'), required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--jitter-m', type=float, default=0., help='Independent uniform XY offsets for every object')
    parser.add_argument('--scene-seed', type=int, default=20261027)
    parser.add_argument('--max-slip-mm', type=float,
                        help='Require an independent full-hold quality audit below this slip limit')
    args = parser.parse_args()
    if args.repeats < 1 or not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('repeats and seconds must be positive')
    if not np.isfinite(args.jitter_m) or args.jitter_m < 0:
        parser.error('jitter-m must be finite and nonnegative')
    if args.max_slip_mm is not None and (not np.isfinite(args.max_slip_mm) or args.max_slip_mm<=0):
        parser.error('max-slip-mm must be positive and finite')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    if args.max_slip_mm is not None and not config.get('grasp_execution'):
        parser.error('max-slip-mm currently requires an explicit grasp_execution hold')
    locked = yaml.safe_dump(config, sort_keys=True)
    (output/'locked-client.yaml').write_text(locked)
    task = yaml.safe_load(args.task.read_text())
    inputs = output/'inputs'
    inputs.mkdir()
    rng = np.random.default_rng(args.scene_seed)
    source_root=Path(__file__).parent
    paths=list(source_root.glob('*.py'))+list((source_root/'assets').rglob('*'))
    sources={p.relative_to(source_root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for p in paths if p.is_file()}
    workspace=args.workspace.resolve()
    inference_paths=['model_server/action_server_wbc_msgpack_unitree.py',
                     'unifolm_wla/model/framework/VLM4A/QwenMMDiT.py']
    inference_sources={name:hashlib.sha256((workspace/'src'/name).read_bytes()).hexdigest()
                       for name in inference_paths if (workspace/'src'/name).is_file()}
    report = {'success':False, 'repeats_planned':args.repeats, 'cases':[],
              'config_sha256':hashlib.sha256(locked.encode()).hexdigest(),
              'task_sha256':hashlib.sha256(yaml.safe_dump(task,sort_keys=True).encode()).hexdigest(),
              'source_sha256':sources,
              'inference_source_sha256':inference_sources,
              'arm_tracking':args.arm_tracking, 'seconds_per_trial':args.seconds,
              'jitter_m':args.jitter_m, 'scene_seed':args.scene_seed,
              'max_slip_mm':args.max_slip_mm,
              'wall_started':time.time(), 'all_failures_retained':True}
    for index in range(1,args.repeats+1):
        scene = yaml.safe_load(yaml.safe_dump(task))
        for obj in scene['objects']:
            obj['xy'] = (np.asarray(obj['xy'])+rng.uniform(-args.jitter_m,args.jitter_m,2)).tolist()
        task_path = inputs/f'{index:03d}.yaml'
        task_path.write_text(yaml.safe_dump(scene))
        trial = output/f'trial-{index:03d}'
        command = [sys.executable,'-u','-m','robot_client.sonic.grasp_trial',
                   '--workspace',str(args.workspace.resolve()),'--simulation','--policy','vla',
                   '--arm-tracking',args.arm_tracking,'--task',str(task_path),
                   '--config',str(output/'locked-client.yaml'),'--seconds',str(args.seconds),
                   '--output',str(trial)]
        started = time.time()
        print(f'Starting {index}/{args.repeats}: {trial}',flush=True)
        with (output/f'trial-{index:03d}-launcher.log').open('w') as stream:
            result = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,start_new_session=True)
            try:
                result.wait()
            except BaseException:
                if result.poll() is None:
                    os.killpg(result.pid,signal.SIGINT)
                    try:result.wait(timeout=40)
                    except subprocess.TimeoutExpired:
                        os.killpg(result.pid,signal.SIGTERM)
                        result.wait(timeout=10)
                report['interrupted']=True
                (output/'suite.json').write_text(json.dumps(report,indent=2)+'\n')
                raise
        case = {'index':index,'path':str(trial),'returncode':result.returncode,
                'elapsed_seconds':time.time()-started,'success':False}
        for filename,key in [('task-report.json','physical'),('client.json','client')]:
            try:case[key] = json.loads((trial/filename).read_text())
            except (FileNotFoundError,json.JSONDecodeError):pass
        case['success'] = bool(result.returncode == 0 and case.get('physical',{}).get('success')
                               and case.get('client',{}).get('status') == 'passed')
        case['task_success']=case['success']
        if args.max_slip_mm is not None:
            from .grasp_quality import audit_trial
            case['quality']=audit_trial(trial,args.max_slip_mm/1000)
            (trial/'grasp-quality.json').write_text(json.dumps(case['quality'],indent=2,allow_nan=False)+'\n')
            case['success'] &= case['quality']['quality_passed']
        case['source_unchanged']=all(hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest
                                     for name,digest in sources.items())
        case['source_unchanged'] &= all(hashlib.sha256((workspace/'src'/name).read_bytes()).hexdigest()==digest
                                      for name,digest in inference_sources.items())
        case['success'] &= case['source_unchanged']
        report['cases'].append(case)
        report['passed'] = sum(c['success'] for c in report['cases'])
        report['success'] = len(report['cases']) == args.repeats and report['passed'] == args.repeats
        report['wall_updated'] = time.time()
        (output/'suite.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f"Finished {index}/{args.repeats}: success={case['success']}; total={report['passed']}/{index}",flush=True)
    raise SystemExit(0 if report['success'] else 1)


if __name__ == '__main__':
    main()
