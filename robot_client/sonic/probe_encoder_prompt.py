"""Paired, fixed-noise prompt-header ablation on frozen model weights."""
import argparse
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from .kinematics import pose_matrix
from .transport import PolicyClient


async def run(args):
    import pyarrow.parquet as pq
    rows = pq.read_table(args.parquet).slice(0, 121).to_pylist()
    samples = []
    for frame, image_index in [(0,1),(60,3),(90,5)]:
        row = rows[frame]
        obs = {'instruction':'Stack the red, green, and yellow blocks.', 'unnorm_key':'UnifoLM_G1_Dex1',
               'observation.state.lower_body':np.asarray(row['observation.state.lower_body'],dtype='f4')}
        for side in ('left','right'):
            t = pose_matrix(row[f'observation.state.{side}_ee_pose_gripper_base'])
            obs[f'observation.state.{side}_ee_6d'] = np.r_[t[:3,3],t[:3,:2].T.reshape(-1)].astype('f4')
            obs[f'observation.state.{side}_gripper'] = np.asarray(row[f'observation.state.{side}_gripper'],dtype='f4')
        for key, camera in [('cam_left_high','head_stereo_left'),('cam_left_wrist','wrist_left'),('cam_right_wrist','wrist_right')]:
            obs[f'observation.images.{key}'] = cv2.imread(str(args.real_images/camera/f'{image_index:03d}.jpg'))
        truth = {side:np.asarray([r[f'action.{side}_ee_pose_gripper_base'] for r in rows[frame:frame+30]])
                 for side in ('left','right')}
        samples.append((f'real-{frame}',obs,truth))
    reply = np.load(sorted((args.sim_trial/'actions').glob('reply*.npz'))[0])
    instruction = yaml.safe_load((args.sim_trial/'task.yaml').read_text())['instruction']
    obs = {'instruction':instruction,'unnorm_key':'UnifoLM_G1_Dex1',
           'observation.state.lower_body':reply['obs_lower_body']}
    for side,index in [('left',0),('right',1)]:
        obs[f'observation.state.{side}_ee_6d'] = reply[f'obs_{side}_ee']
        obs[f'observation.state.{side}_gripper'] = reply['measured_grippers'][index:index+1].astype('f4')
    index = int(reply['request_index'])
    for key in ('cam_left_high','cam_left_wrist','cam_right_wrist'):
        obs[f'observation.images.{key}'] = cv2.imread(str(args.sim_trial/'actions'/f'obs-{index:04d}-{key}.png'))
    samples.append(('sim-reset',obs,None))
    results = []
    async with PolicyClient(args.policy_url,60) as policy:
        test_obs = {**samples[0][1], 'inference_seed':17}
        await policy.request('policy_reset')
        baseline = await policy.request('get_action',obs=test_obs)
        await policy.request('policy_reset')
        repeated = await policy.request('get_action',obs={**test_obs,'encoder_generation_prompt':False})
        difference = max(float(np.max(abs(baseline[k]-repeated[k]))) for k in baseline)
        args.output.with_name('prompt-repeatability.json').write_text(json.dumps({'max_abs_difference':difference}))
        if difference > 1e-6:
            raise AssertionError('fixed-noise default/explicit-false predictions differ')
        for name,obs,truth in samples:
            if any(v is None for k,v in obs.items() if k.startswith('observation.images')):
                raise ValueError(f'missing image for {name}')
            for seed in (17,42,123):
                for header in (False,True):
                    await policy.request('policy_reset')
                    action = await policy.request('get_action',obs={**obs,'encoder_generation_prompt':header,'inference_seed':seed})
                    result = {'sample':name,'seed':seed,'encoder_generation_prompt':header}
                    for side in ('left','right'):
                        pred = action[f'action.{side}_ee_rpy'][0]
                        result[side] = {'delta_last_xyz':(pred[-1,:3]-obs[f'observation.state.{side}_ee_6d'][:3]).tolist(),
                            'gripper_first_last':action[f'action.{side}_gripper'][0,[0,-1],0].tolist()}
                        if truth is not None:
                            result[side]['xyz_rms_m'] = float(np.sqrt(np.mean(np.sum((pred[:,:3]-truth[side][:,:3])**2,axis=1))))
                    results.append(result)
                    args.output.write_text(json.dumps(results,indent=2))
                    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parquet',type=Path,required=True)
    p.add_argument('--real-images',type=Path,required=True)
    p.add_argument('--sim-trial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy-url',default='ws://127.0.0.1:18600')
    asyncio.run(run(p.parse_args()))


if __name__=='__main__':
    main()
