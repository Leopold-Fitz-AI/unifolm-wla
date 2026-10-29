"""Teacher-forced recorded-image sanity check; never commands a simulator.

Recorded future actions are evaluation labels only, never model observations.
Frames must have been decoded from the corresponding video without resampling.
"""
import argparse
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import pose_matrix
from .transport import PolicyClient


async def evaluate(args):
    import pyarrow.parquet as pq
    frames = [0,30,60,75,90,105,120,240]
    rows = pq.read_table(args.parquet).slice(0, max(frames)+30).to_pylist()
    results = []
    async with PolicyClient(args.policy_url, 60) as policy:
        for index, frame in enumerate(frames):
            row = rows[frame]
            obs = {'instruction':'Stack the red, green, and yellow blocks.', 'unnorm_key':'UnifoLM_G1_Dex1',
                   'observation.state.lower_body':np.asarray(row['observation.state.lower_body'], dtype='f4')}
            for side in ('left','right'):
                t = pose_matrix(row[f'observation.state.{side}_ee_pose_gripper_base'])
                obs[f'observation.state.{side}_ee_6d'] = np.r_[t[:3,3],t[:3,:2].T.reshape(-1)].astype('f4')
                obs[f'observation.state.{side}_gripper'] = np.asarray(row[f'observation.state.{side}_gripper'], dtype='f4')
            for key, name in [('cam_left_high','head_stereo_left'), ('cam_left_wrist','wrist_left'),
                               ('cam_right_wrist','wrist_right')]:
                image = cv2.imread(str(args.images/name/f'{index+1:03d}.jpg'))
                if image is None: raise ValueError('missing decoded frame')
                obs[f'observation.images.{key}'] = image
            await policy.request('policy_reset')
            action = await policy.request('get_action', obs=obs)
            result = {'frame':frame}
            for side in ('left','right'):
                truth = np.array([r[f'action.{side}_ee_pose_gripper_base'] for r in rows[frame:frame+30]])
                pred = action[f'action.{side}_ee_rpy'][0]
                gt_grip = np.asarray([r[f'action.{side}_gripper'] for r in rows[frame:frame+30]]).reshape(-1)
                result[side] = {'xyz_rms_m':float(np.sqrt(np.mean(np.sum((pred[:,:3]-truth[:,:3])**2,axis=1)))),
                    'rotation_mean_rad':float((Rotation.from_euler('xyz',truth[:,3:]).inv()*Rotation.from_euler('xyz',pred[:,3:])).magnitude().mean()),
                    'gripper_mae':float(np.mean(abs(action[f'action.{side}_gripper'][0,:,0]-gt_grip))),
                    'predicted_gripper_first_last':action[f'action.{side}_gripper'][0,[0,-1],0].tolist(),
                    'recorded_gripper_first_last':gt_grip[[0,-1]].tolist()}
            results.append(result)
            print(json.dumps(result), flush=True)
            args.output.write_text(json.dumps({'kind':'recorded-observation evaluation, no controller', 'results':results},indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parquet',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy-url',default='ws://127.0.0.1:18600')
    asyncio.run(evaluate(p.parse_args()))


if __name__ == '__main__':
    main()
