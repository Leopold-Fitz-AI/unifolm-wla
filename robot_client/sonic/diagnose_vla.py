"""Controlled offline VLA probes: real/sim state crossed with real/sim RGB.

Never runs a robot controller. Future dataset actions are evaluation labels only.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from .adapter import make_observation
from .kinematics import G1Kinematics, pose_matrix
from .run import load_config
from .transport import PolicyClient


async def probe(args):
    import pyarrow.parquet as pq
    args.output.mkdir(parents=True, exist_ok=True)
    row = pq.read_table(args.dataset_parquet).slice(0, 1).to_pylist()[0]
    q_real = np.r_[row['observation.state.lower_body'], row['observation.state.left_arm'],
                   row['observation.state.right_arm']]
    config = load_config(args.config)
    kin = G1Kinematics(args.sonic_root, config['reference_frame'], config['tool_offsets'])
    reply = np.load(sorted((args.trial/'actions').glob('reply*.npz'))[0])
    keys = ('cam_left_high', 'cam_left_wrist', 'cam_right_wrist')
    real_images = {k: cv2.imread(str(args.real_images/f'{name}-0.jpg'))
                   for k, name in zip(keys, ('head_stereo_left', 'wrist_left', 'wrist_right'))}
    sim_images = {k: cv2.imread(str(args.trial/'actions'/f'obs-0001-{k}.jpg')) for k in keys}
    if any(v is None for v in [*real_images.values(), *sim_images.values()]):
        raise ValueError('missing real/sim camera frames')
    states = {}
    for label, q, grips in [('real', q_real, [row['observation.state.left_gripper'], row['observation.state.right_gripper']]),
                            ('sim', reply['measured_q'], reply['measured_grippers'])]:
        states[label] = make_observation(kin, {'body_q': q}, real_images, np.asarray(grips).reshape(2),
                                         '', config['unnorm_key'])
    # Exact recorded observation frames, avoiding calibration-fit residuals.
    for side in ('left', 'right'):
        t = pose_matrix(row[f'observation.state.{side}_ee_pose_gripper_base'])
        states['real'][f'observation.state.{side}_ee_6d'] = np.r_[t[:3, 3], t[:3, :2].T.reshape(-1)].astype('f4')
    results = []
    mixed_head = dict(sim_images, cam_left_high=real_images['cam_left_high'])
    mixed_wrists = dict(real_images, cam_left_high=sim_images['cam_left_high'])
    async with PolicyClient(config['policy_url'], 60) as policy:
        for state_label in ('real', 'sim'):
            for image_label, images in [('real', real_images), ('sim', sim_images),
                                        ('real_head_sim_wrists', mixed_head), ('sim_head_real_wrists', mixed_wrists)]:
                for command in ('pick up the red cube with your right hand',
                                'pick up the red cube with your left hand',
                                'Stack the red, green, and yellow blocks.'):
                    for repeat in range(args.repeats):
                        await policy.request('policy_reset')
                        obs = copy.deepcopy(states[state_label])
                        obs.update({f'observation.images.{k}': v for k, v in images.items()})
                        obs['instruction'] = command
                        action = await policy.request('get_action', obs=obs)
                        result = {'state': state_label, 'images': image_label, 'instruction': command, 'repeat': repeat}
                        for side in ('left', 'right'):
                            pose = action[f'action.{side}_ee_rpy'][0]
                            result[side] = {'anchor_xyz': obs[f'observation.state.{side}_ee_6d'][:3].tolist(),
                                'first_xyz': pose[0, :3].tolist(), 'last_xyz': pose[-1, :3].tolist(),
                                'gripper_first_last': action[f'action.{side}_gripper'][0, [0, -1], 0].tolist()}
                        np.savez_compressed(args.output/f'probe-{len(results):03d}.npz', **action)
                        results.append(result)
                        print(json.dumps(result), flush=True)
                        (args.output/'probes.json').write_text(json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sonic-root', type=Path, required=True)
    p.add_argument('--dataset-parquet', type=Path, required=True)
    p.add_argument('--real-images', type=Path, required=True)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('vla_grasp.yaml'))
    p.add_argument('--repeats', type=int, default=2)
    asyncio.run(probe(p.parse_args()))


if __name__ == '__main__':
    main()
