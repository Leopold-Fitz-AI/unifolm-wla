"""Replay one real saved RGB/proprioception snapshot for frozen VLA A/B tests.

Only observations are replayed. This sends no commands to any robot/simulator.
Outputs include unchanged observations, processed RGB, predictions and 2-D
projection metrics; no evaluator/object pose is used for candidate assessment.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from .run import load_config
from .transport import PolicyClient
from .vision_target import HeadProjector,condition_images,detect_target


async def run(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    config=load_config(args.trial/'client-config.yaml')
    task=yaml.safe_load((args.trial/'task.yaml').read_text())
    chosen=None
    for path in sorted((args.trial/'actions').glob('reply-*.npz')):
        a=np.load(path)
        if int(a['request_index'])==args.request_index:chosen={k:a[k].copy() for k in a.files};break
    if chosen is None:raise ValueError('recorded request index not found')
    images={key:cv2.imread(str(args.trial/'actions'/f'obs-{args.request_index:04d}-{key}.png'))
            for key in config['camera_mapping']}
    if any(v is None for v in images.values()):raise ValueError('missing recorded lossless RGB')
    obs={'observation.state.left_ee_6d':chosen['obs_left_ee'],
         'observation.state.right_ee_6d':chosen['obs_right_ee'],
         'observation.state.lower_body':chosen['obs_lower_body'],
         'unnorm_key':config['unnorm_key'],'inference_seed':args.seed}
    # Camera-matched grippers were not in early traces; reconstruct from the
    # source action observation only when recorded, otherwise label the fallback.
    obs['observation.state.left_gripper']=chosen['measured_grippers'][0:1]
    obs['observation.state.right_gripper']=chosen['measured_grippers'][1:2]
    q=chosen.get('obs_body_q',chosen['measured_q'])
    projector=HeadProjector(args.sonic_root,config,task['camera_mounts']['unifolm_head'])
    report={'seed':args.seed,'request_index':args.request_index,'source':str(args.trial),
            'gripper_state_source':'request-time measured_grippers; first-frame differences must be considered',
            'uses_simulator_object_state':False,'cases':[]}
    async with PolicyClient(args.url,timeout=30) as policy:
        for mode in ('original','target_color'):
            for color in ('red','green','yellow'):
                visible,counts=(images,{k:0 for k in images}) if mode=='original' else condition_images(images,color)
                detections={c:detect_target(images['cam_left_high'],c) for c in ('red','green','yellow')}
                for key,image in visible.items():
                    obs[f'observation.images.{key}']=image
                    cv2.imwrite(str(output/f'{mode}-{color}-{key}.png'),image)
                obs['instruction']=f'pick up the {color} cube with your left hand'
                # Remove cross-condition seam state; retain exactly the same noise.
                await policy.request('policy_reset')
                started=time.monotonic();action=await policy.request('get_action',obs=obs)
                elapsed=time.monotonic()-started
                np.savez_compressed(output/f'{mode}-{color}.npz',**action)
                px,depth=projector.action_pixels(action,q,0,images['cam_left_high'].shape)
                distances={c:float(np.linalg.norm(px[-1]-d['center_px'])) for c,d in detections.items()}
                report['cases'].append({'mode':mode,'color':color,'seconds':elapsed,'modified_pixels':counts,
                    'head_detections':detections,'last_tcp_pixel':px[-1].tolist(),'last_depth':float(depth[-1]),
                    'distance_px':distances,'left_ee_last':np.asarray(action['action.left_ee_rpy'])[0,-1].tolist(),
                    'left_gripper_range':[float(np.min(action['action.left_gripper'])),float(np.max(action['action.left_gripper']))]})
                (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                print(mode,color,distances,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sonic-root',type=Path,required=True);p.add_argument('--url',default='ws://127.0.0.1:18600')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--request-index',type=int,default=1)
    asyncio.run(run(p.parse_args()))


if __name__=='__main__':main()
