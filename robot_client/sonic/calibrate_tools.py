"""Calibrate the policy tool from official joint/EE observations.

The label gripper_base means gripper pose expressed in the robot base frame;
it does not identify the CAD base_link or the pad center. Use the dataset's
kinematic model, rather than absorb a different wrist chain into a tool fit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .urdf_frames import UrdfFrames, observation_joints

REFERENCE_URDF = Path(__file__).with_name('assets') / 'g1_mode15_kinematics/g1_29dof_mode_15_with_dex1_1.urdf'
OBSERVATION_FIELDS = ('left_arm', 'right_arm', 'lower_body', 'left_ee_pose_gripper_base',
                      'right_ee_pose_gripper_base', 'left_gripper', 'right_gripper', 'recomputed_ee_valid')


def fit_observations(observations, reference_urdf=REFERENCE_URDF, position_tolerance=1e-4,
                     rotation_tolerance=1e-5, enforce=True):
    """Fit alternating samples and independently validate withheld poses."""
    wrist = UrdfFrames(reference_urdf).forward(observation_joints(observations))
    count = len(wrist[0])
    if count < 40:
        raise ValueError('need at least 40 valid observations for fit and held-out validation')
    offsets, residuals = [], []
    for side, frames in zip(('left', 'right'), wrist):
        poses = np.asarray(observations[f'{side}_ee_pose_gripper_base'], float)
        if poses.shape != (count, 6) or not np.isfinite(poses).all():
            raise ValueError('dataset EE observations must be finite (N,6) xyz/rpy')
        target = np.repeat(np.eye(4)[None], count, axis=0)
        target[:, :3, 3] = poses[:, :3]
        target[:, :3, :3] = Rotation.from_euler('xyz', poses[:, 3:]).as_matrix()
        transforms = np.linalg.inv(frames) @ target
        fit = transforms[::2]
        xyz = fit[:, :3, 3].mean(0)
        rotation = Rotation.from_matrix(fit[:, :3, :3]).mean()
        errors = np.linalg.norm(transforms[:, :3, 3] - xyz, axis=1)
        angles = (rotation.inv() * Rotation.from_matrix(transforms[:, :3, :3])).magnitude()
        summary = {'pairs': count, 'fit_pairs': len(fit), 'validation_pairs': count // 2,
                   'position_rms_m': float(np.sqrt(np.mean(errors ** 2))),
                   'position_max_m': float(errors.max()),
                   'validation_position_max_m': float(errors[1::2].max()),
                   'rotation_max_rad': float(angles.max()),
                   'validation_rotation_max_rad': float(angles[1::2].max())}
        summary['rigid_tool_supported'] = bool(errors.max() <= position_tolerance and angles.max() <= rotation_tolerance)
        if enforce and not summary['rigid_tool_supported']:
            raise ValueError(f'{side}: observations do not support a rigid tool on this kinematic model: {summary}')
        offsets.append(np.r_[xyz, rotation.as_euler('xyz')].tolist())
        residuals.append(summary)
    return {'reference_frame': 'pelvis', 'tool_definition': 'dataset gripper end-effector frame; not an assumed pad center',
            'tool_offsets': offsets, 'residuals': residuals, 'reference_urdf': str(reference_urdf),
            'reference_urdf_sha256': hashlib.sha256(Path(reference_urdf).read_bytes()).hexdigest(),
            'method': 'fit mean(T_wrist_inverse * T_dataset_ee_in_base) on even samples; validate odd samples',
            'position_tolerance_m': position_tolerance, 'rotation_tolerance_rad': rotation_tolerance}


def calibrate(sonic_root, parquet, max_rows=3000, stride=1, reference_urdf=REFERENCE_URDF):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if max_rows < 40 or stride < 1:
        raise ValueError('max_rows must be >=40 and stride must be >=1')
    columns = ['observation.state.' + name for name in OBSERVATION_FIELDS]
    table = pq.read_table(parquet, columns=columns)
    # Span the complete recording instead of fitting only its initial posture.
    candidates = np.arange(0, table.num_rows, stride)
    indices = candidates[np.linspace(0, len(candidates) - 1, min(max_rows, len(candidates)), dtype=int)]
    rows = table.take(pa.array(indices)).to_pylist()
    valid = np.array([bool(np.asarray(row[columns[-1]]).reshape(-1)[0]) for row in rows])
    observations = {name: np.asarray([row['observation.state.' + name] for row in rows])[valid]
                    for name in OBSERVATION_FIELDS}
    result = fit_observations(observations, reference_urdf)
    result.update(source='https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Stack_Block',
                  source_file=str(parquet), source_sha256=hashlib.sha256(Path(parquet).read_bytes()).hexdigest(),
                  dataset_rows=table.num_rows, sampled_row_indices=indices[valid].tolist(),
                  ready_pose=[observations[f'{side}_ee_pose_gripper_base'][0].tolist() for side in ('left', 'right')],
                  ready_gripper_value=[float(observations[f'{side}_gripper'][0]) for side in ('left', 'right')])
    if sonic_root is not None:
        execution_urdf = Path(sonic_root) / 'gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf'
        result['execution_model_comparison'] = fit_observations(observations, execution_urdf, enforce=False)
        result['execution_model_comparison']['interpretation'] = 'Diagnostic only: do not absorb a different wrist chain into tool offsets or replace physical execution geometry.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sonic-root', type=Path, help='Optional actual model for mismatch diagnostics')
    parser.add_argument('--reference-urdf', type=Path, default=REFERENCE_URDF)
    parser.add_argument('--dataset-parquet', type=Path, required=True)
    parser.add_argument('--max-rows', type=int, default=3000)
    parser.add_argument('--stride', type=int, default=1, help='Candidate row stride before evenly spread sampling')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = calibrate(args.sonic_root, args.dataset_parquet, args.max_rows, args.stride, reference_urdf=args.reference_urdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
