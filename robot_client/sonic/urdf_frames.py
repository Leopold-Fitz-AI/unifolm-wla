"""Read-only URDF frame evaluation for calibration; no physics or actuation."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


class UrdfFrames:
    """Offline FK from joint origins/axes; never changes the execution model."""
    def __init__(self, path):
        self.path = Path(path)
        root = ET.parse(self.path).getroot()
        self.links = {link.attrib['name'] for link in root.findall('link')}
        self.joints = {j.find('child').attrib['link']: j for j in root.findall('joint')}

    def forward(self, joint_values, links=('left_wrist_yaw_link', 'right_wrist_yaw_link'), reference='pelvis'):
        values = {name: np.asarray(q, float) for name, q in joint_values.items()}
        n = len(next(iter(values.values())))
        if any(q.shape != (n,) or not np.isfinite(q).all() for q in values.values()):
            raise ValueError('joint observations must have finite matching lengths')
        missing = set(values) - {j.attrib['name'] for j in self.joints.values()}
        if missing:
            raise ValueError(f'URDF lacks joints: {sorted(missing)}')
        cache = {}
        def frame(link):
            if link not in self.links:
                raise ValueError(f'unknown URDF link: {link}')
            if link in cache:
                return cache[link]
            transform = np.repeat(np.eye(4)[None], n, axis=0)
            joint = self.joints.get(link)
            if joint is not None:
                origin = joint.find('origin')
                if origin is not None:
                    transform[:, :3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                    transform[:, :3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
                if joint.attrib['name'] in values:
                    if joint.attrib['type'] not in ('revolute', 'continuous'):
                        raise ValueError('body calibration requires rotational joints')
                    axis = np.fromstring(joint.find('axis').attrib['xyz'], sep=' ')
                    rotation = np.repeat(np.eye(4)[None], n, axis=0)
                    rotation[:, :3, :3] = Rotation.from_rotvec(values[joint.attrib['name']][:, None] * axis).as_matrix()
                    transform = transform @ rotation
                transform = frame(joint.find('parent').attrib['link']) @ transform
            cache[link] = transform
            return transform
        inverse = np.linalg.inv(frame(reference))
        return [inverse @ frame(link) for link in links]


def observation_joints(observations):
    """Map the dataset's canonical lower-body and two arm channels to names."""
    leg = ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')
    arm = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
    lower = [f'{side}_{j}_joint' for side in ('left', 'right') for j in leg]
    lower += [f'waist_{j}_joint' for j in ('yaw', 'roll', 'pitch')]
    result = {name: np.asarray(observations['lower_body'])[:, i] for i, name in enumerate(lower)}
    for side in ('left', 'right'):
        result.update({f'{side}_{name}_joint': np.asarray(observations[f'{side}_arm'])[:, i] for i, name in enumerate(arm)})
    return result
