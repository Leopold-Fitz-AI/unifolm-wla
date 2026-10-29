"""Mesh-free G1 FK/IK using the official SONIC URDF and MuJoCo.

The reduced model is ONLY for kinematics. It is never used as the dynamics
model for the official SONIC simulator. Reference poses are pelvis-relative
by default; tool transforms must match the checkpoint's end-effector frame.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .protocol import finite_array

LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
JOINT_NAMES = ([f"{side}_{j}_joint" for side in ("left", "right") for j in LEG]
               + [f"waist_{j}_joint" for j in ("yaw", "roll", "pitch")]
               + [f"{side}_{j}_joint" for side in ("left", "right") for j in ARM])


def pose_matrix(xyz_rpy):
    v = finite_array(xyz_rpy, (6,), "xyz_rpy")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", v[3:]).as_matrix()
    t[:3, 3] = v[:3]
    return t


class G1Kinematics:
    def __init__(self, sonic_root, reference_frame="pelvis", tool_offsets=None):
        urdf = Path(sonic_root) / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
        root = ET.parse(urdf).getroot()
        for link in root.findall("link"):
            for child in list(link):
                if child.tag in ("visual", "collision"):
                    link.remove(child)
        for joint in root.findall("joint"):
            if joint.attrib["name"] not in JOINT_NAMES:
                joint.set("type", "fixed")
        extension = root.find("mujoco")
        if extension is not None:
            root.remove(extension)
        ET.SubElement(ET.SubElement(root, "mujoco"), "compiler", fusestatic="false", discardvisual="true")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        ids = [self.model.joint(name).id for name in JOINT_NAMES]
        self.qadr = self.model.jnt_qposadr[ids]
        self.vadr = self.model.jnt_dofadr[ids]
        self.bounds = self.model.jnt_range[ids].copy()
        self.ref_id = self.model.body(reference_frame).id
        self.ee_ids = [self.model.body(f"{side}_wrist_yaw_link").id for side in ("left", "right")]
        offsets = tool_offsets or [[0] * 6, [0] * 6]
        if len(offsets) != 2:
            raise ValueError("two tool offsets required")
        self.tool = [pose_matrix(v) for v in offsets]

    def check_q(self, q):
        q = finite_array(q, (29,), "body_q")
        if np.any(q < self.bounds[:, 0] - 1e-5) or np.any(q > self.bounds[:, 1] + 1e-5):
            raise ValueError("joint reference exceeds official URDF position limits")
        return np.clip(q, self.bounds[:, 0], self.bounds[:, 1])

    def seed_q(self, measured_q, soft_limit_tolerance=0.03, planner_legs=False):
        """MuJoCo soft joint limits may overshoot slightly under contact/PD.

        Only the optimization seed is clipped. Observation FK still uses the
        measured state; output references continue to pass strict check_q().
        """
        q = finite_array(measured_q, (29,), "measured body_q")
        clipped = np.clip(q, self.bounds[:, 0], self.bounds[:, 1])
        error=np.abs(q-clipped)
        # Planner mode never commands these 12 joints through this arm IK.
        # Native SONIC's measured soft-limit overshoot must not invalidate an
        # otherwise valid arm seed. Raw leg observations remain unmodified.
        if planner_legs:error[:12]=0
        if np.max(error) > soft_limit_tolerance:
            i = int(np.argmax(error))
            raise ValueError(f"measured {JOINT_NAMES[i]}={q[i]:.5f} exceeds URDF {self.bounds[i].tolist()} "
                             f"by more than soft-limit tolerance {soft_limit_tolerance}")
        return clipped

    def forward(self, q):
        self.data.qpos[self.qadr] = finite_array(q, (29,), "body_q")
        mujoco.mj_forward(self.model, self.data)
        def body_t(idx):
            t = np.eye(4)
            t[:3, :3] = self.data.xmat[idx].reshape(3, 3)
            t[:3, 3] = self.data.xpos[idx]
            return t
        inv_ref = np.linalg.inv(body_t(self.ref_id))
        return [inv_ref @ body_t(idx) @ tool for idx, tool in zip(self.ee_ids, self.tool)]

    def solve(self, targets, seed, position_tolerance=0.005, rotation_tolerance=0.05):
        """Hold all 15 leg/waist values fixed; solve the 14 arm joints."""
        q = self.check_q(seed).copy()
        seed_arms = q[15:].copy()
        target = [np.asarray(t, dtype=float) for t in targets]
        def residual(arms):
            q[15:] = arms
            actual = self.forward(q)
            result = []
            for current, desired in zip(actual, target):
                result.extend(current[:3, 3] - desired[:3, 3])
                result.extend(0.2 * Rotation.from_matrix(desired[:3, :3].T @ current[:3, :3]).as_rotvec())
            # Each 7-DOF arm is redundant. A small posture cost prevents a
            # numerically valid pose from causing a large null-space joint jump.
            return np.r_[result, .005 * (arms - seed_arms)]
        def jacobian(arms):
            q[15:] = arms
            actual = self.forward(q)
            world_to_ref = self.data.xmat[self.ref_id].reshape(3, 3).T
            blocks = []
            for idx, tool, current, desired in zip(self.ee_ids, self.tool, actual, target):
                point = self.data.xpos[idx] + self.data.xmat[idx].reshape(3, 3) @ tool[:3, 3]
                jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, jp, jr, point, idx)
                r = Rotation.from_matrix(desired[:3, :3].T @ current[:3, :3]).as_rotvec()
                theta = np.linalg.norm(r)
                x, y, z = r
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                coefficient = 1 / 12 if theta < 1e-4 else (1 - .5 * theta / np.tan(.5 * theta)) / theta**2
                log_jacobian = np.eye(3) - .5 * skew + coefficient * skew @ skew
                blocks.append(np.vstack([world_to_ref @ jp[:, self.vadr[15:]],
                    .2 * log_jacobian @ desired[:3, :3].T @ world_to_ref @ jr[:, self.vadr[15:]]]))
            return np.vstack([*blocks, .005 * np.eye(14)])
        result = least_squares(residual, q[15:], jac=jacobian, bounds=(self.bounds[15:, 0], self.bounds[15:, 1]),
                               max_nfev=80, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        q[15:] = result.x
        errors = residual(result.x)[:12].reshape(2, 6)
        self.last_position_error = float(np.max(np.linalg.norm(errors[:, :3], axis=1)))
        self.last_rotation_error = float(np.max(np.linalg.norm(errors[:, 3:], axis=1)) / .2)
        if (np.max(np.linalg.norm(errors[:, :3], axis=1)) > position_tolerance
                or np.max(np.linalg.norm(errors[:, 3:], axis=1)) / 0.2 > rotation_tolerance):
            raise ValueError(f"IK target unreachable within tolerances: {errors.tolist()}")
        return self.check_q(q).copy()

    def observation_poses(self, q):
        # rot6d is column-major [R00,R10,R20,R01,R11,R21], NOT reshape(-1).
        return [np.concatenate([t[:3, 3], t[:3, :2].T.reshape(-1)]).astype(np.float32)
                for t in self.forward(q)]
