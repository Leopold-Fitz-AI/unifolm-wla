"""Observation conversion and timestamped 30 Hz -> 50 Hz references."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import pose_matrix
from .protocol import finite_array, pack_reference

IMAGE_KEYS = ("cam_left_high", "cam_left_wrist", "cam_right_wrist")
ACTION_DIMS = {"left_ee_rpy": 6, "right_ee_rpy": 6, "lower_body": 15,
               "left_gripper": 1, "right_gripper": 1, "base_command": 4}


@dataclass
class Reference:
    times: np.ndarray
    q: np.ndarray
    quat: np.ndarray
    grippers: np.ndarray
    limited_frames: int = 0
    packet_limited: bool = False
    base_commands: np.ndarray | None = None
    max_ik_position_error: float = 0.0
    max_ik_rotation_error: float = 0.0
    truncated_tail_frames: int = 0
    gripper_clamped_values: int = 0
    policy_latency: float = 0.0
    observation_base_pose: np.ndarray | None = None

    def sample(self, times):
        # np.interp deliberately holds the final pose after the horizon.
        q = np.column_stack([np.interp(times, self.times, self.q[:, j]) for j in range(29)])
        grip = np.column_stack([np.interp(times, self.times, self.grippers[:, j]) for j in range(2)])
        return q, np.repeat(self.quat[None], len(times), axis=0), grip

    def packet(self, now, epoch, rate=50, lookahead=51, previous=None, max_velocity=None):
        first = max(0, int(np.floor((now - epoch) * rate + 1e-7)))
        indices = np.arange(first, first + lookahead, dtype=np.int64)
        q, quat, grip = self.sample(epoch + indices / rate)
        self.packet_limited = False
        if previous is not None and max_velocity is not None:
            last, last_time = previous
            for i in range(len(q)):
                step = max_velocity * (max(0, now - last_time) if i == 0 else 1 / rate)
                limited = last + np.clip(q[i] - last, -step, step)
                self.packet_limited |= bool(np.any(np.abs(limited - q[i]) > 1e-8))
                q[i] = limited
                last = limited
        dq = np.gradient(q, 1 / rate, axis=0)
        return pack_reference(q, dq, quat, indices), q[0], grip[0]


class ActionAdapter:
    def __init__(self, kinematics, model_fps=30, max_joint_velocity=3.0, base_height=0.75,
                 gripper_range=((0, 1), (0, 1)), base_velocity_deadband=(.01, .01, .01),
                 ik_position_tolerance=.005, ik_rotation_tolerance=.05, truncate_infeasible_tail=False,
                 gripper_overshoot_tolerance=0.0, align_first_ee_frame=False):
        if not np.isfinite(model_fps) or model_fps <= 0:
            raise ValueError("model_fps must be positive")
        if not np.isfinite(max_joint_velocity) or max_joint_velocity <= 0:
            raise ValueError("max_joint_velocity must be positive")
        self.kin = kinematics
        self.fps = model_fps
        self.max_velocity = max_joint_velocity
        self.base_height = base_height
        self.planner_mode = False
        self.limit_nominal_velocity = True
        self.ik_position_tolerance = ik_position_tolerance
        self.ik_rotation_tolerance = ik_rotation_tolerance
        self.truncate_infeasible_tail = truncate_infeasible_tail
        self.gripper_tolerance = gripper_overshoot_tolerance
        self.align_first_ee_frame = align_first_ee_frame
        if not np.isfinite(gripper_overshoot_tolerance) or gripper_overshoot_tolerance < 0:
            raise ValueError("gripper overshoot tolerance must be nonnegative and finite")
        if any(not np.isfinite(v) or v <= 0 for v in (ik_position_tolerance, ik_rotation_tolerance)):
            raise ValueError("IK tolerances must be positive and finite")
        self.gripper_range = finite_array(gripper_range, (2, 2), "gripper_range")
        if np.any(self.gripper_range[:, 1] <= self.gripper_range[:, 0]):
            raise ValueError("gripper ranges must increase")
        self.base_deadband = finite_array(base_velocity_deadband, (3,), "base_velocity_deadband")
        if np.any(self.base_deadband < 0):
            raise ValueError("base deadbands must be nonnegative")

    def convert(self, action, seed, base_quat, grippers, request_time):
        arrays = {}
        n = None
        for key, width in ACTION_DIMS.items():
            a = np.asarray(action[f"action.{key}"])
            if a.ndim != 3 or a.shape[0] != 1 or a.shape[2] != width or not 2 <= a.shape[1] <= 300:
                raise ValueError(f"action.{key}: expected (1,T,{width}), 2 <= T <= 300")
            n = a.shape[1] if n is None else n
            arrays[key] = finite_array(a[0], (n, width), key)
        base = arrays["base_command"]
        # Protocol v1 has no root-translation/height command. Do not silently
        # discard locomotion outputs or pretend they are executed by SONIC.
        if not self.planner_mode and (np.any(np.abs(base[:, :3]) > self.base_deadband)
                                    or np.max(np.abs(base[:, 3] - self.base_height)) > 0.02):
            raise ValueError("stationary v1 bridge cannot execute nonzero base velocity or a changed base height; "
                             "planner integration is required for locomotion")
        q = self.kin.check_q(seed).copy()
        alignment = [np.eye(4), np.eye(4)]
        if self.align_first_ee_frame:
            measured = self.kin.forward(q)
            alignment = [m @ np.linalg.inv(pose_matrix(arrays[f"{side}_ee_rpy"][0]))
                         for m, side in zip(measured, ("left", "right"))]
        sequence = [q.copy()]
        max_position_error = max_rotation_error = 0.0
        truncated = 0
        for i in range(n):
            if self.planner_mode:
                # SONIC planner is the sole leg-reference producer in this mode.
                q[12:15] = arrays["lower_body"][i, 12:15]
            else:
                q[:15] = arrays["lower_body"][i]
            try:
                q = self.kin.solve([a @ pose_matrix(arrays[f"{side}_ee_rpy"][i])
                                    for a, side in zip(alignment, ("left", "right"))], q,
                                   position_tolerance=self.ik_position_tolerance,
                                   rotation_tolerance=self.ik_rotation_tolerance)
            except ValueError as exc:
                # Receding-horizon control can use a reachable prefix without
                # executing a distant, infeasible forecast. Preserve its real
                # deadline; a prefix already expired at arrival is discarded.
                if not self.truncate_infeasible_tail or i < 2 or not str(exc).startswith("IK target unreachable"):
                    raise
                truncated, n = n - i, i
                break
            max_position_error = max(max_position_error, self.kin.last_position_error)
            max_rotation_error = max(max_rotation_error, self.kin.last_rotation_error)
            sequence.append(q.copy())
        sequence = np.stack(sequence)
        limited_frames = 0
        for i in range(1, len(sequence)) if self.limit_nominal_velocity else ():
            limited = sequence[i - 1] + np.clip(sequence[i] - sequence[i - 1],
                                                -self.max_velocity / self.fps, self.max_velocity / self.fps)
            limited_frames += int(np.any(np.abs(limited - sequence[i]) > 1e-8))
            sequence[i] = limited
        quat = finite_array(base_quat, (4,), "base_quat")
        if not np.isclose(np.linalg.norm(quat), 1, atol=1e-3):
            raise ValueError("invalid measured base quaternion")
        # A stationary reference preserves heading and requests an upright
        # pelvis. Feeding measured tilt back as a target would reinforce drift.
        yaw = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz")[2]
        xyzw = Rotation.from_euler("z", yaw).as_quat()
        quat = xyzw[[3, 0, 1, 2]]
        grip = np.column_stack([arrays["left_gripper"][:, 0], arrays["right_gripper"][:, 0]])
        if (np.any(grip < self.gripper_range[:, 0] - self.gripper_tolerance)
                or np.any(grip > self.gripper_range[:, 1] + self.gripper_tolerance)):
            raise ValueError("gripper action outside configured checkpoint range")
        bounded_grip = np.clip(grip, self.gripper_range[:, 0], self.gripper_range[:, 1])
        result = Reference(request_time + np.arange(n + 1) / self.fps, sequence, quat,
                           np.vstack([finite_array(grippers, (2,), "grippers"), bounded_grip[:n]]), limited_frames)
        result.base_commands = np.vstack([[0, 0, 0, self.base_height], base[:n]])
        result.max_ik_position_error = max_position_error
        result.max_ik_rotation_error = max_rotation_error
        result.truncated_tail_frames = truncated
        result.gripper_clamped_values = int(np.count_nonzero(bounded_grip[:n] != grip[:n]))
        return result


def make_observation(kin, state, images_bgr, grippers, instruction, unnorm_key=None):
    q = finite_array(state["body_q"], (29,), "body_q")
    poses = kin.observation_poses(q)
    obs = {
        "observation.state.left_ee_6d": poses[0],
        "observation.state.right_ee_6d": poses[1],
        "observation.state.left_gripper": np.array([grippers[0]], dtype=np.float32),
        "observation.state.right_gripper": np.array([grippers[1]], dtype=np.float32),
        "observation.state.lower_body": q[:15].astype(np.float32),
        "instruction": instruction,
    }
    for key in IMAGE_KEYS:
        img = np.asarray(images_bgr[key])
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            raise ValueError(f"{key}: expected uint8 HWC BGR image")
        obs[f"observation.images.{key}"] = img
    if unnorm_key:
        obs["unnorm_key"] = unnorm_key
    return obs


def pose_to_rpy(t):
    return np.r_[t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_euler("xyz")]
