"""Observation / action adapters between the Unitree WBC client and the unified
54-dim action / 60-dim state representation.

Scope: **Dex1 (two-finger gripper) only.** ``_ACTIVE_SLOT_SPECS`` mirrors the
``unitree_base`` entry of
``unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml``. It does NOT
contain ``left_fig6d`` / ``right_fig6d`` (five-finger hand joints) or
``base_pose`` (whole-body relative pose), because those only exist in
``unitree_fullbody_base`` (the WBT datasets).

Supporting a WBT checkpoint therefore needs two changes, and they are in the
*server*, not here (see ``docs/train_action_expert.md``):

  1. add ``left_fig6d`` / ``right_fig6d`` / ``base`` entries to
     ``_ACTIVE_SLOT_SPECS`` below, and
  2. extend the client obs/action protocol in
     ``action_server_wbc_msgpack_unitree.py`` -- the obs side is ``_DATA_KEYS``
     plus the ``observation.state.*`` reads in ``_build_state_unnorm``, and the
     action side is ``_encode_action``.

Everything in this module is consumed by the server through exactly four names
(see its import block): ``WBC_IMAGE_ROLES``, ``build_unitree_fullbody_action_mask``,
``_compose_ee9`` and ``_rot6d_to_matrix``. An earlier revision also carried an
unused and broken 50-dim "active rot6d" layout pipeline; it was removed because
it was dead, crashed with ``KeyError`` on the Dex1 slot spec, and duplicated a
third wire format that nothing implements.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_WORKSPACE_ROOT = str(Path(__file__).resolve().parents[1])
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES  # noqa: E402
from unifolm_wla.dataloader.multi_source_dataset.se3_utils import (  # noqa: E402
    matrix_to_rot6d,
    pose_to_se3,
    rotvec_to_matrix,
)

# (slot name, unified-action slice name(s), unified-state slice name, is_pose)
# `state_key` and `is_pose` are unused by the live server path (it composes EE
# poses itself) but are kept because they document the slot's semantics and are
# what a WBT extension would need.
_ACTIVE_SLOT_SPECS: List[Tuple[str, "str | Tuple[str, ...]", Optional[str], bool]] = [
    ("left_ee", "left_xyz_rotvec", "left_xyz_rot6d", True),
    ("left_gripper", "left_gripper", None, False),
    ("right_ee", "right_xyz_rotvec", "right_xyz_rot6d", True),
    ("right_gripper", "right_gripper", None, False),
    ("waist_joint", "waist_joint", None, False),
    ("base_command", ("base_vx_vy", "base_vw", "height"), None, False),
    ("left_leg_joint", "left_leg_joint", None, False),
    ("right_leg_joint", "right_leg_joint", None, False),
]


def build_unitree_fullbody_action_mask() -> np.ndarray:
    """Boolean (54,) mask of the unified action dims this server actually drives.

    "fullbody" here means arms **plus** lower body (waist, legs, base command),
    as opposed to an arms-only source -- not the WBT dataset family. The result
    is the same mask the Dex1 training targets were produced with
    (``arm_type: dual_with_legs`` in ``unitree.yaml``), so it must match
    ``_ACTIVE_SLOT_SPECS``.
    """
    mask = np.zeros(SLICES["right_leg_joint"].stop, dtype=bool)
    for _name, raw_key, _state_key, _is_rot9d in _ACTIVE_SLOT_SPECS:
        for k in (raw_key if isinstance(raw_key, tuple) else (raw_key,)):
            mask[SLICES[k]] = True
    return mask


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(...,6) rot6d -> (...,3,3) via Gram-Schmidt (inverse of `se3_utils.matrix_to_rot6d`).

    NOTE: Gram-Schmidt *re-orthonormalises*, so a malformed rot6d from the client
    is silently repaired into a plausible rotation rather than rejected. That is
    why the server validates the incoming rot6d separately
    (``ActionSafetyLimiter.check_rot6d``) before it reaches this function.
    """
    shape = rot6d.shape[:-1]
    r = rot6d.reshape(-1, 6)
    a1, a2 = r[:, 0:3], r[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=1)
    R = np.stack([b1, b2, b3], axis=-1)
    return R.reshape(*shape, 3, 3)


def _compose_ee9(anchor_9: np.ndarray, rel_6_chunk: np.ndarray) -> np.ndarray:
    """Compose a (T,6) xyz+rotvec relative action chunk onto a (9,) xyz+rot6d
    absolute state anchor (EE convention: ``STATE_SLICES["*_xyz_rot6d"]``).

    ``T_abs = T_curr @ T_rel`` -- the exact inverse of the training-time
    ``T_rel = T_curr^-1 @ T_future`` (`se3_utils.compute_relative_actions`), which
    is what makes a relative-action model deployable without knowing anything
    about the robot's base frame.

    Returns (T,9) absolute xyz+rot6d.
    """
    xyz_curr = anchor_9[:3]
    R_curr = _rot6d_to_matrix(anchor_9[3:9].reshape(1, 6))[0]
    T_curr = pose_to_se3(xyz_curr, R_curr)  # (4,4)

    R_rel = rotvec_to_matrix(rel_6_chunk[:, 3:6])  # (T,3,3)
    T_rel = pose_to_se3(rel_6_chunk[:, :3], R_rel)  # (T,4,4)
    T_abs = T_curr @ T_rel  # (T,4,4) broadcast

    xyz_abs = T_abs[:, :3, 3]
    rot6d_abs = matrix_to_rot6d(T_abs[:, :3, :3])
    return np.concatenate([xyz_abs, rot6d_abs], axis=1).astype(np.float32)  # (T,9)


# Image role order advertised to the client, aligned with the server's
# `_IMAGE_KEYS`. The client's key names differ from the dataset role names:
#   observation.images.cam_left_high  -> head_left
#   observation.images.cam_left_wrist -> cam_wrist_left
#   observation.images.cam_right_wrist-> cam_wrist_right
WBC_IMAGE_ROLES = ["head_left", "cam_wrist_left", "cam_wrist_right"]
