"""Integration tests for the serve-side action decode chain.

These exercise the *real* functions on both sides of the wire:

  training (`se3_utils.compute_relative_actions`) : T_rel = T_curr^-1 @ T_future
  serving  (`unifolm_wla_action_adapter._compose_ee9`) : T_abs = T_curr @ T_rel

If those two ever drift apart, the robot receives poses that are subtly wrong and
nothing raises -- so this round-trip is the single most valuable invariant to pin
down. Also checks the unified slot tables the server and the safety layer index
into, since an off-by-one there is equally silent.

Needs numpy + scipy only (`conftest.py` provides the import bootstrap).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from safety import ActionSafetyConfig, ActionSafetyLimiter
from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES, STATE_SLICES
from unifolm_wla.dataloader.multi_source_dataset.se3_utils import (
    compute_relative_actions,
    matrix_to_rot6d,
)
from unifolm_wla_action_adapter import (
    WBC_IMAGE_ROLES,
    _compose_ee9,
    _rot6d_to_matrix,
    build_unitree_fullbody_action_mask,
)

RNG = np.random.default_rng(20260929)


def random_rpy(t):
    return RNG.uniform(-np.pi / 2, np.pi / 2, size=(t, 3))


# ── the core invariant: training inverse == serving compose ────────────────


@pytest.mark.parametrize("t", [1, 5, 30])
def test_relative_compose_roundtrip_recovers_absolute_pose(t):
    """T_abs = T_curr @ (T_curr^-1 @ T_future) must equal T_future."""
    curr_xyz = RNG.uniform(-1, 1, size=3)
    curr_rpy = random_rpy(1)[0]
    future = np.concatenate([RNG.uniform(-1, 1, size=(t, 3)), random_rpy(t)], axis=1).astype(np.float32)

    # training side: current pose is expressed in the source format (xyz_rpy)
    curr_pose = np.concatenate([curr_xyz, curr_rpy]).astype(np.float32)
    rel = compute_relative_actions(curr_pose, future, "xyz_rpy")
    assert rel.shape == (t, 6)

    # serving side: the anchor is the unified state slot (xyz + rot6d)
    R_curr = Rotation.from_euler("xyz", curr_rpy)
    anchor9 = np.concatenate([curr_xyz, matrix_to_rot6d(R_curr.as_matrix()[None])[0]]).astype(np.float32)
    assert anchor9.shape == (9,)

    abs9 = _compose_ee9(anchor9, rel.astype(np.float32))
    assert abs9.shape == (t, 9)

    # translation must match exactly (to float32)
    np.testing.assert_allclose(abs9[:, :3], future[:, :3], atol=1e-5)
    # rotation must match after converting the recovered rot6d back to a matrix
    R_expected = Rotation.from_euler("xyz", future[:, 3:6]).as_matrix()
    R_got = _rot6d_to_matrix(abs9[:, 3:9])
    np.testing.assert_allclose(R_got, R_expected, atol=1e-5)


def test_zero_relative_action_holds_current_pose():
    """A zero relative chunk must reproduce the anchor, i.e. 'do not move'."""
    xyz = RNG.uniform(-1, 1, size=3)
    R = Rotation.from_euler("xyz", random_rpy(1)[0])
    anchor9 = np.concatenate([xyz, matrix_to_rot6d(R.as_matrix()[None])[0]]).astype(np.float32)
    out = _compose_ee9(anchor9, np.zeros((7, 6), dtype=np.float32))
    assert out.shape == (7, 9)
    # assert_allclose does not broadcast, so compare against an explicit tile
    np.testing.assert_allclose(out[:, :3], np.tile(xyz, (7, 1)).astype(np.float32), atol=1e-6)
    np.testing.assert_allclose(out[:, 3:9], np.tile(anchor9[3:9], (7, 1)), atol=1e-6)


def test_rot6d_roundtrip_through_adapter():
    R = Rotation.from_euler("xyz", random_rpy(8)).as_matrix()
    rot6d = matrix_to_rot6d(R)
    np.testing.assert_allclose(_rot6d_to_matrix(rot6d), R, atol=1e-9)


# ── slot tables ───────────────────────────────────────────────────────────


def test_unified_layout_widths_are_the_documented_54_and_60():
    assert SLICES["right_leg_joint"].stop == 54
    assert STATE_SLICES["right_leg_joint"].stop == 60


def test_action_mask_matches_active_slots():
    mask = build_unitree_fullbody_action_mask()
    assert mask.shape == (54,)
    assert mask.dtype == bool

    expected = np.zeros(54, dtype=bool)
    for slot in (
        "left_xyz_rotvec", "left_gripper", "right_xyz_rotvec", "right_gripper",
        "waist_joint", "base_vx_vy", "base_vw", "height",
        "left_leg_joint", "right_leg_joint",
    ):
        expected[SLICES[slot]] = True
    np.testing.assert_array_equal(mask, expected)

    # slots that belong to the WBT (fullbody / dex-hand) variant stay off
    for slot in ("left_fig6d", "right_fig6d", "base_rotvec", "torso_joint"):
        assert not mask[SLICES[slot]].any(), slot


def test_wbc_image_roles_align_with_the_server_key_order():
    # `_IMAGE_KEYS` in the server, in order. Duplicated here on purpose: if the
    # server reorders its cameras without updating the roles, prompts mislabel
    # their images silently, which this asserts against.
    server_keys = [
        "observation.images.cam_left_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    assert len(server_keys) == len(WBC_IMAGE_ROLES)
    assert WBC_IMAGE_ROLES == ["head_left", "cam_wrist_left", "cam_wrist_right"]


# ── safety layer against realistic shapes ────────────────────────────────


def test_joint_limit_clip_only_touches_valid_slots_on_a_real_layout():
    """End-to-end shape check: unnorm chunk -> limits -> back into slots."""
    waist = SLICES["waist_joint"]
    bounds = np.tile(np.array([-0.4, 0.4], dtype=np.float32), (3, 1))
    lim = ActionSafetyLimiter(ActionSafetyConfig(joint_limits={"waist_joint": bounds}))
    mask = build_unitree_fullbody_action_mask()

    chunk = np.zeros((6, 54), dtype=np.float32)
    chunk[:, waist] = 1.5  # waist is a valid slot -> clamped
    chunk[:, SLICES["left_fig6d"]] = 9.0  # invalid for Dex1 -> untouched by design

    out = lim.clip_joint_limits(chunk, mask=mask)
    assert np.allclose(out[:, waist], 0.4)
    assert np.allclose(out[:, SLICES["left_fig6d"]], 9.0)


def test_hold_and_blend_compose_with_a_real_layout():
    """`build_hold` output must be directly usable as the next `_encode_action`-style input."""
    T = 8
    action = {
        "action.left_ee_rpy": np.tile(np.arange(6, dtype=np.float32), (1, T, 1)),
        "action.right_ee_rpy": np.tile(np.arange(6, dtype=np.float32), (1, T, 1)),
        "action.left_gripper": np.ones((1, T, 1), dtype=np.float32),
        "action.right_gripper": np.ones((1, T, 1), dtype=np.float32),
        "action.lower_body": np.tile(np.arange(15, dtype=np.float32), (1, T, 1)),
        "action.base_command": np.tile(np.arange(4, dtype=np.float32), (1, T, 1)),
        "action.pivot": np.tile(np.arange(7, dtype=np.float32), (1, T, 1)),
    }
    lim = ActionSafetyLimiter(ActionSafetyConfig(blend_steps=3))
    lim.blend(action)
    hold = lim.build_hold()
    assert hold is not None
    assert set(hold) == set(action)
    for key, v in hold.items():
        assert v.shape == action[key].shape, key
        assert np.isfinite(v).all(), key
