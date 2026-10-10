"""Unit tests for `model_server/safety.py`.

Run with plain pytest (`python -m pytest model_server/tests -q`). Needs only
numpy + scipy + pyyaml; no torch, no GPU. The import bootstrap lives in
`conftest.py` -- see there for why it is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from safety import (
    ActionSafetyConfig,
    ActionSafetyLimiter,
    load_joint_limits,
)
from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES

T = 12
ACTION_SPEC = {
    "action.left_ee_rpy": 6,
    "action.right_ee_rpy": 6,
    "action.left_gripper": 1,
    "action.right_gripper": 1,
    "action.lower_body": 15,
    "action.base_command": 4,
    "action.pivot": 7,
}


def make_action(seed: int = 0, t: int = T, fill: float | None = None) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for key, dim in ACTION_SPEC.items():
        if key in ("action.left_ee_rpy", "action.right_ee_rpy"):
            block = np.zeros((t, dim), dtype=np.float32)
            block[:, :3] = rng.normal(0, 0.05, size=(t, 3))  # xyz
            block[:, 3:] = rng.normal(0, 0.02, size=(t, 3))  # small rpy
        elif fill is not None:
            block = np.full((t, dim), fill, dtype=np.float32)
        else:
            block = rng.normal(0, 0.1, size=(t, dim)).astype(np.float32)
        out[key] = block[None]
    return out


# ── 1. clamp ──────────────────────────────────────────────────────────────


def test_clamp_disabled_is_identity():
    lim = ActionSafetyLimiter(ActionSafetyConfig(clamp_norm=0.0))
    x = np.full((T, 54), 99.0, dtype=np.float32)
    assert np.array_equal(lim.clamp_normalized(x), x)


def test_clamp_bounds_and_counts():
    lim = ActionSafetyLimiter(ActionSafetyConfig(clamp_norm=2.0))
    x = np.zeros((4, 54), dtype=np.float32)
    x[1, 3] = 7.0
    x[2, 4] = -9.0
    out = lim.clamp_normalized(x)
    assert out[1, 3] == pytest.approx(2.0)
    assert out[2, 4] == pytest.approx(-2.0)
    assert np.max(np.abs(out)) <= 2.0
    assert lim.stats()["clamp_dims"] == 2


def test_clamp_respects_mask():
    """Masked-out slots are authoritative zeros and must not be rewritten."""
    mask = np.zeros(54, dtype=bool)
    mask[:9] = True  # only the left-EE block is valid
    lim = ActionSafetyLimiter(ActionSafetyConfig(clamp_norm=2.0))
    x = np.zeros((3, 54), dtype=np.float32)
    x[0, :9] = 5.0  # valid block: must be clamped
    x[0, 40] = 5.0  # invalid slot: must be left alone
    out = lim.clamp_normalized(x, mask=mask)
    assert out[0, 0] == pytest.approx(2.0)
    assert out[0, 40] == pytest.approx(5.0)
    assert out[1, 40] == 0.0


# ── 2. rate limit ─────────────────────────────────────────────────────────


def test_rate_limit_disabled_is_identity():
    lim = ActionSafetyLimiter(ActionSafetyConfig(max_step_norm=0.0))
    x = np.zeros((T, 54), dtype=np.float32)
    x[:, 0] = np.arange(T) * 100.0
    assert np.array_equal(lim.rate_limit(x), x)


def test_rate_limit_bounds_consecutive_differences():
    cap = 0.25
    lim = ActionSafetyLimiter(ActionSafetyConfig(max_step_norm=cap))
    x = np.zeros((T, 54), dtype=np.float32)
    x[:, 0] = np.arange(T) * 10.0  # absurd ramp
    out = lim.rate_limit(x)
    d = np.abs(np.diff(out[:, 0]))
    assert d.max() <= cap + 1e-6
    assert out[0, 0] == 0.0  # first step is untouched
    assert lim.stats()["rate_limit_dims"] > 0


def test_rate_limit_keeps_flat_signal_flat():
    lim = ActionSafetyLimiter(ActionSafetyConfig(max_step_norm=0.5))
    x = np.full((T, 54), 3.0, dtype=np.float32)
    out = lim.rate_limit(x)
    assert np.allclose(out, 3.0)
    assert "rate_limit_dims" not in lim.stats()


# ── 3. joint limits ───────────────────────────────────────────────────────


def test_joint_limits_noop_when_unset():
    lim = ActionSafetyLimiter(ActionSafetyConfig())
    x = np.full((T, 54), 1e3, dtype=np.float32)
    assert np.array_equal(lim.clip_joint_limits(x), x)


def test_joint_limits_clip_and_skip_masked_columns():
    sl = SLICES["left_leg_joint"]
    width = sl.stop - sl.start
    bounds = np.tile(np.array([-0.5, 0.5], dtype=np.float32), (width, 1))
    lim = ActionSafetyLimiter(ActionSafetyConfig(joint_limits={"left_leg_joint": bounds}))

    mask = np.ones(54, dtype=bool)
    mask[sl.start] = False  # pretend the first leg dim is absent
    x = np.zeros((T, 54), dtype=np.float32)
    x[:, sl.start] = 5.0  # masked -> must survive
    x[:, sl.start + 1] = 5.0  # valid -> clamped to 0.5
    out = lim.clip_joint_limits(x, mask=mask)
    assert out[0, sl.start] == pytest.approx(5.0)
    assert out[0, sl.start + 1] == pytest.approx(0.5)
    assert lim.stats()["joint_limit_dims"] > 0


def test_load_joint_limits_rejects_wrong_width(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("left_leg_joint:\n  lower: [0.0, 0.0]\n  upper: [1.0, 1.0]\n")
    with pytest.raises(ValueError, match="expects 6 bounds"):
        load_joint_limits(p)


def test_load_joint_limits_rejects_unknown_slot(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("not_a_slot:\n  lower: [0.0]\n  upper: [1.0]\n")
    with pytest.raises(ValueError, match="unknown action slot"):
        load_joint_limits(p)


def test_load_joint_limits_skips_nulls_and_accepts_both_forms(tmp_path):
    p = tmp_path / "ok.yaml"
    p.write_text(
        "waist_joint:\n  lower: null\n  upper: null\n"
        "left_gripper:\n  lower: [0.0]\n  upper: [1.0]\n"
        "right_gripper: [[0.0], [1.0]]\n"
    )
    limits = load_joint_limits(p)
    assert set(limits) == {"left_gripper", "right_gripper"}
    assert limits["left_gripper"][0].tolist() == [0.0, 1.0]


def test_load_joint_limits_rejects_inverted_bounds(tmp_path):
    p = tmp_path / "inv.yaml"
    p.write_text("left_gripper:\n  lower: [1.0]\n  upper: [0.0]\n")
    with pytest.raises(ValueError, match="lower bound above"):
        load_joint_limits(p)


# ── 4. blending ───────────────────────────────────────────────────────────


def test_blend_disabled_returns_input_and_stores_it():
    lim = ActionSafetyLimiter(ActionSafetyConfig(blend_steps=0))
    a = make_action(seed=1)
    out = lim.blend(a)
    assert out is a
    assert lim._prev_action is not None


def test_blend_without_previous_chunk_passes_through():
    lim = ActionSafetyLimiter(ActionSafetyConfig(blend_steps=5))
    a = make_action(seed=1)
    out = lim.blend(a)
    assert all(np.array_equal(out[k], a[k]) for k in a)


def test_blend_first_step_equals_previous_last_step():
    lim = ActionSafetyLimiter(ActionSafetyConfig(blend_steps=4))
    prev = make_action(seed=2)
    lim.blend(prev)  # establish the previous chunk
    cur = make_action(seed=3)
    out = lim.blend(cur)
    for key in ACTION_SPEC:
        assert out[key][0, 0] == pytest.approx(prev[key][0, -1], abs=1e-5), key
    # steps past the blend window are untouched
    for key in ACTION_SPEC:
        assert np.allclose(out[key][0, 4:], cur[key][0, 4:]), key


def test_blend_is_convex_combination_no_drift():
    """Successive blends must stay inside the hull of their inputs (no blow-up)."""
    lim = ActionSafetyLimiter(ActionSafetyConfig(blend_steps=6))
    key = "action.lower_body"
    lo, hi = -1e9, 1e9
    chunks = [make_action(seed=s) for s in range(6)]
    for c in chunks:
        out = lim.blend(c)
        lo = min(lo, c[key].min())
        hi = max(hi, c[key].max())
    assert out[key].min() >= lo - 1e-5
    assert out[key].max() <= hi + 1e-5


def test_blend_uses_slerp_not_componentwise_rpy():
    """+172 deg -> -172 deg must take the short way (+/-pi), not sweep through 0."""
    lim = ActionSafetyLimiter(ActionSafetyConfig(blend_steps=3))
    prev = make_action(seed=4)
    for key in ("action.left_ee_rpy", "action.right_ee_rpy"):
        prev[key][:, :] = 0.0
        prev[key][0, -1, 3] = 3.0  # ~+172 deg about x
    lim.blend(prev)

    cur = make_action(seed=5)
    for key in ("action.left_ee_rpy", "action.right_ee_rpy"):
        cur[key][:, :] = 0.0
        cur[key][0, :, 3] = -3.0  # ~-172 deg about x

    out = lim.blend(cur)
    mid = out["action.left_ee_rpy"][0, :3, 3]
    # SLERP stays near +/-pi; a naive rpy lerp would produce 1.0, 0.0, -1.0
    assert np.all(np.abs(mid) > 2.5), mid
    assert np.all(np.abs(mid) <= np.pi + 1e-3), mid


def test_blend_stores_blended_output_for_next_seam():
    lim = ActionSafetyLimiter(ActionSafetyConfig(blend_steps=3))
    lim.blend(make_action(seed=6))
    out = lim.blend(make_action(seed=7))
    # the stored tail must match what was emitted, else seams drift
    for key in ACTION_SPEC:
        assert np.allclose(lim._prev_action[key][0, -1], out[key][0, -1]), key


# ── 5. watchdog hold ──────────────────────────────────────────────────────


def test_build_hold_none_without_previous_chunk():
    lim = ActionSafetyLimiter(ActionSafetyConfig())
    assert lim.build_hold() is None


def test_build_hold_freezes_pose_and_zeroes_velocity():
    lim = ActionSafetyLimiter(ActionSafetyConfig(blend_steps=0))
    prev = make_action(seed=8)
    lim.blend(prev)
    hold = lim.build_hold()
    assert hold is not None
    velocity_slices = {"action.base_command": slice(0, 3), "action.pivot": slice(0, 3)}
    for key in ACTION_SPEC:
        assert hold[key].shape == prev[key].shape
        last = prev[key][0, -1]
        vel = velocity_slices.get(key)
        if vel is None:
            # every step equals the previous chunk's last step
            assert np.allclose(hold[key][0], last[None, :]), key
        else:
            # pose-ish components are frozen ...
            assert np.allclose(hold[key][0, :, vel], 0.0), key
            assert np.allclose(hold[key][0, :, vel.stop :], last[vel.stop :][None, :]), key
    assert lim.stats()["holds"] == 1


# ── 6. input validation ───────────────────────────────────────────────────


def test_check_cameras_raises_and_names_the_missing_key():
    lim = ActionSafetyLimiter(ActionSafetyConfig(require_all_cameras=True))
    with pytest.raises(ValueError, match="observation.images.cam_right_wrist"):
        lim.check_cameras(["observation.images.cam_right_wrist"])
    assert lim.stats()["camera_errors"] == 1


def test_check_cameras_can_be_relaxed():
    lim = ActionSafetyLimiter(ActionSafetyConfig(require_all_cameras=False))
    lim.check_cameras(["whatever"])  # must not raise


def test_check_rot6d_accepts_orthonormal_and_rejects_otherwise():
    lim = ActionSafetyLimiter(ActionSafetyConfig(strict_rot6d=False))
    good = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    lim.check_rot6d(good, "left_ee_6d")
    assert "rot6d_errors" not in lim.stats()

    bad = good.copy()
    bad[3] = 2.0  # non-unit first column
    lim.check_rot6d(bad, "left_ee_6d")  # warn only
    assert lim.stats()["rot6d_errors"] == 1


def test_check_rot6d_strict_raises():
    lim = ActionSafetyLimiter(ActionSafetyConfig(strict_rot6d=True))
    bad = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.float32)  # parallel columns
    with pytest.raises(ValueError, match="not an orthonormal rot6d"):
        lim.check_rot6d(bad, "left_ee_6d")


# ── lifecycle ─────────────────────────────────────────────────────────────


def test_reset_clears_chunk_state_and_stats():
    lim = ActionSafetyLimiter(ActionSafetyConfig(blend_steps=3))
    lim.blend(make_action(seed=9))
    lim.clamp_normalized(np.full((T, 54), 50.0, dtype=np.float32))
    assert lim._prev_action is not None
    assert lim.stats()

    lim.reset()
    assert lim._prev_action is None
    assert lim.stats() == {}
    assert lim.build_hold() is None


def test_disabled_limiter_is_a_pure_pass_through():
    lim = ActionSafetyLimiter(ActionSafetyConfig(enabled=False, blend_steps=5, clamp_norm=0.01))
    x = np.full((T, 54), 100.0, dtype=np.float32)
    assert np.array_equal(lim.clamp_normalized(x), x)
    assert np.array_equal(lim.rate_limit(x), x)
    assert np.array_equal(lim.clip_joint_limits(x), x)
    assert lim.build_hold() is None
    assert lim.stats() == {}


def test_config_summary_is_json_able():
    import json

    lim = ActionSafetyLimiter(
        ActionSafetyConfig(joint_limits={"waist_joint": np.zeros((3, 2), dtype=np.float32)})
    )
    s = json.dumps(lim.config_summary())  # must not raise
    assert "waist_joint" in s
