"""Action-chunk safety layer for the Unitree WBC msgpack action server.

The bare server (``action_server_wbc_msgpack_unitree.py``) does exactly one thing:
receive obs -> run the model -> hand back the whole predicted chunk. Everything
between "raw model output" and "what a humanoid is allowed to do" is missing.
This module supplies those stages.

Design rule: **invent no robot constants.** The repository contains no joint
limits and no velocity limits, and ``dataset_statistics.json`` only carries
per-dimension ``offset``/``scale``. So every stage is either

  * *dimensionless* (expressed in normalized action space, where the training
    statistics define the scale), or
  * *caller-supplied* (physical joint ranges come from a file the user fills in).

Stages, in pipeline order:

  1. ``clamp_normalized``   outlier rejection, normalized action space
  2. ``rate_limit``         per-step change bound, normalized action space
  3. ``clip_joint_limits``  physical range clipping, unnormalized action space
  4. ``blend``              cross-chunk seam smoothing, absolute action space
  5. ``build_hold``         watchdog payload, absolute action space
  6. ``check_cameras`` / ``check_rot6d``  input validation

About the normalized-space bounds: ``minmax_q`` normalization maps q01/q99 onto
roughly +/-1, while ``zscore`` maps 1 sigma onto 1. The two therefore have very
different natural ranges, which is why the defaults (5.0 / 10.0) are set to
"catch pathology, never shape the distribution". Tighter values are a legitimate
safety/fidelity tradeoff for zscore dimensions, but they must be validated with
``examples/unifolm_wla/eval_files/unitree/eval_local_episode.py`` first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES

# Keys whose trailing 3 columns are an (x, y, z) Euler angle triple produced by
# `Rotation.as_euler("xyz")` in `_compose_ee_to_xyz_rpy`, and therefore must be
# blended on the rotation manifold rather than component-wise.
_RPY_KEYS: Tuple[str, ...] = ("action.left_ee_rpy", "action.right_ee_rpy")

# Keys/components that are velocities rather than poses. `build_hold` zeroes
# these (a hold means "stop moving"), while pose components are frozen at the
# last commanded value.
_VELOCITY_COMPONENTS: Dict[str, slice] = {
    "action.base_command": slice(0, 3),  # vx, vy, vw   (height at index 3 is kept)
    "action.pivot": slice(0, 3),  # vx, vy, vw   (yaw, pitch, roll, height kept)
}

# Slots that `load_joint_limits` will accept. Deliberately the full unified
# action layout minus nothing -- a caller may bound relative pose deltas or
# base velocities too, not just joint angles.
_LIMITABLE_SLOTS = tuple(SLICES.keys())


@dataclass
class ActionSafetyConfig:
    """All stages are individually switchable; see the module docstring."""

    enabled: bool = True

    # 1. Outlier rejection in normalized space. <= 0 disables.
    clamp_norm: float = 5.0
    # 2. Max |a[k+1] - a[k]| per dimension, normalized space. <= 0 disables.
    max_step_norm: float = 10.0
    # 4. Cross-chunk seam blend length in steps. 0 disables.
    blend_steps: int = 0
    # 5. Watchdog budget for one `predict_action` call, milliseconds. <= 0 disables.
    infer_timeout_ms: float = 2000.0
    # 6. Input validation.
    require_all_cameras: bool = True
    strict_rot6d: bool = False
    rot6d_tol: float = 1e-3

    # 3. Physical limits per unified slot: {slot_name: (width, 2) array of [lo, hi]}.
    joint_limits: Optional[Dict[str, np.ndarray]] = None


class ActionSafetyLimiter:
    """Stateful action post-processor. One instance per server (one episode)."""

    def __init__(self, config: Optional[ActionSafetyConfig] = None) -> None:
        self.cfg = config or ActionSafetyConfig()
        self._prev_action: Optional[Dict[str, np.ndarray]] = None
        self._stats: Dict[str, int] = {}

    # ── bookkeeping ────────────────────────────────────────────────────────

    def _bump(self, key: str, n: int = 1) -> None:
        self._stats[key] = self._stats.get(key, 0) + int(n)

    def reset(self) -> None:
        """Drop cross-chunk state. Call on every episode / policy lifecycle event."""
        self._prev_action = None
        self._stats = {}

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def config_summary(self) -> Dict[str, Any]:
        """JSON-able view of the active policy, advertised via server metadata."""
        return {
            "enabled": bool(self.cfg.enabled),
            "clamp_norm": float(self.cfg.clamp_norm),
            "max_step_norm": float(self.cfg.max_step_norm),
            "blend_steps": int(self.cfg.blend_steps),
            "infer_timeout_ms": float(self.cfg.infer_timeout_ms),
            "require_all_cameras": bool(self.cfg.require_all_cameras),
            "strict_rot6d": bool(self.cfg.strict_rot6d),
            "joint_limit_slots": sorted((self.cfg.joint_limits or {}).keys()),
        }

    @property
    def _active(self) -> bool:
        return bool(self.cfg.enabled)

    # ── 1. outlier clamp (normalized space) ────────────────────────────────

    def clamp_normalized(self, pred_norm: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Bound ``|pred_norm|`` to ``clamp_norm`` on valid dimensions only.

        Mask awareness matters: an invalid slot is *defined* to be 0 (see
        `robot_action_state_processing` S16.18), so it must never be moved by a
        limiter even if a future limit were one-sided.
        """
        limit = float(self.cfg.clamp_norm)
        if not self._active or limit <= 0:
            return pred_norm

        out = np.array(pred_norm, dtype=np.float32, copy=True)
        if mask is None:
            before = np.count_nonzero(np.abs(out) > limit)
            np.clip(out, -limit, limit, out=out)
        else:
            m = np.asarray(mask, dtype=bool).reshape(-1)
            sub = out[:, m]  # fancy indexing -> copy, so write the result back
            before = np.count_nonzero(np.abs(sub) > limit)
            out[:, m] = np.clip(sub, -limit, limit)
        if before:
            self._bump("clamp_dims", int(before))
            logging.warning("[safety] clamp_normalized clipped %d value(s) to +/-%.3f", before, limit)
        return out

    # ── 2. per-step rate limit (normalized space) ─────────────────────────

    def rate_limit(self, pred_norm: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Cumulative clip of consecutive-step differences.

        For the relative-action layout a step-to-step difference is (to first
        order) the commanded *velocity* between two prediction steps, so bounding
        it bounds motion speed without needing any physical unit.
        """
        cap = float(self.cfg.max_step_norm)
        if not self._active or cap <= 0 or pred_norm.shape[0] < 2:
            return pred_norm

        out = np.array(pred_norm, dtype=np.float32, copy=True)
        m = None if mask is None else np.asarray(mask, dtype=bool).reshape(-1)
        clipped = 0
        for k in range(1, out.shape[0]):
            delta = out[k] - out[k - 1]
            if m is None:
                over = np.abs(delta) > cap
                delta = np.clip(delta, -cap, cap)
            else:
                # `delta[m]` is a copy -- assign the clipped result back explicitly.
                sub = delta[m]
                over_m = np.abs(sub) > cap
                over = np.zeros_like(delta, dtype=bool)
                over[m] = over_m
                delta[m] = np.clip(sub, -cap, cap)
            if over.any():
                clipped += int(over.sum())
                out[k] = out[k - 1] + delta
        if clipped:
            self._bump("rate_limit_dims", clipped)
            logging.warning("[safety] rate_limit clipped %d step difference(s) to %.3f", clipped, cap)
        return out

    # ── 3. physical joint limits (unnormalized space) ─────────────────────

    def clip_joint_limits(self, unnorm: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Clip configured slots to their physical ``[lo, hi]`` range.

        Runs in *unnormalized* space because limits are physical quantities.
        No-op when no limits were supplied -- the repository ships none.
        """
        limits = self.cfg.joint_limits
        if not self._active or not limits:
            return unnorm

        out = np.array(unnorm, dtype=np.float32, copy=True)
        m = np.ones(out.shape[-1], dtype=bool) if mask is None else np.asarray(mask, dtype=bool).reshape(-1)
        clipped = 0
        for slot, bounds in limits.items():
            sl = SLICES[slot]
            width = sl.stop - sl.start
            lo, hi = bounds[:, 0], bounds[:, 1]
            for j in range(width):
                col = sl.start + j
                if not m[col]:
                    continue  # invalid slot: leave the canonical 0 untouched
                over = np.count_nonzero((out[:, col] < lo[j]) | (out[:, col] > hi[j]))
                if over:
                    np.clip(out[:, col], lo[j], hi[j], out=out[:, col])
                    clipped += int(over)
        if clipped:
            self._bump("joint_limit_dims", clipped)
            logging.warning("[safety] clip_joint_limits clamped %d value(s)", clipped)
        return out

    # ── 4. cross-chunk seam blending (absolute action space) ──────────────

    def blend(self, action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Ramp the head of a new chunk out of the previous chunk's last step.

        Why here and not on the relative deltas: the client consumes *absolute*
        poses, and consecutive chunks are composed onto different state anchors,
        so the discontinuity only exists in absolute space.

        The blended result (not the raw prediction) is stored as the new
        ``_prev_action`` -- chaining the raw prediction instead would let the
        seam drift across successive chunks.
        """
        n = int(self.cfg.blend_steps)
        if not self._active or n <= 0 or self._prev_action is None:
            self._store(action)
            return action

        n = min(n, self._min_horizon(action))
        if n <= 0:
            self._store(action)
            return action

        out: Dict[str, np.ndarray] = {}
        for key, cur in action.items():
            cur = np.asarray(cur)
            prev = self._prev_action.get(key)
            if prev is None or np.asarray(prev).shape[-1] != cur.shape[-1]:
                out[key] = cur
                continue
            prev_last = np.asarray(prev)[0, -1]
            out[key] = (
                _blend_xyz_rpy(prev_last, cur[0], n)[None]
                if key in _RPY_KEYS
                else _blend_linear(prev_last, cur[0], n)[None]
            )
        self._bump("blend_calls")
        self._store(out)
        return out

    def _store(self, action: Dict[str, np.ndarray]) -> None:
        self._prev_action = {k: np.array(v, dtype=np.float32, copy=True) for k, v in action.items()}

    @staticmethod
    def _min_horizon(action: Dict[str, np.ndarray]) -> int:
        return min((np.asarray(v).shape[1] for v in action.values()), default=0)

    # ── 5. watchdog payload ───────────────────────────────────────────────

    def build_hold(self) -> Optional[Dict[str, np.ndarray]]:
        """Freeze at the last commanded pose, with velocities zeroed.

        Returns ``None`` when there is no previous chunk (i.e. the very first
        request timed out). The caller must then fall back to the real -- slow --
        prediction: synthesising a hold from the state alone is not possible,
        because the unified *state* carries no ``height`` command slot
        (`map_state` never fills `STATE_SLICES["height"]`), so a hold built that
        way would command a physically meaningless height.
        """
        if not self._active or self._prev_action is None:
            return None
        out: Dict[str, np.ndarray] = {}
        for key, prev in self._prev_action.items():
            prev = np.asarray(prev)
            _, T, _ = prev.shape
            held = np.repeat(prev[0, -1][None, :], T, axis=0)
            vdims = _VELOCITY_COMPONENTS.get(key)
            if vdims is not None:
                held[:, vdims] = 0.0
            out[key] = held[None].astype(np.float32)
        self._bump("holds")
        return out

    # ── 6. input validation ───────────────────────────────────────────────

    def check_cameras(self, missing_keys: Sequence[str]) -> None:
        """Fail loudly on a missing camera.

        `_build_example` used to drop undecodable cameras silently and then
        truncate ``image_roles`` to stay aligned, producing a prompt with one
        fewer ``role: <image>`` block than training -- a distribution shift that
        raises nothing and degrades accuracy quietly.
        """
        if not self._active or not missing_keys:
            return
        # A decode failure (not just an absent key) is the caller's signal too.
        msg = (
            f"missing camera observation(s): {list(missing_keys)}. "
            "The model was trained with all cameras present; serving with fewer "
            "silently changes the prompt. Pass --no-require_all_cameras to override."
        )
        self._bump("camera_errors")
        if self.cfg.require_all_cameras:
            raise ValueError(msg)
        logging.warning("[safety] %s", msg)

    def check_rot6d(self, vec9: np.ndarray, name: str) -> None:
        """Verify the caller's 9-dim EE pose really is ``xyz + orthonormal rot6d``.

        `_rot6d_to_matrix` re-orthonormalises via Gram-Schmidt, so a malformed
        rot6d is *silently corrected* into a plausible-looking rotation. Nothing
        downstream can detect it, which is why the check has to live here.
        """
        if not self._active:
            return
        v = np.asarray(vec9, dtype=np.float64).reshape(-1)
        if v.size < 9:
            return
        a1, a2 = v[3:6], v[6:9]
        n1 = float(np.linalg.norm(a1))
        n2 = float(np.linalg.norm(a2))
        if n1 < 1e-8 or n2 < 1e-8:
            residual = float("inf")
        else:
            # Both vectors should be unit length and mutually orthogonal.
            residual = max(abs(n1 - 1.0), abs(n2 - 1.0), abs(float(a1 @ a2)) / (n1 * n2))
        if residual <= self.cfg.rot6d_tol:
            return
        msg = (
            f"{name} is not an orthonormal rot6d (residual={residual:.3e} > tol={self.cfg.rot6d_tol:.1e}). "
            "Expected xyz + [R00,R10,R20,R01,R11,R21] with unit, orthogonal columns."
        )
        self._bump("rot6d_errors")
        if self.cfg.strict_rot6d:
            raise ValueError(msg)
        logging.warning("[safety] %s", msg)


# ── blending helpers ──────────────────────────────────────────────────────


def _blend_weights(n: int) -> np.ndarray:
    """Ramp weights for a seam blend of length ``n``.

    ``w[0] == 0`` so the first emitted step is *exactly* the previous chunk's last
    commanded step (that is the whole point: no jump at the seam), and
    ``w[n-1] == 1`` so the new chunk is fully adopted by the end of the window --
    which keeps step ``n`` continuous with step ``n-1`` too.
    """
    return (np.arange(n, dtype=np.float32) / float(max(n - 1, 1)))[:, None]


def _blend_linear(prev_last: np.ndarray, cur: np.ndarray, n: int) -> np.ndarray:
    """Ramp ``cur[:n]`` out of ``prev_last``."""
    w = _blend_weights(n)
    out = np.array(cur, dtype=np.float32, copy=True)
    out[:n] = (1.0 - w) * prev_last[None, :] + w * cur[:n]
    return out


def _blend_xyz_rpy(prev_last: np.ndarray, cur: np.ndarray, n: int) -> np.ndarray:
    """Linear on xyz, exact SLERP on the rpy triple.

    Component-wise rpy interpolation is wrong across a +/-pi wrap and wrong for
    large angles; scaling the relative rotation vector by the interpolation
    weight is the geodesic (SLERP) at a fraction of the cost
    (`rot_prev * exp(w * log(rot_prev^-1 rot_cur))`).
    """
    out = np.array(cur, dtype=np.float32, copy=True)
    w = _blend_weights(n)
    out[:n, :3] = (1.0 - w) * prev_last[None, :3] + w * cur[:n, :3]

    rot_prev = Rotation.from_euler("xyz", prev_last[3:6].astype(np.float64))
    rot_cur = Rotation.from_euler("xyz", cur[:n, 3:6].astype(np.float64))
    rel = rot_prev.inv() * rot_cur
    rot_blend = rot_prev * Rotation.from_rotvec(rel.as_rotvec() * w)
    out[:n, 3:6] = rot_blend.as_euler("xyz").astype(np.float32)
    return out


# ── limit-file loading ────────────────────────────────────────────────────


def load_joint_limits(path: str | Path) -> Dict[str, np.ndarray]:
    """Load per-slot ``[lo, hi]`` bounds from a YAML/JSON file.

    Accepted per slot, either form::

        waist_joint:      [[-1.0, 1.0], [-0.5, 0.5], [-0.5, 0.5]]   # (width, 2)
        left_leg_joint:
          lower: [-2.6, -0.1, ...]
          upper: [ 2.6,  2.6, ...]

    Shape is validated against ``SLICES`` so a wrong-width file fails here
    instead of silently clamping the wrong columns. ``null`` bounds become
    +/-inf (unbounded on that side).

    NOTE: the repository ships no robot limits. Use
    ``model_server/joint_limits.example.yaml`` as a template and fill in values
    from Unitree's documentation for your specific G1 configuration.
    """
    import json

    import yaml

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"joint-limits file not found: {path}")
    text = path.read_text()
    raw = json.loads(text) if path.suffix == ".json" else (yaml.safe_load(text) or {})
    if not isinstance(raw, dict):
        raise ValueError(f"joint-limits file must contain a mapping, got {type(raw).__name__}")

    def _pair(value: Any) -> Optional[Tuple[List[float], List[float]]]:
        if value is None:
            return None
        if isinstance(value, dict):
            lo, hi = value.get("lower"), value.get("upper")
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            lo, hi = value[0], value[1]
        else:
            raise ValueError(f"unsupported bound spec: {value!r}")
        if lo is None or hi is None:
            return None
        return list(lo), list(hi)

    limits: Dict[str, np.ndarray] = {}
    for slot, value in raw.items():
        if slot not in SLICES:
            raise ValueError(f"unknown action slot {slot!r}; valid slots: {list(_LIMITABLE_SLOTS)}")
        pair = _pair(value)
        if pair is None:
            logging.info("[safety] joint limits for %r left unset -> not clipped", slot)
            continue
        lo, hi = pair
        width = SLICES[slot].stop - SLICES[slot].start
        if len(lo) != width or len(hi) != width:
            raise ValueError(
                f"slot {slot!r} expects {width} bounds per side, got lower={len(lo)}, upper={len(hi)}"
            )
        bounds = np.array(
            [[-np.inf if a is None else float(a), np.inf if b is None else float(b)] for a, b in zip(lo, hi)],
            dtype=np.float32,
        )
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError(f"slot {slot!r} has a lower bound above its upper bound")
        limits[slot] = bounds
    return limits
