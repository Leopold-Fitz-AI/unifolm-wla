#!/usr/bin/env python3
# Copyright 2025 unifolm_wla community. All rights reserved.
# Licensed under the MIT License.
"""Unitree G1 msgpack/websocket action server.

Speaks the exact protocol of
(``WebsocketClientPolicy``):

  - On connect, the server immediately sends msgpack-packed ``metadata``. The
    client reads it as ``self._server_metadata`` and drives what obs keys it
    will send from ``metadata["data_keys"]``.
  - Each request is a msgpack dict ``{"type": <str>, ...}``. ``get_action``
    carries ``{"obs": {...}}`` and expects an action dict back; the other
    lifecycle types (``policy_reset`` / ``episode_end`` / ...) are no-ops for
    this stateless inference server.

Observation keys the client sends (declared in ``metadata["data_keys"]``):
    observation.images.cam_left_high     -> head_left camera   (H,W,3) uint8 BGR
    observation.images.cam_left_wrist    -> left  wrist camera
    observation.images.cam_right_wrist   -> right wrist camera
    observation.state.left_ee_6d         -> (9,) xyz + rot6d (first two cols)
    observation.state.right_ee_6d        -> (9,) xyz + rot6d
    observation.state.left_gripper       -> (1,)
    observation.state.right_gripper      -> (1,)
    observation.state.lower_body         -> (15,) left_leg(6)+right_leg(6)+waist(3)

Each obs value may carry a leading time/batch axis ((T,...) or (1,...)); the
server uses the LAST frame.

Action keys returned (whole predicted chunk, UNNORMALIZED, ABSOLUTE). Each key
carries a leading batch axis of 1:
    action.left_ee_rpy    (1, T, 6)  xyz + rpy, composed onto the current left EE
    action.right_ee_rpy   (1, T, 6)  xyz + rpy, composed onto the current right EE
    action.left_gripper   (1, T, 1)
    action.right_gripper  (1, T, 1)
    action.lower_body     (1, T, 15) left_leg(6)+right_leg(6)+waist(3)
    action.base_command   (1, T, 4)  vx, vy, vw, height
    action.pivot          (1, T, 7)  vx, vy, vw, yaw, pitch, roll, height
        vx/vy/vw/height are the same 4 values as base_command; yaw/pitch/roll
        come from the waist (lower_body last 3, ordered yaw,roll,pitch) remapped
        into pivot's yaw,pitch,roll order.

The EE state the client sends is ALREADY xyz+rot6d (the state layout the model
was trained on: STATE_SLICES["*_xyz_rot6d"]), so it is placed straight into the
unified state -- NOT run through map_state's pose_to_xyz_rot6d_from_format
(which expects a 6/7-dim rpy/quat pose). The rot6d convention (first two matrix
columns, [R00,R10,R20,R01,R11,R21]) matches se3_utils.matrix_to_rot6d exactly.

Usage:
    python -m model_server.action_server_wbc_msgpack_unitree \\
        --ckpt_path playground/Checkpoints/<run_id>/checkpoints/steps_<N>_model.safetensors \\
        --host 0.0.0.0 --port 8600 --instruction "pick up the object"
"""

import argparse
import asyncio
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import websockets.asyncio.server as _ws_server
import websockets.frames
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
from safety import (  # noqa: E402
    ActionSafetyConfig,
    ActionSafetyLimiter,
    load_joint_limits,
)
from tools import msgpack_numpy  # noqa: E402
from tools.debug_saver import StepDebugSaver  # noqa: E402
from unifolm_wla_action_adapter import (  # noqa: E402
    WBC_IMAGE_ROLES,
    build_unitree_fullbody_action_mask,
    _compose_ee9,
    _rot6d_to_matrix,
)

_WORKSPACE_ROOT = str(Path(__file__).resolve().parents[1])
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES, STATE_SLICES, STATE_DIM  # noqa: E402

from unifolm_wla.model.framework.base_framework import baseframework  # noqa: E402
from unifolm_wla.model.framework.share_tools import read_mode_config  # noqa: E402

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Client obs keys -> image role order (matches WBC_IMAGE_ROLES).
_IMAGE_KEYS = [
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

# The obs / action contract advertised to the client via metadata["data_keys"].
_DATA_KEYS = [
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
    "observation.state.left_ee_6d",
    "observation.state.right_ee_6d",
    "observation.state.left_gripper",
    "observation.state.right_gripper",
    "observation.state.lower_body",
    "action.left_ee_rpy",
    "action.right_ee_rpy",
    "action.left_gripper",
    "action.right_gripper",
    "action.lower_body",
    "action.base_command",
    "action.pivot",
]


def _last_frame(arr: np.ndarray) -> np.ndarray:
    """Client may send obs with a leading (T,...)/(1,...) axis. For a state
    vector that means ndim==2 -> take the last row; for an image ndim==4 ->
    take the last frame. A bare vector/image is returned unchanged."""
    a = np.asarray(arr)
    if a.ndim == 2 or a.ndim == 4:
        return a[-1]
    return a


def _compose_ee_to_xyz_rpy(anchor_9: np.ndarray, rel_6_chunk: np.ndarray) -> np.ndarray:
    """Compose a (T,6) xyz+rotvec RELATIVE EE chunk onto a (9,) xyz+rot6d
    ABSOLUTE EE state anchor and return (T,6) xyz + rpy (euler 'xyz').

    Reuses the adapter's `_compose_ee9` for the SE3 compose (T_abs = T_curr @
    T_rel -> xyz+rot6d) and only converts the trailing rot6d to rpy, which is
    the client's action.left/right_ee_rpy convention.
    """
    ee9 = _compose_ee9(anchor_9, rel_6_chunk)                   # (T,9) xyz+rot6d
    xyz_abs = ee9[:, :3]
    rpy_abs = Rotation.from_matrix(_rot6d_to_matrix(ee9[:, 3:9])).as_euler("xyz")
    return np.concatenate([xyz_abs, rpy_abs], axis=1).astype(np.float32)  # (T,6)


# Fallback camera resize target, matching the shipped
# `unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml` (`image_size: [336, 448]`).
_DEFAULT_IMAGE_SIZE: Tuple[int, int] = (336, 448)


def _resolve_image_size(args, mode_config: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Decide the camera resize target ``(H, W)``.

    Priority:
      1. explicit ``--image_size H W`` (pass ``0 0`` to disable resizing),
      2. ``image_size`` read from the training data config recorded in the
         checkpoint (``datasets.vla_data.data_config_path``),
      3. ``_DEFAULT_IMAGE_SIZE``.

    Why the lookup: the model saw camera frames at the *training* resolution, so
    serving at any other resolution is a train/serve mismatch that degrades
    accuracy **without raising anything**. Deriving it from the checkpoint's own
    data config makes this correct by default; when that config is not reachable
    from the serving machine we warn loudly and fall back.
    """
    if args.image_size is not None:
        if all(args.image_size):
            return (int(args.image_size[0]), int(args.image_size[1]))
        logging.warning("--image_size %s contains a zero -> camera resizing disabled",
                        list(args.image_size))
        return None

    data_cfg_path = ((mode_config.get("datasets") or {}).get("vla_data") or {}).get("data_config_path")
    if data_cfg_path:
        try:
            from unifolm_wla.dataloader.multi_source_dataset.config import load_config

            data_cfg = load_config(data_cfg_path)
            if data_cfg.image_size:
                size = (int(data_cfg.image_size[0]), int(data_cfg.image_size[1]))
                logging.info("image_size=%s (auto-detected from %s)", size, data_cfg_path)
                return size
            logging.info("data config %s sets no image_size -> using dataset-native resolution",
                         data_cfg_path)
            return None
        except Exception as exc:  # noqa: BLE001 - never fail startup over this
            logging.warning("could not read image_size from data config %r (%s); "
                            "falling back to %s", data_cfg_path, exc, _DEFAULT_IMAGE_SIZE)
    else:
        logging.warning("checkpoint config records no datasets.vla_data.data_config_path; "
                        "falling back to %s", _DEFAULT_IMAGE_SIZE)

    logging.warning("VERIFY that image_size=%s matches the checkpoint's training resolution, "
                    "or pass --image_size H W explicitly.", _DEFAULT_IMAGE_SIZE)
    return _DEFAULT_IMAGE_SIZE


def _build_safety(args) -> ActionSafetyLimiter:
    """Assemble the action safety policy from CLI args.

    Physical joint ranges are never invented: without ``--joint_limits`` they are
    simply not clipped, and we say so loudly at startup.
    """
    limits = load_joint_limits(args.joint_limits) if args.joint_limits else None
    if args.safety and args.joint_limits and not limits:
        logging.warning(
            "joint-limits file %s defined no usable slot (all bounds null?) -> nothing clipped",
            args.joint_limits)
    elif limits:
        logging.info("joint limits active for slots: %s", sorted(limits))
    elif args.safety:
        logging.warning(
            "no --joint_limits supplied -> physical joint ranges are NOT clipped. "
            "See model_server/joint_limits.example.yaml.")

    cfg = ActionSafetyConfig(
        enabled=bool(args.safety),
        clamp_norm=float(args.action_clip_norm),
        max_step_norm=float(args.action_max_step),
        blend_steps=int(args.action_blend_steps),
        infer_timeout_ms=float(args.infer_timeout_ms),
        require_all_cameras=bool(args.require_all_cameras),
        strict_rot6d=bool(args.strict_rot6d),
        joint_limits=limits,
    )
    if cfg.enabled and cfg.blend_steps <= 0:
        logging.info(
            "cross-chunk seam blending is OFF (--action_blend_steps 0). Enable it if the client "
            "jumps straight to each new chunk's first step.")
    return ActionSafetyLimiter(cfg)


class ActionServerWBCMsgpack:

    def __init__(self, args):
        self.args = args
        self.instruction = args.instruction
        logging.info("Loading model from: %s", args.ckpt_path)
        self.model = baseframework.from_pretrained(args.ckpt_path, base_vlm=args.base_vlm)
        if args.use_bf16:
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(DEVICE).eval()

        mode_config, _ = read_mode_config(args.ckpt_path)
        am_cfg = mode_config.get("framework", {}).get("action_model", {})
        self._action_chunk_size = int(am_cfg.get("action_horizon", 30))
        logging.info("action_chunk_size=%d", self._action_chunk_size)

        self._norm_stats_all = self.model.norm_stats
        self._norm_arrays = {
            key: {
                "action_offset": np.array(stats["action"]["offset"], dtype=np.float32),
                "action_scale": np.array(stats["action"]["scale"], dtype=np.float32),
                "state_offset": np.array(stats["state"]["offset"], dtype=np.float32),
                "state_scale": np.array(stats["state"]["scale"], dtype=np.float32),
            }
            for key, stats in self._norm_stats_all.items()
        }
        if args.unnorm_key is not None:
            self._default_unnorm_key = args.unnorm_key
        elif len(self._norm_stats_all) == 1:
            self._default_unnorm_key = next(iter(self._norm_stats_all.keys()))
        else:
            self._default_unnorm_key = None
            logging.warning(
                "Multi-dataset checkpoint; client must send unnorm_key per request. Available: %s",
                list(self._norm_stats_all.keys()))

        self._action_mask = build_unitree_fullbody_action_mask()  # (54,) bool
        self._image_size = _resolve_image_size(args, mode_config)
        self._safety = _build_safety(args)
        logging.info("safety=%s", self._safety.config_summary())

        cv2.setNumThreads(0)
        self._img_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="imgdecode")

        self._debug_saver = StepDebugSaver(args.debug_save_dir)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "env": "unifolm_wla_unitree_wbc_msgpack",
            "ckpt_path": self.args.ckpt_path,
            "data_keys": _DATA_KEYS,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": list(self._norm_stats_all.keys()),
            "default_unnorm_key": self._default_unnorm_key,
            "safety": self._safety.config_summary(),
        }

    # ── obs -> unified state ───────────────────────────────────────────────

    def _resolve_unnorm_key(self, obs: dict) -> str:
        req_key = obs.get("unnorm_key")
        if req_key and req_key in self._norm_stats_all:
            return req_key
        if self._default_unnorm_key is not None:
            return self._default_unnorm_key
        raise ValueError(
            f"No unnorm_key in obs and no default set. Available: {list(self._norm_stats_all.keys())}")

    def _build_state_unnorm(self, obs: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Client obs keys -> unified (60,) state_unnorm + (60,) state_mask.

        The EE poses arrive ALREADY as xyz+rot6d (9-dim), so they go straight
        into STATE_SLICES["*_xyz_rot6d"] with no format conversion.
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)
        mask = np.zeros(STATE_DIM, dtype=bool)

        left_ee = _last_frame(obs["observation.state.left_ee_6d"]).astype(np.float32).ravel()
        right_ee = _last_frame(obs["observation.state.right_ee_6d"]).astype(np.float32).ravel()
        # The client is contractually required to send an orthonormal rot6d; a
        # malformed one is otherwise silently Gram-Schmidt-corrected downstream.
        self._safety.check_rot6d(left_ee, "observation.state.left_ee_6d")
        self._safety.check_rot6d(right_ee, "observation.state.right_ee_6d")
        state[STATE_SLICES["left_xyz_rot6d"]] = left_ee[:9]
        mask[STATE_SLICES["left_xyz_rot6d"]] = True
        state[STATE_SLICES["right_xyz_rot6d"]] = right_ee[:9]
        mask[STATE_SLICES["right_xyz_rot6d"]] = True

        left_grip = _last_frame(obs["observation.state.left_gripper"]).astype(np.float32).ravel()
        right_grip = _last_frame(obs["observation.state.right_gripper"]).astype(np.float32).ravel()
        state[STATE_SLICES["left_gripper"]] = left_grip[:1]
        mask[STATE_SLICES["left_gripper"]] = True
        state[STATE_SLICES["right_gripper"]] = right_grip[:1]
        mask[STATE_SLICES["right_gripper"]] = True

        # lower_body = left_leg(6) + right_leg(6) + waist(3)
        lower = _last_frame(obs["observation.state.lower_body"]).astype(np.float32).ravel()
        state[STATE_SLICES["left_leg_joint"]] = lower[0:6]
        mask[STATE_SLICES["left_leg_joint"]] = True
        state[STATE_SLICES["right_leg_joint"]] = lower[6:12]
        mask[STATE_SLICES["right_leg_joint"]] = True
        state[STATE_SLICES["waist_joint"]] = lower[12:15]
        mask[STATE_SLICES["waist_joint"]] = True

        return state, mask

    def _decode_one_image(self, key_obs: Tuple[str, dict]) -> Optional[np.ndarray]:
        """Client obs[key] -> (H,W,3) uint8 RGB, resized to self._image_size. Runs on
        the image thread pool; cv2 releases the GIL for resize/cvtColor."""
        key, obs = key_obs
        if key not in obs:
            return None
        img = _last_frame(obs[key])            # (H,W,3) uint8 BGR (OpenCV)
        if self._image_size is not None:
            target_h, target_w = self._image_size
            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)  # cv2: (W,H)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _build_example(self, obs: dict) -> dict:
        unnorm_key = self._resolve_unnorm_key(obs)

        decoded = list(self._img_pool.map(self._decode_one_image, ((k, obs) for k in _IMAGE_KEYS)))
        # Validate BEFORE filtering: dropping a camera silently would change the
        # prompt (one fewer `role: <image>` block) without raising anything.
        self._safety.check_cameras([k for k, img in zip(_IMAGE_KEYS, decoded) if img is None])
        images = [img for img in decoded if img is not None]

        state_unnorm, state_mask = self._build_state_unnorm(obs)
        norm = self._norm_arrays[unnorm_key]
        state_norm = (state_unnorm - norm["state_offset"]) / (norm["state_scale"] + 1e-8)

        instruction = obs.get("instruction") or obs.get("task") or self.instruction

        example = {
            "image": images,
            "image_roles": WBC_IMAGE_ROLES[: len(images)],
            "lang": instruction,
            "action_mask": self._action_mask,
            "state": state_norm.astype(np.float32),
            "state_mask": state_mask,
            "arm_type": "dual_with_legs",
            "robot_type": "unitree",
        }
        return {"example": example, "unnorm_key": unnorm_key, "state_unnorm": state_unnorm}

    # ── unified action -> client action dict ───────────────────────────────

    def _encode_action(self, pred_norm: np.ndarray, unnorm_key: str, state_unnorm: np.ndarray) -> Dict[str, np.ndarray]:
        norm = self._norm_arrays[unnorm_key]
        unnorm = (pred_norm * norm["action_scale"] + norm["action_offset"]).astype(np.float32)  # (T,54)
        # Physical limits are physical quantities, so they are applied here (post
        # un-normalization) rather than in normalized space.
        unnorm = self._safety.clip_joint_limits(unnorm, mask=self._action_mask)

        left_anchor = state_unnorm[STATE_SLICES["left_xyz_rot6d"]]
        right_anchor = state_unnorm[STATE_SLICES["right_xyz_rot6d"]]
        left_ee_rpy = _compose_ee_to_xyz_rpy(left_anchor, unnorm[:, SLICES["left_xyz_rotvec"]])    # (T,6)
        right_ee_rpy = _compose_ee_to_xyz_rpy(right_anchor, unnorm[:, SLICES["right_xyz_rotvec"]])  # (T,6)

        lower_body = np.concatenate([
            unnorm[:, SLICES["left_leg_joint"]],    # (T,6)
            unnorm[:, SLICES["right_leg_joint"]],   # (T,6)
            unnorm[:, SLICES["waist_joint"]],       # (T,3)
        ], axis=1).astype(np.float32)               # (T,15)

        left_gripper = unnorm[:, SLICES["left_gripper"]].astype(np.float32)    # (T,1)
        right_gripper = unnorm[:, SLICES["right_gripper"]].astype(np.float32)  # (T,1)

        vx_vy = unnorm[:, SLICES["base_vx_vy"]]     # (T,2)
        vw = unnorm[:, SLICES["base_vw"]]           # (T,1)
        height = unnorm[:, SLICES["height"]]        # (T,1)
        base_command = np.concatenate([vx_vy, vw, height], axis=1).astype(np.float32)  # (T,4)

        # pivot = [vx, vy, vw, yaw, pitch, roll, height]. The waist (lower_body
        # last 3) is ordered [yaw, roll, pitch], so remap: yaw=waist[0],
        # pitch=waist[2], roll=waist[1].
        waist = unnorm[:, SLICES["waist_joint"]]    # (T,3) = [yaw, roll, pitch]
        pivot = np.concatenate([
            vx_vy, vw,
            waist[:, 0:1],   # yaw
            waist[:, 2:3],   # pitch
            waist[:, 1:2],   # roll
            height,
        ], axis=1).astype(np.float32)               # (T,7)

        # Each key carries a leading batch axis of 1: (1, T, D). The client
        # indexes action_pred[key][0, :chunk].
        action = {
            "action.left_ee_rpy": left_ee_rpy[None],
            "action.right_ee_rpy": right_ee_rpy[None],
            "action.left_gripper": left_gripper[None],
            "action.right_gripper": right_gripper[None],
            "action.lower_body": lower_body[None],
            "action.base_command": base_command[None],
            "action.pivot": pivot[None],
        }
        # Seam smoothing happens last, in absolute space, because consecutive
        # chunks are composed onto different state anchors -- the discontinuity
        # only exists once the poses are absolute.
        return self._safety.blend(action)

    # ── inference ──────────────────────────────────────────────────────────

    def get_action(self, obs: dict) -> Dict[str, np.ndarray]:
        t0 = time.perf_counter()
        prep = self._build_example(obs)
        t1 = time.perf_counter()
        with torch.no_grad():
            out = self.model.predict_action(examples=[prep["example"]])
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Watchdog: a chunk that arrives after its own execution window has
        # elapsed is worse than no chunk, so degrade to "hold the last commanded
        # pose" instead of handing the client stale motion.
        infer_ms = (t2 - t1) * 1e3
        timeout_ms = float(self._safety.cfg.infer_timeout_ms)
        if self._safety.cfg.enabled and timeout_ms > 0 and infer_ms > timeout_ms:
            hold = self._safety.build_hold()
            logging.error(
                "[safety] predict_action took %.1f ms > %.1f ms budget; returning %s",
                infer_ms, timeout_ms,
                "a HOLD pose" if hold is not None else
                "the real chunk (no previous chunk exists to hold)")
            if hold is not None:
                return hold

        pred_norm = out["normalized_actions"][0]   # (T,54)
        pred_norm = self._safety.clamp_normalized(pred_norm, mask=self._action_mask)
        pred_norm = self._safety.rate_limit(pred_norm, mask=self._action_mask)
        action = self._encode_action(pred_norm, prep["unnorm_key"], prep["state_unnorm"])

        self._debug_saver.save_step(
            images=prep["example"]["image"],
            left_pose=action["action.left_ee_rpy"][0],
            right_pose=action["action.right_ee_rpy"][0],
            prefix_len=0,
            chunk_id=0,
            state_unnorm=prep["state_unnorm"],
            pred_norm=pred_norm,          # post-safety: matches the wire action
            unnorm_key=prep["unnorm_key"],
        )
        t3 = time.perf_counter()
        logging.info("[get_action] T=%d preprocess=%.1fms predict=%.1fms encode=%.1fms",
                     pred_norm.shape[0], (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)
        return action

    # ── websocket serving (mirrors robotdeploy policy_server.py) ───────────

    async def _handler(self, websocket: "_ws_server.ServerConnection"):
        logging.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                mtype = msg["type"]

                if mtype == "get_action":
                    return_data = self.get_action(msg["obs"])
                elif mtype in ("policy_reset", "update_dataset"):
                    # Episode boundary: drop the cross-chunk seam state so the
                    # next chunk does not blend into the previous episode.
                    self._safety.reset()
                    return_data = {}
                elif mtype in ("policy_train", "policy_save", "policy_load"):
                    return_data = None
                elif mtype in ("episode_end", "obs_init"):
                    # `episode_end` is an unambiguous boundary; `obs_init` is left
                    # as a pure no-op because clients may send it per chunk, which
                    # would defeat seam blending.
                    if mtype == "episode_end":
                        self._safety.reset()
                    return_data = None
                elif mtype == "obs_chunk":
                    # streaming obs: no response expected by the client
                    continue
                else:
                    raise ValueError(f"Invalid message type: {mtype}")

                if mtype == "get_action":
                    logging.debug("[safety] stats=%s", self._safety.stats())

                await websocket.send(packer.pack(return_data))

            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _run(self, host: str, port: int):
        async with _ws_server.serve(
            self._handler, host, port,
            compression=None, max_size=None, ping_interval=None, ping_timeout=None,
        ) as server:
            await server.serve_forever()

    def run(self, host: str = "0.0.0.0", port: int = 8600):
        logging.info("WBC msgpack server (unifolm_wla/Unitree) listening on ws://%s:%d", host, port)
        logging.info("metadata=%s", self.metadata)
        asyncio.run(self._run(host, port))


def main():
    parser = argparse.ArgumentParser(
        description="unifolm_wla Unitree G1 msgpack/websocket action server (unitree_rl_wbc client protocol)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt_path", required=True, help="Path to checkpoint .pt/.safetensors")
    parser.add_argument("--instruction", default="",
                        help="Default language instruction if obs carries none (obs['instruction']/'task' override).")
    parser.add_argument("--unnorm_key", default=None,
                        help="Dataset key for norm stats (auto-detected if single dataset)")
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="Camera resize target (H W). Default: auto-detect from the checkpoint's "
                             "training data config (datasets.vla_data.data_config_path -> image_size), "
                             f"falling back to {_DEFAULT_IMAGE_SIZE[0]} {_DEFAULT_IMAGE_SIZE[1]}. "
                             "Pass '0 0' to disable resizing.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--debug_save_dir", default=None,
                        help="If set, save a per-step camera+action debug image to this "
                             "directory on a background thread.")
    parser.add_argument("--base_vlm", default=None, metavar="PATH",
                        help="Override framework.qwenvl.base_vlm. Needed when the checkpoint "
                             "recorded a training-cluster path that does not exist on this machine.")

    safety = parser.add_argument_group("action safety")
    safety.add_argument("--safety", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable the safety layer (clamp / rate limit / joint limits / "
                             "seam blend / watchdog / input validation).")
    safety.add_argument("--action_clip_norm", type=float, default=5.0, metavar="FLOAT",
                        help="Clamp |normalized action| to this value (0 disables). Normalized "
                             "units: 1.0 is ~q99 for minmax_q dims but 1 sigma for zscore dims, "
                             "so the default only catches pathology.")
    safety.add_argument("--action_max_step", type=float, default=10.0, metavar="FLOAT",
                        help="Max per-step change in normalized action space (0 disables).")
    safety.add_argument("--action_blend_steps", type=int, default=0, metavar="N",
                        help="Blend the first N steps of each new chunk out of the previous "
                             "chunk's last commanded step (0 disables). Enable this if your "
                             "client jumps straight to each new chunk's step 0.")
    safety.add_argument("--joint_limits", default=None, metavar="PATH",
                        help="YAML/JSON file of per-slot [lo, hi] physical bounds. See "
                             "model_server/joint_limits.example.yaml. Without it, joint ranges "
                             "are NOT clipped.")
    safety.add_argument("--infer_timeout_ms", type=float, default=2000.0, metavar="MS",
                        help="If one predict_action call exceeds this, return a hold pose "
                             "instead of a stale chunk (0 disables).")
    safety.add_argument("--require_all_cameras", action=argparse.BooleanOptionalAction, default=True,
                        help="Raise when a camera observation is missing (default) instead of "
                             "silently serving a prompt with fewer images than training.")
    safety.add_argument("--strict_rot6d", action=argparse.BooleanOptionalAction, default=False,
                        help="Raise instead of warning when a client EE rot6d is not orthonormal.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    server = ActionServerWBCMsgpack(args)
    server.run(args.host, args.port)


if __name__ == "__main__":
    main()
