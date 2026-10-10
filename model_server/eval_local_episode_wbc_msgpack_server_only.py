"""Full-episode msgpack-server-vs-GT accuracy check on a local Unitree episode.

Protocol, per the real client:
  * connect over a websocket, then immediately receive one msgpack-packed
    ``metadata`` frame (declaring ``data_keys`` -- the obs keys to send and the
    action keys returned);
  * send ``{"type": "get_action", "obs": {...client obs keys...}}`` and receive
    a dict of action keys back (whole chunk, unnormalized, absolute).

The obs keys the server expects are the CLIENT format, NOT the raw dataset
format:
    observation.images.cam_left_high / cam_left_wrist / cam_right_wrist  (H,W,3) uint8
    observation.state.left_ee_6d / right_ee_6d   (9,) xyz + rot6d
    observation.state.left_gripper / right_gripper  (1,)
    observation.state.lower_body  (15,) left_leg(6)+right_leg(6)+waist(3)
We reconstruct these directly from the sample's unified ``state_unnorm`` (which
already holds absolute EE as xyz+rot6d in STATE_SLICES) and the raw images --
this is exactly the state the server rebuilds internally, so the round-trip is
apples-to-apples with the dataset's GT.

The server returns action.left_ee_rpy / right_ee_rpy as (T,6) xyz+rpy ABSOLUTE
(composed onto the current EE anchor), so only EE xyz is directly comparable to
the dataset's absolutized output (translation is representation-agnostic): left EE
xyz = action.left_ee_rpy[:, 0:3], right EE xyz = action.right_ee_rpy[:, 0:3].

Usage:
    # First, in another terminal:
    python -m model_server.action_server_wbc_msgpack_unitree \\
        --ckpt_path playground/Checkpoints/<run_id>/checkpoints/steps_<N>_model.safetensors \\
        --host 127.0.0.1 --port 8600

    # Then:
    python -m model_server.eval_local_episode_wbc_msgpack_server_only \\
        --ckpt_path playground/Checkpoints/<run_id>/checkpoints/steps_<N>_model.safetensors \\
        --host 127.0.0.1 --port 8600 --episode_idx 0 \\
        --save_dir results/eval_local_episode_wbc_msgpack
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_WORKSPACE_ROOT = str(Path(__file__).resolve().parents[1])
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

# msgpack (numpy-aware) -- byte-compatible with the server's serializer.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools import msgpack_numpy  # noqa: E402

from unifolm_wla.dataloader.multi_source_dataset.config import load_config  # noqa: E402
from unifolm_wla.dataloader.multi_source_dataset.single_source_dataset import create_single_source_dataset  # noqa: E402
from unifolm_wla.dataloader.multi_source_dataset.action_mapping import STATE_SLICES  # noqa: E402

_UNITREE_EVAL_DIR = str(
    Path(_WORKSPACE_ROOT) / "examples" / "unifolm_wla" / "eval_files" / "unitree"
)
if _UNITREE_EVAL_DIR not in sys.path:
    sys.path.insert(0, _UNITREE_EVAL_DIR)

from eval_local_episode import (  # noqa: E402
    ABS_DIM_COLOURS,
    ABS_DIM_LABELS,
    REL_DIM_COLOURS,
    REL_DIM_LABELS,
    _episode_frame_range,
    _plot_episode_grid,
    _to_numpy,
    ensure_dataset_statistics,
    extract_rel_ee,
    relative_to_absolute_ee,
    unnormalize,
)


def parse_args():
    p = argparse.ArgumentParser(description="Episode-level wbc-msgpack-server-vs-GT accuracy check.")
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="unifolm_wla training-run checkpoint -- used to locate dataset_statistics.json. The "
                        "already-running msgpack server was pointed at its own (matching) checkpoint separately.")
    p.add_argument("--host", type=str, default="127.0.0.1",
                   help="Host of the already-running action_server_wbc_msgpack_unitree.py server.")
    p.add_argument("--port", type=int, default=8600)
    p.add_argument("--unnorm_key", type=str, default=None,
                   help="unnorm_key sent to the server in each obs (ignored by single-dataset ckpts).")
    p.add_argument("--data_config_path", type=str,
                   default="unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml")
    p.add_argument("--source_idx", type=int, default=0)
    p.add_argument("--task_idx", type=int, default=0)
    p.add_argument("--episode_idx", type=int, default=0)
    p.add_argument("--source_name", type=str, default=None)
    p.add_argument("--stride", type=int, default=None,
                   help="Frames consumed per chunk before the next obs is sent. Defaults to the "
                        "source's chunk_size (non-overlapping full-chunk stitching).")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--save_dir", type=str, default="results/eval_local_episode_wbc_msgpack")
    return p.parse_args()


def _chw_to_hwc_uint8(img_t) -> np.ndarray:
    """sample['images'][role] is a CHW float/uint8 tensor. Return (H,W,3) uint8
    RGB -- the raw-array image format the msgpack server expects."""
    arr = _to_numpy(img_t)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.max() <= 1.0 + 1e-3:
        arr = arr * 255.0
    return np.ascontiguousarray(arr.clip(0, 255).astype(np.uint8))


def build_client_obs(sample: dict, state_unnorm: np.ndarray, instruction: str, unnorm_key: str) -> dict:
    """Reconstruct the CLIENT-format obs the msgpack server expects, from the
    sample's raw images + unified state_unnorm.

    EE poses come straight out of STATE_SLICES["*_xyz_rot6d"] (9-dim xyz+rot6d),
    which is exactly the 9-dim the client's ``left_ee_6d`` carries; grippers and
    the merged lower_body (left_leg 6 + right_leg 6 + waist 3) likewise mirror
    the client's payload."""
    imgs = sample["images"]
    obs = {
        "observation.images.cam_left_high": _chw_to_hwc_uint8(imgs["head_left"]),
        "observation.images.cam_left_wrist": _chw_to_hwc_uint8(imgs["cam_wrist_left"]),
        "observation.images.cam_right_wrist": _chw_to_hwc_uint8(imgs["cam_wrist_right"]),
        "observation.state.left_ee_6d": state_unnorm[STATE_SLICES["left_xyz_rot6d"]].astype(np.float32),
        "observation.state.right_ee_6d": state_unnorm[STATE_SLICES["right_xyz_rot6d"]].astype(np.float32),
        "observation.state.left_gripper": state_unnorm[STATE_SLICES["left_gripper"]].astype(np.float32),
        "observation.state.right_gripper": state_unnorm[STATE_SLICES["right_gripper"]].astype(np.float32),
        "observation.state.lower_body": np.concatenate([
            state_unnorm[STATE_SLICES["left_leg_joint"]],
            state_unnorm[STATE_SLICES["right_leg_joint"]],
            state_unnorm[STATE_SLICES["waist_joint"]],
        ]).astype(np.float32),
        "instruction": instruction,
    }
    if unnorm_key:
        obs["unnorm_key"] = unnorm_key
    return obs


async def _run_msgpack_client(host: str, port: int, ordered_obs: list) -> list:
    """Speak the unitree_rl_wbc policy_client protocol: connect, read the one
    metadata frame, then one get_action round-trip per chunk-triggering obs.

    `ordered_obs` is [(chunk_local_start, client_obs_dict), ...] in episode
    order. Returns [(chunk_local_start, action_dict), ...] in the same order,
    where action_dict has the client action keys."""
    import websockets

    uri = f"ws://{host}:{port}"
    packer = msgpack_numpy.Packer()
    results = []
    async with websockets.connect(uri, max_size=None) as ws:
        metadata = msgpack_numpy.unpackb(await ws.recv())
        print("Server metadata:", flush=True)
        for k in ("env", "data_keys", "action_chunk_size", "available_unnorm_keys", "default_unnorm_key"):
            if k in metadata:
                print(f"    {k}: {metadata[k]}", flush=True)
        for chunk_local_start, obs in ordered_obs:
            await ws.send(packer.pack({"type": "get_action", "obs": obs}))
            raw = await ws.recv()
            if isinstance(raw, str):
                # On an internal error the server sends a plain-text traceback
                # (not msgpack-packed) and then closes the connection.
                raise RuntimeError(f"Server raised an exception:\n{raw}")
            reply = msgpack_numpy.unpackb(raw)
            results.append((chunk_local_start, reply))
    return results


def report_episode_l1(name: str, gt: np.ndarray, pred: np.ndarray, dim_labels: list) -> None:
    l1_per_dim = np.mean(np.abs(pred - gt), axis=0)
    print(f"\nPer-dim mean L1 ({name}):")
    for d, (lbl, val) in enumerate(zip(dim_labels, l1_per_dim)):
        print(f"    dim {d:2d}  {lbl:8s}  {val:.4f}")
    print(f"  Overall {name} L1: {l1_per_dim.mean():.4f}")


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt_path)

    # 0. Self-heal dataset_statistics.json next to the checkpoint.
    ensure_dataset_statistics(ckpt_path, args.data_config_path)

    # 1. Dataset.
    print(f"Building dataset from {args.data_config_path} ...")
    config = load_config(args.data_config_path)
    enabled = [d for d in config.datasets if d.enabled]
    if args.source_idx < 0 or args.source_idx >= len(enabled):
        raise IndexError(
            f"source_idx={args.source_idx} out of range; "
            f"{len(enabled)} enabled dataset source(s) in {args.data_config_path}."
        )
    ds_cfg = enabled[args.source_idx]
    src = create_single_source_dataset(ds_cfg, config)
    src_name = ds_cfg.name
    chunk_size = src.chunk_size
    print(f"Using source[{args.source_idx}]='{src_name}', chunk_size={chunk_size}")

    # 2. Episode bounds.
    task_offset, from_idx, to_idx = _episode_frame_range(src, args.task_idx, args.episode_idx)
    ep_len_raw = to_idx - from_idx
    ep_len = min(ep_len_raw, args.max_frames) if args.max_frames is not None else ep_len_raw
    print(f"Episode {args.episode_idx} in task {args.task_idx}: "
          f"frames [{from_idx}, {from_idx + ep_len}) (raw episode length {ep_len_raw})")

    stride = args.stride if args.stride is not None else chunk_size
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    # 3. Norm stats.
    stats_path = ckpt_path.parents[1] / "dataset_statistics.json"
    print(f"Loading norm stats from {stats_path} ...")
    with open(stats_path, "r") as f:
        all_stats = json.load(f)

    source_name = args.source_name or src_name
    if source_name not in all_stats:
        source_name = next(iter(all_stats.keys()))
        print(f"Source '{src_name}' missing in norm_stats; falling back to '{source_name}'")
    action_stats = all_stats[source_name]["action"]
    offset = np.array(action_stats["offset"], dtype=np.float32)
    scale = np.array(action_stats["scale"], dtype=np.float32)
    print(f"Using norm stats for '{source_name}'")

    # 4. Chunk-by-chunk GT decode, and build each chunk-triggering frame's
    # CLIENT-format obs for the server round-trip below.
    gt_abs_all = np.zeros((ep_len, 14), dtype=np.float32)
    gt_rel_all = np.zeros((ep_len, 14), dtype=np.float32)
    fill_mask = np.zeros(ep_len, dtype=bool)
    chunk_starts = []
    ordered_obs = []    # [(chunk_local_start, client_obs_dict), ...] in episode order
    task_label = None

    chunk_frame_starts = list(range(0, ep_len, stride))
    print(f"Processing {len(chunk_frame_starts)} chunks for GT decode and "
          f"building each chunk-triggering frame's client obs...")
    for chunk_local_start in tqdm(chunk_frame_starts, desc="Chunks (GT + obs)"):
        frame_in_sub = from_idx + chunk_local_start
        src_local_idx = task_offset + frame_in_sub
        sample = src[src_local_idx]
        if task_label is None:
            task_label = sample.get("task", "")

        gt_action_norm = _to_numpy(sample["action"])       # (T, 54)
        state_unnorm = _to_numpy(sample["state_unnorm"])    # (60,)

        T = len(gt_action_norm)
        write_len = min(stride, ep_len - chunk_local_start, T)
        if write_len <= 0:
            continue

        gt_n = gt_action_norm[:T]
        gt_unnorm = unnormalize(gt_n, offset, scale)
        gt_rel = extract_rel_ee(gt_unnorm)
        gt_abs = relative_to_absolute_ee(state_unnorm, gt_unnorm)

        sl = slice(chunk_local_start, chunk_local_start + write_len)
        gt_rel_all[sl] = gt_rel[:write_len]
        gt_abs_all[sl] = gt_abs[:write_len]
        fill_mask[sl] = True
        chunk_starts.append(chunk_local_start)

        ordered_obs.append((
            chunk_local_start,
            build_client_obs(sample, state_unnorm, sample["task"], args.unnorm_key),
        ))

    # 5. Server: drive the already-running msgpack server over its websocket
    # protocol (one get_action round-trip per chunk, whole chunk per reply).
    print(f"\nQuerying msgpack server at {args.host}:{args.port} for {len(ordered_obs)} chunks...")
    server_chunks = asyncio.run(_run_msgpack_client(args.host, args.port, ordered_obs))
    print(f"Received {len(server_chunks)} chunks.")

    # 6. Stitch each returned chunk into the episode timeline and extract the
    # 6-dim [L_xyz(3), R_xyz(3)] EE-xyz subset from the rpy action keys
    # (left EE xyz = action.left_ee_rpy[:, 0:3], right = action.right_ee_rpy[:, 0:3]).
    srv_ee_xyz_all = np.zeros((ep_len, 6), dtype=np.float32)
    srv_fill_mask = np.zeros(ep_len, dtype=bool)
    for chunk_local_start, action in server_chunks:
        left = np.asarray(action["action.left_ee_rpy"])[0]    # (1,T,6) -> (T,6)
        right = np.asarray(action["action.right_ee_rpy"])[0]  # (1,T,6) -> (T,6)
        T = min(left.shape[0], right.shape[0])
        write_len = min(stride, ep_len - chunk_local_start, T)
        if write_len <= 0:
            continue
        ee_xyz = np.concatenate([left[:write_len, 0:3], right[:write_len, 0:3]], axis=1)  # (write_len, 6)
        sl = slice(chunk_local_start, chunk_local_start + write_len)
        srv_ee_xyz_all[sl] = ee_xyz
        srv_fill_mask[sl] = True

    gt_ee_xyz_all = np.concatenate([gt_abs_all[:, 0:3], gt_abs_all[:, 7:10]], axis=1)

    fill_mask = fill_mask & srv_fill_mask
    if not fill_mask.any():
        raise RuntimeError("No frames were evaluated by both GT and the server; check episode "
                            "bounds/stride and that the msgpack server is reachable.")
    if not fill_mask.all():
        last_filled = int(np.where(fill_mask)[0].max()) + 1
        gt_abs_all = gt_abs_all[:last_filled]
        gt_rel_all = gt_rel_all[:last_filled]
        gt_ee_xyz_all, srv_ee_xyz_all = (
            a[:last_filled] for a in (gt_ee_xyz_all, srv_ee_xyz_all)
        )
        ep_len = last_filled
        chunk_starts = [s for s in chunk_starts if s < last_filled]

    # 7. Plots.
    task_str = (task_label or "").replace("\n", " ")[:80]
    title_suffix = (
        f"source='{src_name}', task={args.task_idx}, ep={args.episode_idx}, "
        f"len={ep_len}, stride={stride}, chunk={chunk_size} | {task_str}"
    )
    EE_XYZ_LABELS = ["L_x", "L_y", "L_z", "R_x", "R_y", "R_z"]
    EE_XYZ_COLOURS = ["steelblue"] * 3 + ["darkorange"] * 3

    ee_xyz_png = save_dir / f"episode{args.episode_idx:04d}_ee_xyz_2way.png"
    print(f"Saving GT/server EE-xyz episode plot -> {ee_xyz_png}")
    _plot_episode_grid(
        gt_ee_xyz_all, srv_ee_xyz_all, chunk_starts, EE_XYZ_LABELS, EE_XYZ_COLOURS,
        f"[ABS EE xyz] {title_suffix}", ee_xyz_png,
    )

    # 8. Summary.
    print("\n" + "=" * 60)
    print(f"Episode {args.episode_idx} | {ep_len} frames | {len(chunk_starts)} chunks")
    print("=" * 60)
    report_episode_l1("abs EE xyz (server vs GT)", gt_ee_xyz_all, srv_ee_xyz_all, EE_XYZ_LABELS)
    print(f"\nPlots saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
