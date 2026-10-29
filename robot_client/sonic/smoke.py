"""Loopback integration fixture: real WebSocket/ZMQ and real G1 MuJoCo IK.

Uses a deterministic MODEL FIXTURE and a kinematic SONIC receiver fixture.
It does NOT run the SONIC neural controller, physics, or UnifoLM weights.
"""

import argparse
import asyncio
import json
from pathlib import Path
import time

import cv2
import msgpack
import numpy as np
from websockets.asyncio.server import serve
import zmq

from model_server.tools import msgpack_numpy
from .adapter import pose_to_rpy
from .kinematics import G1Kinematics
from .protocol import ISAAC_TO_MUJOCO, HEADER_SIZE
from .run import load_config, run


def decode_packet(raw):
    """Independent receiver for assertions, not used by the production sender."""
    if raw[:4] != b"pose":
        raise ValueError("unexpected topic")
    header = json.loads(raw[4:4 + HEADER_SIZE].rstrip(b"\0"))
    if header["v"] != 1 or header["endian"] != "le":
        raise ValueError("unexpected protocol")
    offset = 4 + HEADER_SIZE
    arrays = {}
    for field in header["fields"]:
        dtype = np.dtype({"f32": "<f4", "i64": "<i8"}[field["dtype"]])
        count = int(np.prod(field["shape"]))
        arrays[field["name"]] = np.frombuffer(raw, dtype, count=count, offset=offset).reshape(field["shape"])
        offset += count * dtype.itemsize
    if offset != len(raw):
        raise ValueError("trailing payload or truncated message")
    return arrays


def fixture_action(kin, q0, frames=30):
    rows = []
    for i in range(frames):
        q = q0.copy()
        q[15] += 0.015 * np.sin((i + 1) / frames * np.pi)
        q[22] += 0.015 * np.sin((i + 1) / frames * np.pi)
        rows.append([pose_to_rpy(t) for t in kin.forward(q)])
    poses = np.asarray(rows)
    return {
        "action.left_ee_rpy": poses[None, :, 0],
        "action.right_ee_rpy": poses[None, :, 1],
        "action.lower_body": np.repeat(q0[None, None, :15], frames, axis=1),
        "action.left_gripper": np.zeros((1, frames, 1)),
        "action.right_gripper": np.zeros((1, frames, 1)),
        "action.base_command": np.tile([0, 0, 0, 0.75], (1, frames, 1)),
    }


async def smoke(sonic_root, seconds=3, config_overrides=None):
    config = load_config(Path(__file__).with_name("sim.yaml"))
    config["sonic_mode"] = "joint"
    kin = G1Kinematics(sonic_root)
    q0 = np.zeros(29)
    q0[[0, 4, 6, 10]] = -0.1
    q0[[3, 9]] = 0.2
    q0[[18, 25]] = 0.5
    action = fixture_action(kin, q0)
    stats = {"model_requests": 0, "received_packets": 0, "max_joint_excursion_rad": 0.0,
             "validation": "protocol_fixture_and_G1_MuJoCo_kinematics",
             "sonic_neural_controller": False, "unifolm_weights": False}
    context = zmq.Context()
    state_pub = context.socket(zmq.PUB)
    camera_pub = context.socket(zmq.PUB)
    receiver = context.socket(zmq.SUB)
    for socket in (state_pub, camera_pub, receiver):
        socket.setsockopt(zmq.LINGER, 0)
    ports = [socket.bind_to_random_port("tcp://127.0.0.1") for socket in (state_pub, camera_pub)]
    # XPUB-style readiness is unnecessary here: the fixture publishes continuously.
    receiver.setsockopt(zmq.SUBSCRIBE, b"pose")
    port_socket = context.socket(zmq.PUB)
    action_port = port_socket.bind_to_random_port("tcp://127.0.0.1")
    port_socket.close(0)
    receiver.connect(f"tcp://127.0.0.1:{action_port}")
    config["state_endpoint"] = f"tcp://127.0.0.1:{ports[0]}"
    config["camera_endpoint"] = f"tcp://127.0.0.1:{ports[1]}"
    config["action_bind"] = f"tcp://127.0.0.1:{action_port}"
    config["inference_interval"] = 0.35
    config.update(config_overrides or {})
    q = q0.copy()
    running = True

    async def model(ws):
        await ws.send(msgpack_numpy.packb({"env": "unifolm_wla_unitree_wbc_msgpack", "data_keys": []}))
        async for raw in ws:
            request = msgpack_numpy.unpackb(raw)
            if request["type"] == "policy_reset":
                reply = {}
            else:
                obs = request["obs"]
                assert obs["observation.state.lower_body"].shape == (15,)
                for side in ("left", "right"):
                    pose = obs[f"observation.state.{side}_ee_6d"]
                    r = pose[3:].reshape(2, 3).T
                    np.testing.assert_allclose(r.T @ r, np.eye(2), atol=1e-5)
                for camera in ("cam_left_high", "cam_left_wrist", "cam_right_wrist"):
                    assert obs[f"observation.images.{camera}"].shape == (48, 64, 3)
                stats["model_requests"] += 1
                reply = action
            await ws.send(msgpack_numpy.packb(reply))

    async def sensors():
        nonlocal q
        i = 0
        ok, jpg = cv2.imencode(".jpg", np.full((48, 64, 3), [20, 80, 140], dtype=np.uint8))
        assert ok
        while running:
            try:
                while True:
                    arrays = decode_packet(receiver.recv(zmq.NOBLOCK))
                    q = arrays["joint_pos"][0, ISAAC_TO_MUJOCO].astype(float)
                    kin.check_q(q)
                    assert np.isfinite(arrays["joint_vel"]).all()
                    assert np.all(np.diff(arrays["frame_index"]) == 1)
                    stats["received_packets"] += 1
                    stats["max_joint_excursion_rad"] = max(stats["max_joint_excursion_rad"], float(np.max(np.abs(q - q0))))
            except zmq.Again:
                pass
            state = {"index": i, "body_q": q.tolist(), "base_quat": [1, 0, 0, 0],
                     "left_hand_q": [0] * 7, "right_hand_q": [0] * 7}
            state_pub.send(b"g1_debug" + msgpack.packb(state, use_bin_type=True))
            cameras = {"images": {k: jpg.tobytes() for k in ("ego_view", "left_wrist", "right_wrist")},
                       "timestamps": {"ego_view": time.monotonic()}}
            camera_pub.send(msgpack.packb(cameras, use_bin_type=True))
            i += 1
            await asyncio.sleep(0.02)

    sensor_task = asyncio.create_task(sensors())
    try:
        async with serve(model, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            config["policy_url"] = f"ws://127.0.0.1:{port}"
            stats.update(await run(config, sonic_root, seconds))
        if stats["model_requests"] < 2 or stats["received_packets"] < 10 or stats["max_joint_excursion_rad"] < 1e-4:
            raise AssertionError(f"incomplete loopback: {stats}")
        return stats
    finally:
        running = False
        await sensor_task
        for socket in (state_pub, camera_pub, receiver):
            socket.close()
        context.term()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic-root", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = asyncio.run(smoke(args.sonic_root, args.seconds))
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
