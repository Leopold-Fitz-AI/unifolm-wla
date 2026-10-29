"""Bounded sensor reads and the existing UnifoLM msgpack protocol."""

import base64

import cv2
import msgpack
import numpy as np
import zmq
from websockets.asyncio.client import connect

from model_server.tools import msgpack_numpy


class LatestSubscriber:
    def __init__(self, context, endpoint, topic=b""):
        self.socket = context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, topic)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)
        self.topic = topic
        self.message = None
        self.received_at = float("-inf")
        self.sequence = None

    def read(self, now, max_age):
        try:
            raw = self.socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            pass
        else:
            if not raw.startswith(self.topic):
                raise ValueError("unexpected ZMQ topic")
            message = msgpack.unpackb(raw[len(self.topic):], raw=False)
            # Re-publishing an old control tick must not refresh the watchdog.
            sequence = message.get("index", message.get("timestamps"))
            if sequence is None:
                raise ValueError("sensor packet has no index/timestamps for freshness checks")
            if self.sequence is None or sequence != self.sequence:
                self.message, self.received_at, self.sequence = message, now, sequence
        if now - self.received_at > max_age:
            return None
        return self.message

    def close(self):
        self.socket.close()


def decode_cameras(message, mapping):
    """Accept current SONIC nested images and its simulator's flat schema.

    Both wire formats encode JPEG as bytes or base64. cv2 returns BGR, exactly
    what the UnifoLM server requires. Missing views are errors, not duplicates.
    """
    source = message.get("images", message)
    images = {}
    for destination, origin in mapping.items():
        encoded = source[origin]
        if isinstance(encoded, str):
            encoded = base64.b64decode(encoded, validate=True)
        if not isinstance(encoded, bytes):
            raise ValueError(f"{origin}: camera payload must be JPEG bytes/base64")
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"{origin}: JPEG decode failed")
        images[destination] = image
    return images


class PolicyClient:
    def __init__(self, url, timeout=5):
        self.url, self.timeout = url, timeout
        self.socket = None
        self.metadata = None

    async def __aenter__(self):
        import asyncio
        self.socket = await connect(self.url, compression=None, max_size=64 * 1024 * 1024,
                                    open_timeout=self.timeout)
        try:
            self.metadata = await asyncio.wait_for(self.receive(), self.timeout)
            if self.metadata.get("env") != "unifolm_wla_unitree_wbc_msgpack":
                raise ValueError("not an UnifoLM WBC server")
            await self.request("policy_reset")
        except BaseException:
            await self.socket.close()
            raise
        return self

    async def receive(self):
        raw = await self.socket.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"model server failed: {raw[:2000]}")
        return msgpack_numpy.unpackb(raw)

    async def request(self, kind, **fields):
        import asyncio
        await self.socket.send(msgpack_numpy.packb({"type": kind, **fields}))
        return await asyncio.wait_for(self.receive(), self.timeout)

    async def __aexit__(self, *exc):
        await self.socket.close()


def attach_sim_hands(packet, grips, open_q, closed_q):
    """Explicit Dex3 surrogate for simulation; this is NOT a Dex1 driver."""
    import json
    from .protocol import HEADER_SIZE
    header = json.loads(packet[4:4 + HEADER_SIZE].rstrip(b"\0"))
    payload = packet[4 + HEADER_SIZE:]
    for i, side in enumerate(("left", "right")):
        q = np.asarray(open_q[i]) + float(grips[i]) * (np.asarray(closed_q[i]) - np.asarray(open_q[i]))
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("hand endpoints must be finite (2,7)")
        header["fields"].append({"name": f"{side}_hand_joints", "dtype": "f32", "shape": [7]})
        payload += np.asarray(q, dtype="<f4").tobytes()
    packed_header = json.dumps(header, separators=(",", ":")).encode()
    if len(packed_header) >= HEADER_SIZE:
        raise ValueError("SONIC header overflow")
    return b"pose" + packed_header.ljust(HEADER_SIZE, b"\0") + payload


def measured_grippers(state, open_q, closed_q):
    result = []
    for i, side in enumerate(("left", "right")):
        actual = np.asarray(state[f"{side}_hand_q"], dtype=float)
        delta = np.asarray(closed_q[i]) - np.asarray(open_q[i])
        if actual.shape != (7,) or not np.isfinite(actual).all() or np.dot(delta, delta) <= 0:
            raise ValueError("invalid hand feedback or surrogate endpoints")
        result.append(np.clip(np.dot(actual - open_q[i], delta) / np.dot(delta, delta), 0, 1))
    return np.array(result)


def model_gripper_values(closed_fraction, open_value, closed_value):
    """Raw Dex1 motor position increases towards OPEN, unlike a close ratio."""
    return np.asarray(open_value) + np.asarray(closed_fraction) * (np.asarray(closed_value) - open_value)


def gripper_closed_fraction(values, open_value, closed_value):
    return np.clip((np.asarray(values) - open_value) / (np.asarray(closed_value) - open_value), 0, 1)
