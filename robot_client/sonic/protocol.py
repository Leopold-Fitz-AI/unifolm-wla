"""SONIC's single-part ZMQ protocol (1280-byte header, NOT old 1024).

Contract checked against NVlabs/GR00T-WholeBodyControl at 1fa5f7f,
gear_sonic/utils/teleop/zmq/zmq_planner_sender.py and the C++ endpoint.
"""

import json

import numpy as np

HEADER_SIZE = 1280
# q from g1_debug is absolute MuJoCo/SDK order. The v1 INPUT is IsaacLab order.
# This permutation is mujoco_to_isaaclab in SONIC's policy_parameters.hpp.
MUJOCO_TO_ISAAC = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])
ISAAC_TO_MUJOCO = np.argsort(MUJOCO_TO_ISAAC)
UPPER_MUJOCO_IN_ISAAC = np.array([12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])


def pack_fields(topic, data):
    """Same field ordering, endian and count as SONIC's native builders."""
    codes = {"float32": "f32", "int32": "i32", "uint8": "u8"}
    fields, payload = [], []
    for name, value in data.items():
        a = np.asarray(value)
        if a.dtype.name not in codes or not np.isfinite(a).all():
            raise ValueError(f"unsupported/nonfinite planner field {name}")
        a = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<"))
        fields.append({"name": name, "dtype": codes[a.dtype.name], "shape": list(a.shape)})
        payload.append(a.tobytes())
    header = json.dumps({"v": 1, "endian": "le", "count": 1, "fields": fields}, separators=(",", ":")).encode()
    if len(header) >= HEADER_SIZE:
        raise ValueError("SONIC header overflow")
    return topic.encode() + header.ljust(HEADER_SIZE, b"\0") + b"".join(payload)


def pack_command(start=False, stop=False, planner=True):
    return pack_fields("command", {"start": np.array([start], np.uint8), "stop": np.array([stop], np.uint8),
                                   "planner": np.array([planner], np.uint8)})


def finite_array(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite {shape}, got {a.shape}")
    return a


def pack_reference(q, dq, quat_wxyz, frame_index):
    """q/dq are absolute radians in MuJoCo order; velocity is radians/second."""
    q = np.asarray(q)
    if q.ndim != 2 or q.shape[1] != 29 or len(q) < 2:
        raise ValueError("reference must contain at least 2 frames of 29 joints")
    n = len(q)
    q = finite_array(q, (n, 29), "q")
    dq = finite_array(dq, (n, 29), "dq")
    quat = finite_array(quat_wxyz, (n, 4), "body_quat")
    if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-4):
        raise ValueError("body_quat must be unit wxyz quaternions")
    indices = np.asarray(frame_index)
    if indices.shape != (n,) or indices.dtype.kind not in "iu" or not np.all(np.diff(indices) == 1):
        raise ValueError("frame_index must be consecutive integer frame numbers")
    data = {
        "joint_pos": np.asarray(q[:, MUJOCO_TO_ISAAC], dtype="<f4"),
        "joint_vel": np.asarray(dq[:, MUJOCO_TO_ISAAC], dtype="<f4"),
        "body_quat": np.asarray(quat, dtype="<f4"),
        "frame_index": np.asarray(indices, dtype="<i8"),
    }
    fields = [{"name": k, "dtype": "i64" if k == "frame_index" else "f32", "shape": list(v.shape)}
              for k, v in data.items()]
    header = json.dumps({"v": 1, "endian": "le", "count": 1, "fields": fields},
                        separators=(",", ":")).encode()
    if len(header) >= HEADER_SIZE:
        raise ValueError("SONIC header overflow")
    return b"pose" + header.ljust(HEADER_SIZE, b"\0") + b"".join(a.tobytes() for a in data.values())
