"""RGB-D sensor transport and calibrated, image-only tabletop localization.

Packets contain rendered sensor pixels and robot camera calibration. They never
contain object identities, poses, contacts, or task-evaluator ground truth.
"""
import cv2
import msgpack
import mujoco
import numpy as np
import zmq


class RGBDPublisher:
    def __init__(self, env, port, camera="unifolm_head", shape=(336, 448)):
        self.env, self.camera, self.shape = env, camera, shape
        self.renderer = mujoco.Renderer(env.mj_model, height=shape[0], width=shape[1])
        self.renderer.enable_depth_rendering()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.bind(f"tcp://127.0.0.1:{port}")
        self.last_time = -1
        self.index = 0

    def publish(self, rgb):
        model, data = self.env.mj_model, self.env.mj_data
        if data.time >= self.last_time and data.time-self.last_time < .1:
            return
        root = model.body("pelvis").id
        rotation = data.xmat[root].reshape(3, 3).copy()
        if not np.isclose(np.linalg.det(rotation), 1):
            return
        self.last_time = float(data.time)
        self.renderer.update_scene(data, camera=self.camera)
        depth = self.renderer.render().astype("<f4")
        camera = model.camera(self.camera).id
        transform = np.eye(4)
        transform[:3, :3] = rotation.T @ data.cam_xmat[camera].reshape(3, 3)
        transform[:3, 3] = rotation.T @ (data.cam_xpos[camera] - data.xpos[root])
        h, w = self.shape
        focal = .5*h/np.tan(np.deg2rad(model.cam_fovy[camera]/2))
        packet = {"index": self.index, "simulation_time": float(data.time), "shape": [h, w],
                  "bgr": np.ascontiguousarray(rgb[:, :, ::-1]).tobytes(), "depth_f32": depth.tobytes(),
                  "intrinsics": [focal, focal, (w-1)/2, (h-1)/2],
                  "camera_to_pelvis": transform.tolist(), "up_pelvis": (rotation.T @ [0, 0, 1]).tolist()}
        self.index += 1
        self.socket.send(b"rgbd" + msgpack.packb(packet, use_bin_type=True), zmq.NOBLOCK)

    def close(self):
        self.renderer.close()
        self.socket.close()
        self.context.term()


def decode_rgbd(packet):
    h, w = packet["shape"]
    if not (1 <= h <= 2048 and 1 <= w <= 2048):
        raise ValueError("invalid RGB-D shape")
    bgr = np.frombuffer(packet["bgr"], np.uint8).reshape(h, w, 3)
    depth = np.frombuffer(packet["depth_f32"], "<f4").reshape(h, w)
    transform = np.asarray(packet["camera_to_pelvis"], float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("invalid camera calibration")
    return bgr, depth, transform


def locate_cube(packet, grounding, cube_height=.05):
    """Refine a VLM-selected bounding box using color and depth, without object state."""
    bgr, depth, transform = decode_rgbd(packet)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, value = cv2.split(hsv)
    color = grounding["color"]
    hue_masks = {"red": (hue < 12) | (hue > 168), "blue": (hue > 95) & (hue < 135),
                 "green": (hue > 35) & (hue < 90), "yellow": (hue >= 20) & (hue <= 35),
                 "orange": (hue >= 12) & (hue < 20)}
    if color not in hue_masks:
        raise ValueError("unsupported grounded color")
    h, w = depth.shape
    box = np.asarray(grounding["bbox"]) * [w/1000, h/1000, w/1000, h/1000]
    pad_x, pad_y = max(4, .3*(box[2]-box[0])), max(4, .3*(box[3]-box[1]))
    x0, y0 = np.maximum(np.floor(box[:2]-[pad_x, pad_y]), [0, 0]).astype(int)
    x1, y1 = np.minimum(np.ceil(box[2:]+[pad_x, pad_y]), [w, h]).astype(int)
    roi = np.zeros((h, w), bool)
    roi[y0:y1, x0:x1] = True
    mask = hue_masks[color] & (sat > 90) & (value > 35) & roi & np.isfinite(depth) & (depth > .05) & (depth < 3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count < 2:
        raise ValueError("no target-color depth component inside model bounding box")
    label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    # Select before erosion: erosion may split a narrow top face from the side.
    interior = cv2.erode((labels == label).astype(np.uint8), np.ones((3, 3), np.uint8))
    ys, xs = np.where(interior)
    if len(xs) < 8:
        raise ValueError("target has too few valid depth pixels")
    z = depth[ys, xs]
    fx, fy, cx, cy = packet["intrinsics"]
    # MuJoCo camera axes: +x image-right, +y image-up, forward is -z.
    camera_points = np.column_stack([(xs-cx)*z/fx, -(ys-cy)*z/fy, -z])
    points = camera_points @ transform[:3, :3].T + transform[:3, 3]
    up = np.asarray(packet["up_pelvis"], float)
    up /= np.linalg.norm(up)
    heights = points @ up
    # The horizontal top face produces a narrow height mode. Reject mixed
    # foreground depth at anti-aliased boundaries; use geometric extents rather
    # than pixel-density medians (which bias the center toward the camera).
    edges = np.arange(heights.min(), max(heights.max()+.001, heights.min()+.002), .001)
    hist, edges = np.histogram(heights, bins=edges)
    top_height = edges[np.argmax(hist)] + .0005
    top_points = points[np.abs(heights-top_height) < .0015]
    top = (top_points.min(axis=0)+top_points.max(axis=0))/2
    top += up*(np.median(top_points @ up) - top @ up)
    center = top - .5*cube_height*up
    if not (.08 < center[0] < .6 and abs(center[1]) < .5 and -.15 < center[2] < .5):
        raise ValueError(f"visual target outside tabletop reach envelope: {center.tolist()}")
    return {"center_pelvis": center, "top_pelvis": top, "up_pelvis": up,
            "pixels": len(xs), "bbox_pixels": box.tolist()}


def resolve_grounding(reply, packet, instruction):
    """Respect the VLM's native pointing format; never choose an object by text color."""
    from model_server.grounding import parse_grounding
    if "color" in reply and "bbox" in reply:
        return reply
    parsed = parse_grounding(reply.get("raw_text", ""))
    if parsed.get("format") != "native_point_2d":
        return parsed
    bgr, _, _ = decode_rgbd(packet)
    h, w = bgr.shape[:2]
    point = np.asarray(parsed["point_2d"])
    x, y = np.rint(point*[w/1000, h/1000]).astype(int)
    patch = bgr[max(0,y-5):min(h,y+6), max(0,x-5):min(w,x+6)]
    if patch.size == 0:
        raise ValueError("grounding point is outside the camera")
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    valid = (hsv[:,:,1] > 90) & (hsv[:,:,2] > 35)
    hues = hsv[:,:,0][valid]
    if len(hues) < 5:
        raise ValueError("model point has no identifiable colored cube pixels")
    votes = {"red": np.count_nonzero((hues<12)|(hues>168)), "blue": np.count_nonzero((hues>95)&(hues<135)),
             "green": np.count_nonzero((hues>35)&(hues<90)), "yellow": np.count_nonzero((hues>=20)&(hues<=35)),
             "orange": np.count_nonzero((hues>=12)&(hues<20))}
    color = max(votes, key=votes.get)
    if votes[color] < .6*len(hues):
        raise ValueError("ambiguous color at model-selected point")
    command = instruction.casefold()
    right = "right hand" in command or "右手" in command
    left = "left hand" in command or "左手" in command
    if left == right:
        raise ValueError("this benchmark requires one explicit left/right hand command")
    return {**parsed, "color": color, "hand": "right" if right else "left",
            "hand_source": "explicit command parser", "color_source": "pixels at UnifoLM-selected point",
            "bbox": np.r_[np.maximum(point-100,0), np.minimum(point+100,1000)].tolist()}
