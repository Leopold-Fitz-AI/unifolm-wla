"""Strict structured output for the optional UnifoLM VLM grounding endpoint."""
import json
import numpy as np


def grounding_prompt(instruction):
    return (f"Robot command: {instruction}\n"
            "Identify the requested cube in this tabletop image. Return only JSON with keys "
            '"color" (red/blue/green/yellow/orange), "hand" (left/right, obey the command), '
            'and "bbox" [x_min,y_min,x_max,y_max] using coordinates normalized to 0..1000. '
            "The box must tightly enclose only the target cube. If it is absent return "
            '{"error":"target_not_visible"}. Do not include explanation.')


def parse_grounding(text):
    decoder = json.JSONDecoder()
    candidates = []
    errors = []
    points = []
    for index, char in enumerate(text):
        if char != "{": continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and all(k in value for k in ("color", "hand", "bbox")):
            if value not in candidates: candidates.append(value)
        elif isinstance(value, dict) and "error" in value:
            errors.append(value)
        elif isinstance(value, dict) and "point_2d" in value:
            point = np.asarray(value["point_2d"], float)
            if point.shape == (2,) and np.isfinite(point).all() and np.all((point >= 0) & (point <= 1000)):
                if point.tolist() not in points: points.append(point.tolist())
    if len(candidates) != 1:
        if not candidates and points:
            points = np.asarray(points)
            if np.max(np.ptp(points, axis=0)) > 200:
                raise ValueError("native grounding points refer to multiple separated objects")
            return {"format": "native_point_2d", "points_2d": points.tolist(),
                    "point_2d": points.mean(axis=0).tolist(), "raw_text": text}
        if not candidates and errors:
            raise ValueError(f"grounding failed: {errors[-1]['error']}")
        raise ValueError("grounding model returned missing/ambiguous semantic JSON")
    result = candidates[0]
    if "error" in result:
        raise ValueError(f"grounding failed: {result['error']}")
    color, hand = result.get("color", "").lower(), result.get("hand", "").lower()
    box = np.asarray(result.get("bbox"), dtype=float)
    if color not in ("red", "blue", "green", "yellow", "orange") or hand not in ("left", "right"):
        raise ValueError("grounding returned an unsupported color/hand")
    if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0) or np.any(box > 1000)
            or box[2] - box[0] < 2 or box[3] - box[1] < 2):
        raise ValueError("grounding returned an invalid normalized bounding box")
    return {"color": color, "hand": hand, "bbox": box.tolist(), "raw_text": text}
