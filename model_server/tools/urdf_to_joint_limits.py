#!/usr/bin/env python3
"""Extract G1 joint limits from the official Unitree URDF into a
`--joint_limits` file for `model_server/action_server_wbc_msgpack_unitree.py`.

Why a script instead of a checked-in table
------------------------------------------
The authoritative source is the official robot description, not this repository:

    https://github.com/unitreerobotics/unitree_ros  ->  robots/g1_description/

so the limits are **derived**, not transcribed. Re-run this when the description
changes, and the provenance stays auditable.

Which URDF
----------
Pick the file that matches your machine's `mode_machine` (visible in the Unitree
app under Device -> Data -> Robot -> Machine Type). For the Dex1 two-finger
configuration the description matrix lists:

    g1_29dof_mode_15_with_dex1_1.urdf   mode_machine=15   leg 6x2, waist 3, arm 7x2, hand 2x2

Use `--list-joints` to dump every joint+limit in a URDF before trusting it.

Two things that bite
--------------------
1. **Left/right are NOT mirror-symmetric for `hip_roll`.** In the official
   description the left hip roll is about [-30, +170] deg and the right is
   [-170, +30] deg. Copying the left column to the right produces a range that
   looks plausible and is wrong.
2. **The unified `waist_joint` slot is ordered `[yaw, roll, pitch]`** -- the
   dataset order, which is also the natural order of the URDF joint names here.
   That is *not* the order the server emits in `action.pivot`, where it remaps to
   `[yaw, pitch, roll]` (see `_encode_action`). Emitting these bounds in the
   wrong order silently clamps the wrong axis.

About the margin
----------------
A clip bound set exactly at the mechanical limit only fires once the joint is
already at the stop, so it is useless as a safety net. `--margin-deg` shrinks
each range inward by that many degrees on both ends. The default (5 deg) is a
**recommendation, not a specification** -- pick it from your controller's
tracking error and your risk tolerance, and validate on hardware with the robot
suspended.

Usage
-----
    python -m model_server.tools.urdf_to_joint_limits \
        --urdf third_party/unitree_ros/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf \
        --out model_server/joint_limits.g1_dex1.yaml --margin-deg 5
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Unified slot -> the URDF joint names that fill it, in unified-index order.
# The leg order follows the description's own joint ordering and
# `docs/robot_action_state_processing.md` S3.1.1 / S16.19:
#   hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
DEFAULT_SLOT_MAP: Dict[str, List[str]] = {
    "left_leg_joint": [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ],
    "right_leg_joint": [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ],
    # DATASET ORDER == [yaw, roll, pitch]. Not the action.pivot order.
    "waist_joint": [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ],
}

# Slots the server also accepts limits for, but which have no URDF counterpart
# (they are task-space/command quantities). Emitted as explicit nulls so the
# file documents that they were considered and deliberately left unset.
#
# NOTE: names here must be real `SLICES` keys. `base_command` is a *group label*
# in the dataset config (it expands to base_vx_vy + base_vw + height), and
# `load_joint_limits` rejects it -- that mistake is what
# `test_shipped_example_file_loads` now guards against.
NULL_SLOTS = [
    ("base_vx_vy", "2", "(vx, vy) -- base velocity command, not a joint"),
    ("base_vw", "1", "(vw) -- base yaw-rate command, not a joint"),
    ("height", "1", "body height command, not a joint"),
    ("left_xyz_rotvec", "6", "(dx, dy, dz, drx, dry, drz) -- relative EE delta, not a joint"),
    ("right_xyz_rotvec", "6", "same as left"),
]


def parse_urdf_joint_limits(urdf_path: Path) -> Dict[str, Tuple[str, float, float]]:
    """Return {joint_name: (type, lower, upper)} for every joint carrying <limit>."""
    root = ET.parse(urdf_path).getroot()
    out: Dict[str, Tuple[str, float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name is None or limit is None:
            continue
        lower, upper = limit.get("lower"), limit.get("upper")
        if lower is None or upper is None:
            continue
        out[name] = (joint.get("type", "?"), float(lower), float(upper))
    if not out:
        raise ValueError(
            f"{urdf_path} contains no joint with both `lower` and `upper` -- is it a URDF?"
        )
    return out


def build_limits(
    urdf_limits: Dict[str, Tuple[str, float, float]],
    slot_map: Dict[str, List[str]],
    margin_rad: float,
) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]], List[str]]:
    """Return (soft, hard, missing) bounds keyed by unified slot.

    ``soft`` has ``margin_rad`` removed from both ends; ``hard`` is verbatim URDF.
    """
    soft: Dict[str, List[List[float]]] = {}
    hard: Dict[str, List[List[float]]] = {}
    missing: List[str] = []
    for slot, joints in slot_map.items():
        lows, highs, hlows, hhighs = [], [], [], []
        for jname in joints:
            if jname not in urdf_limits:
                missing.append(jname)
                continue
            _, lo, hi = urdf_limits[jname]
            hlows.append(lo)
            hhighs.append(hi)
            slo, shi = lo + margin_rad, hi - margin_rad
            if slo >= shi:
                raise ValueError(
                    f"margin {math.degrees(margin_rad):.1f} deg collapses joint {jname!r} "
                    f"({math.degrees(lo):.1f}..{math.degrees(hi):.1f} deg)"
                )
            lows.append(slo)
            highs.append(shi)
        if lows:
            soft[slot] = [lows, highs]
            hard[slot] = [hlows, hhighs]
    return soft, hard, missing


def render_yaml(
    soft: Dict[str, List[List[float]]],
    hard: Dict[str, List[List[float]]],
    urdf: Path,
    margin_deg: float,
) -> str:
    def fmt(values: Sequence[float]) -> str:
        return "[" + ", ".join(f"{v:.4f}" for v in values) + "]"

    lines = [
        "# G1 joint limits for model_server/safety.py (`--joint_limits`).",
        "#",
        "# GENERATED FILE -- do not hand-edit. Regenerate with:",
        "#   python -m model_server.tools.urdf_to_joint_limits \\",
        f"#       --urdf {urdf} \\",
        f"#       --out <this file> --margin-deg {margin_deg:g}",
        "#",
        "# Source of truth: the official Unitree robot description above.",
        "# Values below are the URDF `<limit>` values shrunk inward by",
        f"# {margin_deg:g} deg on each side, because a clip bound set exactly at the mechanical",
        "# stop only fires once the joint is already there.",
        "#",
        "# VERIFY BEFORE USE ON HARDWARE:",
        "#   - the URDF matches your machine's mode_machine",
        "#   - waist order here is the DATASET order [yaw, roll, pitch]",
        "#     (action.pivot uses [yaw, pitch, roll] -- see _encode_action)",
        "#   - hip_roll is NOT mirror-symmetric between left and right",
        "#   - validate with the robot suspended before applying any load",
        "",
    ]
    for slot in soft:
        lines.append(f"{slot}:")
        hard_lo, hard_hi = hard[slot]
        lines.append(f"  # hard (URDF): lower {fmt(hard_lo)}")
        lines.append(f"  #              upper {fmt(hard_hi)}")
        lines.append(f"  lower: {fmt(soft[slot][0])}")
        lines.append(f"  upper: {fmt(soft[slot][1])}")
        lines.append("")

    lines.append("# --- intentionally unset: no URDF counterpart -------------------------")
    for slot, width, why in NULL_SLOTS:
        lines.append(f"# {slot} ({width} dims): {why}")
        lines.append(f"{slot}:")
        lines.append("  lower: null")
        lines.append("  upper: null")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--urdf", type=Path, required=True, help="Path to the G1 URDF")
    p.add_argument("--out", type=Path, default=None, help="Write the YAML here (default: stdout)")
    p.add_argument("--margin-deg", type=float, default=5.0,
                   help="Shrink each range inward by this many degrees (0 = use hard limits)")
    p.add_argument("--list-joints", action="store_true",
                   help="Print every joint+limit in the URDF and exit (sanity check)")
    args = p.parse_args(argv)

    if not args.urdf.is_file():
        print(f"error: URDF not found: {args.urdf}", file=sys.stderr)
        return 2

    urdf_limits = parse_urdf_joint_limits(args.urdf)

    if args.list_joints:
        print(f"{args.urdf}  ({len(urdf_limits)} joints with limits)")
        for name, (jtype, lo, hi) in sorted(urdf_limits.items()):
            print(f"  {name:44s} {jtype:10s} {lo:+.4f} .. {hi:+.4f} rad "
                  f"({math.degrees(lo):+7.1f} .. {math.degrees(hi):+7.1f} deg)")
        return 0

    soft, hard, missing = build_limits(urdf_limits, DEFAULT_SLOT_MAP, math.radians(args.margin_deg))
    if missing:
        print(f"error: these joints are absent from {args.urdf.name}: {missing}\n"
              f"       Wrong URDF variant? Use --list-joints to inspect.", file=sys.stderr)
        return 3

    yaml_text = render_yaml(soft, hard, args.urdf, args.margin_deg)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml_text)
        n = sum(len(v[0]) for v in soft.values())
        print(f"wrote {args.out}  ({n} joint bounds from {len(soft)} slots, "
              f"margin {args.margin_deg:g} deg)")
    else:
        print(yaml_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
