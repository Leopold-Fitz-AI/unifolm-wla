"""Tests for `model_server/tools/urdf_to_joint_limits.py`.

Uses a synthetic URDF fixture so the tests do not depend on
`third_party/unitree_ros/` being cloned (that directory is gitignored). The
fixture nevertheless reproduces the two details that make real G1 limits
error-prone:

  * `hip_roll` is **not** mirror-symmetric between left and right, and
  * the unified `waist_joint` slot is ordered `[yaw, roll, pitch]`.

One opt-in test validates the actually-generated file when the official URDF is
present; it skips otherwise.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from safety import load_joint_limits
from tools.urdf_to_joint_limits import (
    DEFAULT_SLOT_MAP,
    build_limits,
    main,
    parse_urdf_joint_limits,
    render_yaml,
)
from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_URDF = (
    REPO_ROOT
    / "third_party/unitree_ros/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf"
)

# Degrees, mirroring the shape of the official description (values are the real
# ones where they matter for the assertions below).
_JOINTS_DEG = {
    "left_hip_pitch_joint": (-145.0, 165.0),
    "left_hip_roll_joint": (-30.0, 170.0),
    "left_hip_yaw_joint": (-158.0, 158.0),
    "left_knee_joint": (-5.0, 165.0),
    "left_ankle_pitch_joint": (-50.0, 30.0),
    "left_ankle_roll_joint": (-15.0, 15.0),
    "right_hip_pitch_joint": (-145.0, 165.0),
    "right_hip_roll_joint": (-170.0, 30.0),
    "right_hip_yaw_joint": (-158.0, 158.0),
    "right_knee_joint": (-5.0, 165.0),
    "right_ankle_pitch_joint": (-50.0, 30.0),
    "right_ankle_roll_joint": (-15.0, 15.0),
    "waist_yaw_joint": (-150.0, 150.0),
    "waist_roll_joint": (-29.8, 29.8),
    "waist_pitch_joint": (-29.8, 29.8),
}


def _write_urdf(path: Path, joints_deg: dict = _JOINTS_DEG, drop: tuple = ()) -> Path:
    parts = ['<?xml version="1.0"?>', '<robot name="g1">']
    for name, (lo, hi) in joints_deg.items():
        if name in drop:
            continue
        parts.append(
            f'  <joint name="{name}" type="revolute">'
            f'<limit lower="{math.radians(lo):.6f}" upper="{math.radians(hi):.6f}"'
            f' effort="25" velocity="32"/></joint>'
        )
    parts.append("</robot>")
    path.write_text("\n".join(parts))
    return path


@pytest.fixture
def urdf(tmp_path) -> Path:
    return _write_urdf(tmp_path / "g1.urdf")


# ── parsing ───────────────────────────────────────────────────────────────


def test_parse_returns_lower_and_upper_in_radians(urdf):
    limits = parse_urdf_joint_limits(urdf)
    assert len(limits) == len(_JOINTS_DEG)
    jtype, lo, hi = limits["left_hip_pitch_joint"]
    assert jtype == "revolute"
    assert lo == pytest.approx(math.radians(-145.0))
    assert hi == pytest.approx(math.radians(165.0))


def test_parse_rejects_file_without_limits(tmp_path):
    p = tmp_path / "empty.urdf"
    p.write_text('<?xml version="1.0"?><robot name="x"><link name="a"/></robot>')
    with pytest.raises(ValueError, match="no joint with both"):
        parse_urdf_joint_limits(p)


# ── the two gotchas ──────────────────────────────────────────────────────


def test_hip_roll_asymmetry_is_preserved(urdf):
    """The single most likely silent error: copying the left column to the right."""
    soft, hard, _ = build_limits(parse_urdf_joint_limits(urdf), DEFAULT_SLOT_MAP, 0.0)
    left, right = soft["left_leg_joint"], soft["right_leg_joint"]
    assert left[0][1] == pytest.approx(math.radians(-30.0))   # left hip_roll lower
    assert left[1][1] == pytest.approx(math.radians(170.0))
    assert right[0][1] == pytest.approx(math.radians(-170.0))  # right hip_roll lower
    assert right[1][1] == pytest.approx(math.radians(30.0))
    assert left != right, "left/right leg limits must not be identical"
    # ...while the joints that ARE symmetric stay symmetric
    assert left[0][0] == pytest.approx(right[0][0])  # hip_pitch lower
    assert left[1][0] == pytest.approx(right[1][0])  # hip_pitch upper


def test_waist_is_emitted_in_dataset_order_yaw_roll_pitch():
    """`waist_joint` is [yaw, roll, pitch]; pivot output is [yaw, pitch, roll]."""
    assert DEFAULT_SLOT_MAP["waist_joint"] == [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ]
    # Give each axis a distinguishable range and check the order survives.
    deg = dict(_JOINTS_DEG)
    deg["waist_yaw_joint"] = (-150.0, 150.0)
    deg["waist_roll_joint"] = (-10.0, 10.0)
    deg["waist_pitch_joint"] = (-20.0, 20.0)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = _write_urdf(Path(d) / "w.urdf", deg)
        soft, _, _ = build_limits(parse_urdf_joint_limits(p), DEFAULT_SLOT_MAP, 0.0)
    upper = soft["waist_joint"][1]
    assert upper == pytest.approx([math.radians(150.0), math.radians(10.0), math.radians(20.0)])


# ── margins ───────────────────────────────────────────────────────────────


def test_margin_shrinks_both_ends(urdf):
    soft, hard, _ = build_limits(parse_urdf_joint_limits(urdf), DEFAULT_SLOT_MAP, math.radians(5.0))
    for slot in soft:
        for lo_h, hi_h, lo_s, hi_s in zip(hard[slot][0], hard[slot][1], soft[slot][0], soft[slot][1]):
            assert lo_s == pytest.approx(lo_h + math.radians(5.0))
            assert hi_s == pytest.approx(hi_h - math.radians(5.0))
            assert lo_s < hi_s


def test_zero_margin_equals_hard_limits(urdf):
    soft, hard, _ = build_limits(parse_urdf_joint_limits(urdf), DEFAULT_SLOT_MAP, 0.0)
    assert soft == hard


def test_excessive_margin_is_rejected(tmp_path):
    deg = dict(_JOINTS_DEG)
    deg["left_ankle_roll_joint"] = (-2.0, 2.0)  # only 4 deg of travel
    p = _write_urdf(tmp_path / "tight.urdf", deg)
    with pytest.raises(ValueError, match="collapses joint"):
        build_limits(parse_urdf_joint_limits(p), DEFAULT_SLOT_MAP, math.radians(5.0))


def test_missing_joint_is_reported(tmp_path):
    p = _write_urdf(tmp_path / "missing.urdf", drop=("left_knee_joint",))
    _, _, missing = build_limits(parse_urdf_joint_limits(p), DEFAULT_SLOT_MAP, 0.0)
    assert missing == ["left_knee_joint"]


# ── output contract ──────────────────────────────────────────────────────


def test_written_file_loads_and_widths_match_slices(tmp_path):
    p = _write_urdf(tmp_path / "g1.urdf")
    out = tmp_path / "limits.yaml"
    assert main(["--urdf", str(p), "--out", str(out), "--margin-deg", "5"]) == 0

    limits = load_joint_limits(out)  # the real consumer
    for slot in ("left_leg_joint", "right_leg_joint", "waist_joint"):
        expected = SLICES[slot].stop - SLICES[slot].start
        assert limits[slot].shape == (expected, 2), slot
        assert (limits[slot][:, 0] < limits[slot][:, 1]).all(), slot
    # slots that have no URDF counterpart are emitted as nulls and must be skipped
    for slot in ("base_vx_vy", "base_vw", "height", "left_xyz_rotvec", "right_xyz_rotvec"):
        assert slot not in limits, slot


def test_generated_slot_names_are_real_slices_keys():
    """`base_command` is a group label, not a slot -- emitting it must not happen."""
    from tools.urdf_to_joint_limits import NULL_SLOTS

    for slot, _width, _why in NULL_SLOTS:
        assert slot in SLICES, f"{slot!r} is not a SLICES key; load_joint_limits will reject it"
    assert all(s != "base_command" for s, _, _ in NULL_SLOTS)


@pytest.mark.parametrize(
    "name",
    ["joint_limits.example.yaml", "joint_limits.g1_dex1.yaml"],
)
def test_shipped_limit_files_load_cleanly(name):
    """Every limits file we ship must be accepted by the real loader.

    Regression for a bug where `joint_limits.example.yaml` used the group label
    `base_command`, which `load_joint_limits` rejects outright -- the template
    would have failed the moment anyone pointed `--joint_limits` at it.
    """
    path = REPO_ROOT / "model_server" / name
    if not path.is_file():
        pytest.skip(f"{name} not present")
    limits = load_joint_limits(path)  # raises on any unknown slot / bad width
    for slot, bounds in limits.items():
        width = SLICES[slot].stop - SLICES[slot].start
        assert bounds.shape == (width, 2), slot


def test_cli_errors_on_missing_urdf(tmp_path, capsys):
    assert main(["--urdf", str(tmp_path / "nope.urdf"), "--out", str(tmp_path / "o.yaml")]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_errors_on_wrong_variant(tmp_path, capsys):
    p = _write_urdf(tmp_path / "partial.urdf", drop=("waist_roll_joint",))
    assert main(["--urdf", str(p), "--out", str(tmp_path / "o.yaml")]) == 3
    assert "absent from" in capsys.readouterr().err


def test_list_joints_mode(urdf, capsys):
    assert main(["--urdf", str(urdf), "--list-joints"]) == 0
    out = capsys.readouterr().out
    assert "left_knee_joint" in out and "rad" in out


def test_rendered_yaml_is_parseable_and_carries_provenance(urdf):
    soft, hard, _ = build_limits(parse_urdf_joint_limits(urdf), DEFAULT_SLOT_MAP, math.radians(5.0))
    text = render_yaml(soft, hard, urdf, 5.0)
    doc = yaml.safe_load(text)
    assert set(doc) >= {"left_leg_joint", "right_leg_joint", "waist_joint"}
    assert "hard (URDF)" in text
    assert "mode_machine" in text  # the "did you pick the right file" reminder


# ── optional: the real generated artifact ────────────────────────────────


@pytest.mark.skipif(not REAL_URDF.is_file(), reason="official URDF not cloned (third_party/ is gitignored)")
def test_real_g1_dex1_yaml_matches_the_official_urdf():
    """Guard the checked-in `joint_limits.g1_dex1.yaml` against silent drift."""
    generated = REPO_ROOT / "model_server" / "joint_limits.g1_dex1.yaml"
    assert generated.is_file(), "run the generator to (re)create it"
    limits = load_joint_limits(generated)
    urdf_limits = parse_urdf_joint_limits(REAL_URDF)

    # Every bound must sit inside the URDF's hard range and be strictly tighter
    # (the file is generated with a 5 deg margin).
    for slot, joints in DEFAULT_SLOT_MAP.items():
        for i, jname in enumerate(joints):
            _, lo_h, hi_h = urdf_limits[jname]
            lo, hi = limits[slot][i]
            assert lo_h < lo < hi < hi_h, f"{slot}[{i}] ({jname}) not inside hard range"
