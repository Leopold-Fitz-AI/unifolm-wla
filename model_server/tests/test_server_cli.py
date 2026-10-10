"""CLI-surface tests for `action_server_wbc_msgpack_unitree.py`.

`main()` is extracted with `ast` and executed against a stub server class, so the
argument parser can be exercised without importing the module (which needs cv2,
websockets, torch, transformers and accelerate) and without loading a checkpoint.

This is the cheap way to catch the class of bug fixed in GUIDE S8.4: a flag that
is documented but does not exist, or a default that silently disagrees with the
training configuration.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_PATH = _REPO_ROOT / "model_server" / "action_server_wbc_msgpack_unitree.py"


def _load_main_namespace():
    """Return (callable main, dict capturing the parsed args and run() call)."""
    src = _SERVER_PATH.read_text()
    tree = ast.parse(src)
    segments = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            segments["main"] = ast.get_source_segment(src, node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id == "_DEFAULT_IMAGE_SIZE":
                segments["default_image_size"] = ast.get_source_segment(src, node)
    assert "main" in segments, "could not locate main() in the server module"

    captured: dict = {}

    class StubServer:
        def __init__(self, args):
            captured["args"] = args

        def run(self, host, port):
            captured["host"] = host
            captured["port"] = port

    ns = {
        "argparse": argparse,
        "logging": logging,
        "sys": sys,
        "ActionServerWBCMsgpack": StubServer,
    }
    exec(segments["default_image_size"], ns)
    exec(segments["main"], ns)
    return ns["main"], captured, ns["_DEFAULT_IMAGE_SIZE"]


def _parse(monkeypatch, argv_extra):
    main, captured, default_size = _load_main_namespace()
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["prog", "--ckpt_path", "fake/model.safetensors", *argv_extra])
    main()
    return captured["args"], default_size


def test_safety_defaults(monkeypatch):
    args, default_size = _parse(monkeypatch, [])
    assert default_size == (336, 448)
    assert args.safety is True
    assert args.action_clip_norm == 5.0
    assert args.action_max_step == 10.0
    assert args.action_blend_steps == 0
    assert args.infer_timeout_ms == 2000.0
    assert args.require_all_cameras is True
    assert args.strict_rot6d is False
    assert args.joint_limits is None
    assert args.base_vlm is None
    # image_size stays None so the server auto-detects from the checkpoint
    assert args.image_size is None


def test_no_safety_flag(monkeypatch):
    args, _ = _parse(monkeypatch, ["--no-safety"])
    assert args.safety is False


def test_safety_numeric_overrides(monkeypatch):
    args, _ = _parse(monkeypatch, [
        "--action_clip_norm", "1.25",
        "--action_max_step", "0.5",
        "--action_blend_steps", "4",
        "--infer_timeout_ms", "250",
    ])
    assert args.action_clip_norm == 1.25
    assert args.action_max_step == 0.5
    assert args.action_blend_steps == 4
    assert args.infer_timeout_ms == 250.0


def test_boolean_optional_variants(monkeypatch):
    args, _ = _parse(monkeypatch, ["--no-require_all_cameras", "--strict_rot6d"])
    assert args.require_all_cameras is False
    assert args.strict_rot6d is True


def test_path_flags(monkeypatch):
    args, _ = _parse(monkeypatch, [
        "--joint_limits", "model_server/joint_limits.example.yaml",
        "--base_vlm", "/somewhere/tokenizer",
    ])
    assert args.joint_limits == "model_server/joint_limits.example.yaml"
    assert args.base_vlm == "/somewhere/tokenizer"


def test_image_size_override_and_disable(monkeypatch):
    args, _ = _parse(monkeypatch, ["--image_size", "320", "448"])
    assert args.image_size == [320, 448]
    args, _ = _parse(monkeypatch, ["--image_size", "0", "0"])
    assert args.image_size == [0, 0]


def test_ckpt_path_is_required(monkeypatch):
    main, _captured, _size = _load_main_namespace()
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit):
        main()


def test_no_documented_flag_is_missing(monkeypatch):
    """Regression guard for GUIDE S8.4 (`--backend` was documented but absent)."""
    args, _ = _parse(monkeypatch, [])
    assert not hasattr(args, "backend")
    # And the docstring of the server-side eval script must not advertise one.
    eval_src = (_REPO_ROOT / "model_server" / "eval_local_episode_wbc_msgpack_server_only.py").read_text()
    docstring = ast.get_docstring(ast.parse(eval_src)) or ""
    assert "--backend" not in docstring
