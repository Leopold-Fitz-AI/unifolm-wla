"""Read-only preflight for a Linux SONIC simulation host."""

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess

SONIC_REVISION = "1fa5f7fa3966cbd4845c4e641d13c3f0cd5b0d9b"


def check(sonic_root, checkpoint=None):
    root = Path(sonic_root)
    report = {"platform": platform.platform(), "compatible_sonic_revision": SONIC_REVISION}
    errors = []
    if platform.system() != "Linux":
        errors.append("official SONIC C++/TensorRT simulation controller needs a Linux NVIDIA host")
    if not shutil.which("nvidia-smi"):
        errors.append("nvidia-smi not available")
    for module in ("mujoco", "zmq", "websockets", "msgpack", "scipy", "cv2", "yaml"):
        if importlib.util.find_spec(module) is None:
            errors.append(f"missing client module: {module}")
    if root.exists():
        rev = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        manifest = root / "SONIC_REVISION"
        report["sonic_revision"] = rev.stdout.strip() if not rev.returncode else (
            manifest.read_text().strip() if manifest.is_file() else "unknown")
        if report["sonic_revision"] != SONIC_REVISION:
            errors.append("SONIC revision differs from the verified protocol; review before running")
        pointers = [str(p.relative_to(root)) for p in (root / "gear_sonic/data/robot_model/model_data/g1/meshes").glob("*")
                    if p.is_file() and p.read_bytes()[:80].startswith(b"version https://git-lfs")]
        if pointers:
            errors.append(f"{len(pointers)} G1 meshes are LFS pointers; run git lfs pull in SONIC")
        report["controller_binaries"] = [str(p) for p in (root / "gear_sonic_deploy/target").rglob("g1_deploy_onnx_ref") if p.is_file()]
        if not report["controller_binaries"]:
            errors.append("SONIC controller is not built")
    else:
        errors.append("SONIC checkout does not exist")
    if checkpoint is None or not Path(checkpoint).is_file():
        errors.append("provide an existing UnifoLM checkpoint using --checkpoint")
    report["ready"] = not errors
    report["errors"] = errors
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    report = check(args.sonic_root, args.checkpoint)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
