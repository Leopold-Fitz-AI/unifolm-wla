"""Own a PTY for the official SONIC controller, restricted to local simulation.

The JSON command file accepts {"sequence": 1, "keys": "]"}; increment sequence
for each command. Keys are exactly the official controller's keyboard input.
This avoids depending on an SSH terminal remaining open during engine builds.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import socket
import subprocess
import time
import tty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--input-mode", choices=("planner", "joint"), default="planner")
    args = parser.parse_args()
    root = args.workspace.resolve()
    lock = (root / "artifacts/controller.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("this workspace already has an active controller launcher") from exc
    # Also detect manually launched native controllers before opening a PTY.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", 15557))
        except OSError as exc:
            raise RuntimeError("SONIC state port 15557 is occupied; inspect the existing controller") from exc
    sonic = root / "GR00T-WholeBodyControl"
    master, slave = pty.openpty()
    tty.setraw(slave)
    env = dict(os.environ, SONIC_DDS_DOMAIN="73")
    command = [str(sonic / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"), "lo",
               str(root / "models/GEAR-SONIC/model_decoder.onnx"), str(root / "artifacts/startup-reference"),
               "--encoder-file", str(root / "models/GEAR-SONIC/model_encoder.onnx"),
               "--obs-config", str(root / "models/GEAR-SONIC/observation_config.yaml"),
               "--policy-precision", "32", "--input-type", "zmq_manager" if args.input_mode == "planner" else "zmq", "--output-type", "zmq",
               "--zmq-port", "15556", "--zmq-out-port", "15557", "--disable-crc-check"]
    if args.input_mode == "planner":
        # Upstream infers planner schema from the directory name "V2".
        planner_file = root / "models/GEAR-SONIC/planner/target_vel/V2/planner_sonic.onnx"
        planner_file.parent.mkdir(parents=True, exist_ok=True)
        if not planner_file.exists():
            planner_file.symlink_to(root / "models/GEAR-SONIC/planner_sonic.onnx")
        command += ["--planner-file", str(planner_file), "--planner-precision", "32"]
    process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave, env=env,
                               cwd=sonic / "gear_sonic_deploy", start_new_session=True)
    os.close(slave)
    control = root / "artifacts/controller-command.json"
    # A restart must not replay commands left by the preceding run.
    sequence = json.loads(control.read_text())["sequence"] if control.exists() else None
    deadline = time.monotonic() + args.seconds
    print(f"SONIC simulation controller PID {process.pid}; command file: {control}", flush=True)
    try:
        with (root / "logs/controller.log").open("wb") as log:
            while process.poll() is None and time.monotonic() < deadline:
                ready, _, _ = select.select([master], [], [], .05)
                if ready:
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    log.write(data)
                    log.flush()
                if control.exists():
                    try:
                        request = json.loads(control.read_text())
                    except json.JSONDecodeError:
                        continue
                    if request["sequence"] != sequence:
                        keys = request["keys"]
                        if any(key not in "]Oo\r\nTtIi" for key in keys):
                            raise ValueError("unsupported controller key")
                        os.write(master, keys.encode())
                        sequence = request["sequence"]
                        print(f"sent controller keys {keys!r}", flush=True)
            if process.poll() is not None and process.returncode:
                raise RuntimeError(f"SONIC exited {process.returncode}; see controller.log")
    finally:
        if process.poll() is None:
            os.write(master, b"O")
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
        os.close(master)


if __name__ == "__main__":
    main()
