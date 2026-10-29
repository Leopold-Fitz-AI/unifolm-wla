"""Run one isolated SONIC tabletop trial against an already running model server.

Exit success requires the independent physical scorer, not merely completion of
the geometric motion sequence. Only subprocesses created here are stopped.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import yaml

class CollisionReportWatchdog:
    """Bound both initial reporting and later silence using a monotonic clock.

    Reading the same successful report repeatedly is not a heartbeat. A
    partial write or a replace race gets the remaining grace period, without
    resetting the deadline or being mistaken for a physical collision.
    """
    def __init__(self, *, first_report_timeout=60., stale_timeout=5., clock=time.monotonic):
        if not (0 < first_report_timeout < float('inf') and 0 < stale_timeout < float('inf')):
            raise ValueError('collision report timeouts must be finite and positive')
        self.clock=clock
        self.started=clock()
        self.first_report_timeout=first_report_timeout
        self.stale_timeout=stale_timeout
        self.last_valid_at=None
        self.last_version=None
        self.last_steps=None

    @property
    def has_valid_report(self):
        return self.last_valid_at is not None

    def check(self, directory):
        directory=Path(directory)
        marker=directory/'safety-stop.json'
        if marker.exists():return True
        now=self.clock()
        reason=details=None
        try:
            path=directory/'task-report.json'
            before=path.stat()
            report=json.loads(path.read_text())
            after=path.stat()
            version=lambda s:(s.st_ino,s.st_mtime_ns,s.st_size)
            if version(before)==version(after):
                audit=report.get('table_collision_audit',{}) if isinstance(report,dict) else {}
                if not isinstance(audit,dict) or audit.get('passed') is not True:
                    reason='independent_table_collision_audit_failed'
                    details={'audit':audit}
                else:
                    steps=audit.get('checked_physics_steps')
                    # Filesystems can expose identical metadata for successive
                    # same-sized writes. A valid increasing physics counter is
                    # itself fresh evidence; a rewritten frozen counter is not.
                    valid_steps=isinstance(steps,int) and not isinstance(steps,bool) and steps>0
                    if valid_steps:
                        progressed=self.last_steps is None or steps>self.last_steps
                    else:
                        progressed=(steps is None and self.last_steps is None
                                    and version(after)!=self.last_version)
                    if progressed:
                        self.last_valid_at=now
                        self.last_version=version(after)
                        self.last_steps=steps
        except (OSError,json.JSONDecodeError):
            pass
        if reason is None:
            reference=self.started if self.last_valid_at is None else self.last_valid_at
            limit=self.first_report_timeout if self.last_valid_at is None else self.stale_timeout
            if now-reference >= limit:
                reason='collision_report_initial_timeout' if self.last_valid_at is None else 'collision_report_stale'
                details={'seconds_without_valid_report':now-reference,'timeout_seconds':limit}
        if reason is None:return False
        marker.write_text(json.dumps(dict(reason=reason,**details),indent=2)+'\n')
        return True


def request_collision_stop(directory, watchdog=None):
    """Independent supervisor can only stop, never alter task-policy actions.

    Repeated polling must pass the same watchdog; the optional instance keeps
    single-shot collision checks convenient for callers and tests.
    """
    return (watchdog or CollisionReportWatchdog()).check(directory)


def validate_trial_policy(config, policy):
    if config.get('execution_mode')=='model_only' and policy!='vla':
        raise ValueError('model_only trials require --policy vla')


def main():
    from .run import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task", type=Path, help="Default: tabletop_vla.yaml for VLA, tabletop.yaml for visual")
    parser.add_argument("--config", type=Path, help="Default: vla_grasp.yaml for VLA, grasp_client.yaml for visual")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulation", action="store_true", required=True)
    parser.add_argument("--policy", choices=("visual", "vla"), default="visual")
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--arm-tracking", choices=("sonic", "joint_pd", "arms_pd"), default="sonic")
    parser.add_argument("--instruction", help="Override instruction; evaluation target/hand stay defined by the task")
    parser.add_argument("--time-scale", type=float, help="Uniform arm/gripper reference time dilation")
    parser.add_argument("--inference-seed", type=int, help="Fixed diffusion noise for reproducible frozen-weight inference")
    args = parser.parse_args()
    args.task = args.task or Path(__file__).with_name("tabletop_vla.yaml" if args.policy == "vla" else "tabletop.yaml")
    args.config = args.config or Path(__file__).with_name("vla_grasp.yaml" if args.policy == "vla" else "grasp_client.yaml")
    root, output = args.workspace.resolve(), args.output.resolve()
    trial_lock = (root/"artifacts/grasp-trial.lock").open("a")
    fcntl.flock(trial_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Never join an existing simulation or overwrite an earlier trial.
    for port in (15555, 15556, 15557, 15558):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    output.mkdir(parents=True, exist_ok=False)
    task = yaml.safe_load(args.task.read_text())
    if args.instruction:
        task["instruction"] = args.instruction
    config = load_config(args.config)
    validate_trial_policy(config,args.policy)
    model_only=config.get('execution_mode')=='model_only'
    if model_only:
        if not task.get('robot_contacts_from_start',False):
            raise ValueError('model_only trials require robot contacts from initialization')
        config['safety_stop_file']=str(output/'safety-stop.json')
    # A high-arm simulation boot must expose the same upper-body goal to the
    # native leg policy during landing, not merely hold it in the arm PD servo.
    config['startup_upper_body_hold']=task.get('startup_arm_q') is not None
    if config.get('visual_target'):
        config['visual_target']['camera_mounts']=task['camera_mounts']
    if config.get('synchronized_observation',False) and not task.get('synchronized_cameras',False):
        raise ValueError('synchronized observation requires the synchronized camera task publisher')
    if task.get('gripper_type') in ('parallel','dex1_cad'):
        endpoint=.0445 if task.get('gripper_type')=='dex1_cad' else .045
        if (any(abs(v) > 1e-9 for h in config['hand_open'] for v in h[:2])
            or any(abs(v-endpoint) > 1e-9 for h in config['hand_closed'] for v in h[:2])):
            raise ValueError(f'parallel task requires metre-valued hand endpoints 0/{endpoint}')
    if args.time_scale is not None:
        config["action_time_scale"] = args.time_scale
    if args.inference_seed is not None:
        config["inference_seed"] = args.inference_seed
    config["instruction"] = task["instruction"]
    config["episode_state_file"] = str(output/"episode.json")
    config["trace_dir"] = str(output/"actions")
    config["arm_tracking"] = args.arm_tracking
    (output/"task.yaml").write_text(yaml.safe_dump(task))
    (output/"client-config.yaml").write_text(yaml.safe_dump(config))
    sonic = root/"GR00T-WholeBodyControl"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="egl", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", PYTHONPATH=str(sonic)+os.pathsep+str(root/"src"))
    children, logs = [], []
    collision_watchdog=CollisionReportWatchdog() if model_only else None

    def launch(module, arguments, log):
        stream = (output/log).open("w")
        logs.append(stream)
        process = subprocess.Popen([sys.executable, "-u", "-m", module, *map(str, arguments)],
                                   cwd=root/"src", env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        children.append(process)
        return process

    try:
        rgbd_args = ["--rgbd-port", 15558] if args.policy == "visual" else []
        if config.get('grasp_hold',{}).get('enabled',False) or config.get('robot_sensor_endpoint'):
            rgbd_args = ['--tactile-port',15558]
        simulator = launch("robot_client.sonic.launch_sim", ["--sonic-root", sonic, "--headless", "--overview",
            "--camera-port", 15555, "--domain-id", 73, "--release-file", output/"release",
            "--task-config", output/"task.yaml", "--task-output", output,
            "--episode-file", output/"episode.json", *rgbd_args,
            "--arm-tracking", args.arm_tracking], "simulator.log")
        controller_log = root/"logs/controller.log"
        # Port preflight above proves no live controller owns this workspace log.
        controller_log.unlink(missing_ok=True)
        controller = launch("robot_client.sonic.launch_controller", ["--workspace", root,
                            "--input-mode", "planner", "--seconds", args.seconds+100], "controller-launcher.log")
        deadline = time.monotonic()+80
        while time.monotonic() < deadline:
            if simulator.poll() is not None or controller.poll() is not None:
                raise RuntimeError("simulator/controller exited during initialization; inspect trial logs")
            if model_only and request_collision_stop(output,collision_watchdog):
                raise RuntimeError('collision safety stop during initialization')
            if (controller_log.exists() and "Init Done" in controller_log.read_text(errors="replace")
                and (not model_only or collision_watchdog.has_valid_report)):
                break
            time.sleep(.1)
        else:
            raise TimeoutError("SONIC initialization exceeded 80 seconds")
        client_module = "robot_client.sonic.visual_grasp" if args.policy == "visual" else "robot_client.sonic.run"
        result_args = ["--output", output] if args.policy == "visual" else ["--report", output/"client.json"]
        client = launch(client_module, ["--sonic-root", sonic, "--simulation",
            "--config", output/"client-config.yaml", *result_args, "--max-seconds", args.seconds], "client.log")
        deadline = time.monotonic()+args.seconds+10
        released = False
        safety_requested_at=None
        while client.poll() is None and time.monotonic() < deadline:
            if not released and "transitioning to CONTROL" in controller_log.read_text(errors="replace"):
                (output/"release").touch()
                released = True
            if simulator.poll() is not None or controller.poll() is not None:
                raise RuntimeError("simulation/control stopped during trial")
            if model_only and request_collision_stop(output,collision_watchdog):
                if safety_requested_at is None:safety_requested_at=time.monotonic()
                if time.monotonic()-safety_requested_at>3:
                    raise RuntimeError('client did not honor independent collision safety stop')
            time.sleep(.1)
        if client.poll() is None:
            raise TimeoutError("grasp client did not finish")
        if client.returncode:
            raise RuntimeError("grasp client failed; inspect client.log")
        # The physics loop observes the atomic episode completion marker.
        deadline = time.monotonic()+3
        report = {}
        while time.monotonic() < deadline:
            path = output/"task-report.json"
            try:
                report = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if report.get("episode", {}).get("status") in ("passed", "failed"):
                break
            time.sleep(.05)
        print(json.dumps(report, indent=2), flush=True)
        if not report.get("success"):
            raise RuntimeError("physical grasp acceptance failed; task-report.json records the measurements")
    finally:
        for process in reversed(children):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
        if (root/"logs/controller.log").exists():
            (output/"controller.log").write_bytes((root/"logs/controller.log").read_bytes())
        for stream in logs:
            stream.close()


if __name__ == "__main__":
    main()
