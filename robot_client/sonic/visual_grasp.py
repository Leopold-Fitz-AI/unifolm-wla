"""UnifoLM semantic grounding + RGB-D/IK grasp execution through native SONIC.

This is a hybrid controller, explicitly separate from the raw VLA action head.
It never reads the task evaluator or the simulator's object poses.
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
import time

import cv2
import numpy as np
import zmq

from .episode import write_episode
from .kinematics import G1Kinematics, pose_matrix
from .planner import PlannerMapper
from .protocol import pack_command
from .rgbd import decode_rgbd, locate_cube, resolve_grounding
from .run import load_config
from .transport import LatestSubscriber, PolicyClient, gripper_closed_fraction

LOG = logging.getLogger(__name__)
SOURCE = "unifolm_vlm_rgbd_geometric"


async def run(config, sonic_root, output, rgbd_endpoint, max_seconds=120):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    kin = G1Kinematics(sonic_root, config["reference_frame"], config["tool_offsets"])
    physical = G1Kinematics(sonic_root, config["reference_frame"], [[.14, 0, 0, 0, 0, 0]] * 2)
    tool = [pose_matrix(p) for p in config["tool_offsets"]]
    planner = PlannerMapper(**config.get("planner", {}))
    context = zmq.Context()
    states = LatestSubscriber(context, config["state_endpoint"], b"g1_debug")
    rgbds = LatestSubscriber(context, rgbd_endpoint, b"rgbd")
    pub = context.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(config["action_bind"])
    hand_open, hand_closed = np.asarray(config["hand_open"]), np.asarray(config["hand_closed"])
    opened, closed = np.asarray(config["gripper_open_value"]), np.asarray(config["gripper_closed_value"])
    ready_fraction = gripper_closed_fraction(config.get("ready_gripper_value", opened), opened, closed)
    ready_hand = hand_open + ready_fraction[:, None]*(hand_closed-hand_open)
    dt = 1/config["control_fps"]
    epoch = time.monotonic()
    episode_id = str(time.time_ns())
    write_episode(config.get("episode_state_file"), episode_id, "warming", SOURCE)
    completed = False
    feedback_start = None
    ready_seed = ready_goal = previous = solved = None
    ground_task = None
    grounding = None
    phase = "warming"
    phase_start = None
    entry = target = orientation = other_pose = None
    selected = None
    offset = np.zeros(3)
    last_start = last_ik = last_vision = float("-inf")
    packets = 0
    phases = []
    trace = (output/"control.jsonl").open("w", buffering=1)
    try:
        async with PolicyClient(config["policy_url"], 60) as policy:
            if "ground_object" not in policy.metadata.get("capabilities", []):
                raise RuntimeError("model server does not advertise ground_object")
            while time.monotonic()-epoch < max_seconds:
                tick = time.monotonic()
                if tick-last_start > .5:
                    pub.send(pack_command(start=True))
                    last_start = tick
                state = states.read(tick, config["sensor_max_age"])
                rgbd = rgbds.read(tick, config["sensor_max_age"])
                if state is None or rgbd is None:
                    if feedback_start is not None or tick-epoch > 40:
                        raise TimeoutError("state/RGB-D stream missing or stale")
                    await asyncio.sleep(dt)
                    continue
                if feedback_start is None:
                    feedback_start = tick
                warm = config.get("startup_hold_seconds", 5)
                elapsed = tick-feedback_start
                if elapsed < warm:
                    pub.send(planner.idle_packet(state, config["base_height"], dt))
                    await asyncio.sleep(dt)
                    continue
                q = kin.seed_q(state["body_q"])
                current = physical.forward(q)
                if ready_goal is None:
                    ready_seed = q.copy()
                    ready_goal = kin.solve([pose_matrix(p) for p in config["ready_pose"]], q)
                ready_elapsed = elapsed-warm
                duration = config["ready_transition_seconds"]
                if ready_elapsed < duration+config["ready_settle_seconds"]:
                    ratio = np.clip(ready_elapsed/duration, 0, 1)
                    command = ready_seed+(ratio*ratio*(3-2*ratio))*(ready_goal-ready_seed)
                    hands = ready_hand
                elif grounding is None:
                    command, hands = ready_goal.copy(), ready_hand
                    if ground_task is None:
                        image, depth, transform = decode_rgbd(rgbd)
                        cv2.imwrite(str(output/"grounding-input.jpg"), image)
                        np.savez_compressed(output/"grounding-sensor.npz", depth=depth,
                                            camera_to_pelvis=transform, intrinsics=rgbd["intrinsics"])
                        ground_task = asyncio.create_task(policy.request("ground_object", obs={
                            "instruction": config["instruction"], "observation.images.cam_left_high": image}))
                        LOG.info("Requesting UnifoLM visual grounding: %s", config["instruction"])
                    if ground_task.done():
                        reply = ground_task.result()
                        (output/"grounding-raw.json").write_text(json.dumps(reply, indent=2))
                        grounding = resolve_grounding(reply, rgbd, config["instruction"])
                        (output/"grounding.json").write_text(json.dumps(grounding, indent=2))
                        if grounding.get("hand") not in ("left", "right"):
                            raise ValueError("invalid grounded hand")
                        selected = ("left", "right").index(grounding["hand"])
                        detection = locate_cube(rgbd, grounding)
                        target = detection["center_pelvis"]
                        orientation = current[selected][:3, :3].copy()
                        other_pose = kin.forward(q)[1-selected].copy()
                        entry = current[selected][:3, 3].copy()
                        solved = q.copy()
                        phase, phase_start = "approach", tick
                        phases.append({"phase": phase, "wall_time": time.time()})
                        (output/"grounding.json").write_text(json.dumps(grounding, indent=2))
                        (output/"visual-target.json").write_text(json.dumps({
                            "center_pelvis": target.tolist(), "pixels": detection["pixels"],
                            "source": "VLM bbox + RGB-D pixels and robot calibration"}, indent=2))
                        write_episode(config.get("episode_state_file"), episode_id, "running", SOURCE)
                        LOG.info("VLM selected %s %s at %s", grounding["color"], grounding["hand"], target)
                else:
                    durations = {"approach": 4.0, "descend": 6.0, "settle": 3.0, "close": 3.0, "lift": 6.0, "hold": 8.0}
                    age = tick-phase_start
                    if phase in ("approach", "descend") and tick-last_vision > .2:
                        try:
                            detection = locate_cube(rgbd, grounding)
                            target = .7*target+.3*detection["center_pelvis"]
                        except ValueError:
                            # Occlusion by the approaching hand is expected;
                            # retain the last sensor-derived target, never a sim pose.
                            pass
                        last_vision = tick
                    up = np.asarray(rgbd["up_pelvis"])
                    if phase == "approach": goal = target+.12*up
                    elif phase in ("descend", "settle", "close"): goal = target+.003*up
                    else: goal = target+.173*up
                    ratio = np.clip(age/durations[phase], 0, 1)
                    desired = entry+(ratio*ratio*(3-2*ratio))*(goal-entry)
                    if tick-last_ik >= .08:
                        error = desired-current[selected][:3, 3]
                        offset = np.clip(offset+.5*(tick-last_ik if np.isfinite(last_ik) else dt)*error, -.02, .02)
                        wrist_goal = np.eye(4)
                        wrist_goal[:3, :3] = orientation
                        wrist_goal[:3, 3] = desired+offset-orientation @ [.14, 0, 0]
                        # Keep the idle arm's measured pose feasible while the
                        # waist balances; its joints retain the reset reference.
                        solve_seed = q.copy()
                        solve_seed[12:15] = ready_goal[12:15]
                        targets = kin.forward(solve_seed)
                        targets[selected] = wrist_goal @ tool[selected]
                        solved = kin.solve(targets, solve_seed, position_tolerance=.01)
                        idle = slice(15+7*(1-selected), 22+7*(1-selected))
                        solved[idle] = ready_goal[idle]
                        trace.write(json.dumps({"wall_time": time.time(), "phase": phase,
                            "sensor_simulation_time": rgbd["simulation_time"],
                            "target_pelvis": target.tolist(), "desired_tcp_pelvis": desired.tolist(),
                            "measured_tcp_pelvis": current[selected][:3, 3].tolist(),
                            "position_integral": offset.tolist(), "measured_q": q.tolist(),
                            "reference_q": solved.tolist(), "hand": grounding["hand"]})+"\n")
                        last_ik = tick
                    command = solved.copy()
                    hands = ready_hand.copy()
                    hands[selected] = hand_closed[selected] if phase in ("close", "lift", "hold") else hand_open[selected]
                    if age >= durations[phase]:
                        if phase == "hold":
                            completed = True
                            break
                        phase = {"approach": "descend", "descend": "settle", "settle": "close", "close": "lift", "lift": "hold"}[phase]
                        phase_start = tick
                        entry = current[selected][:3, 3].copy()
                        offset[:] = 0
                        phases.append({"phase": phase, "wall_time": time.time()})
                        LOG.info("Grasp phase: %s", phase)
                if previous is not None:
                    command = previous + np.clip(command-previous, -config["max_joint_velocity"]*dt, config["max_joint_velocity"]*dt)
                velocity = np.zeros(29) if previous is None else (command-previous)/dt
                pub.send(planner.packet(command, velocity, [0, 0, 0, config["base_height"]], state, dt, hands))
                previous = command.copy()
                packets += 1
                await asyncio.sleep(max(0, dt-(time.monotonic()-tick)))
            if not completed:
                raise TimeoutError(f"visual grasp timed out in {phase}")
            return {"status": "passed", "policy_source": SOURCE, "instruction": config["instruction"],
                    "grounding": grounding, "phases": phases, "packets": packets,
                    "elapsed_seconds": time.monotonic()-epoch,
                    "note": "control sequence completed; physical task success is scored independently"}
    finally:
        write_episode(config.get("episode_state_file"), episode_id, "passed" if completed else "failed", SOURCE)
        if ground_task is not None and not ground_task.done():
            ground_task.cancel()
            await asyncio.gather(ground_task, return_exceptions=True)
        for _ in range(3):
            pub.send(pack_command(stop=True))
            await asyncio.sleep(.05)
        states.close()
        rgbds.close()
        pub.close()
        context.term()
        trace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("grasp_client.yaml"))
    parser.add_argument("--instruction")
    parser.add_argument("--rgbd-endpoint", default="tcp://127.0.0.1:15558")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--simulation", action="store_true", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if args.instruction:
        config["instruction"] = args.instruction
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        result = asyncio.run(run(config, args.sonic_root, args.output, args.rgbd_endpoint, args.max_seconds))
    except Exception as exc:
        (args.output/"client.json").write_text(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        raise
    (args.output/"client.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
