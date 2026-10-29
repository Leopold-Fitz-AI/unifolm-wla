"""python -m robot_client.sonic.run --sonic-root PATH --simulation

The official SONIC control process is started separately in SIM mode. This
client does not open DDS, start motors, or run a controller on real hardware.
"""

import argparse
import asyncio
import logging
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import yaml
import zmq

from .adapter import ActionAdapter, Reference, make_observation
from .kinematics import G1Kinematics
from .kinematics import pose_matrix
from .episode import write_episode
from .planner import PlannerMapper
from .protocol import pack_command
from .transport import (LatestSubscriber, PolicyClient, attach_sim_hands,
                        decode_cameras, measured_grippers, model_gripper_values, gripper_closed_fraction)

LOG = logging.getLogger(__name__)
SOURCE = "unifolm_raw_vla"
MODEL_ONLY_SOURCE = "unifolm_model_only_sonic_adapter"


def execution_contract(config):
    """Validate task ownership before opening sockets or constructing helpers.

    This contract describes a model-only *task policy* through our SONIC
    adapter. It does not claim that this adapter is Unitree's WBC client.
    Validate again in run(): callers may override an already loaded config.
    """
    mode=config.get('execution_mode','legacy')
    if mode not in ('legacy','model_only'):
        raise ValueError('execution_mode must be legacy or model_only')
    assists=[key for key in ('visual_target','grasp_execution','grasp_lift_reflex',
        'grasp_hold_force','stationary_single_arm','base_position_hold',
        'align_first_ee_frame','cartesian_tracking') if config.get(key)]
    hold=config.get('grasp_hold') or {}
    if hold.get('enabled',False):assists.append('grasp_hold.enabled')
    if mode=='model_only':
        if assists:raise ValueError('model_only forbids task/trajectory overrides: '+', '.join(assists))
        if config.get('action_time_scale',1.)!=1.:
            raise ValueError('model_only requires action_time_scale=1')
        if config.get('encoder_generation_prompt',False):
            raise ValueError('model_only requires the released encoder prompt convention')
        needs_odometry=(config.get('ready_gate',{}).get('enabled',False)
                        or config.get('compensate_base_motion',False))
        if needs_odometry and not config.get('robot_sensor_endpoint'):
            raise ValueError('model_only ready/base feedback requires robot_sensor_endpoint')
        if config.get('compensate_base_motion',False) and not config.get('waist_tracking_compensation',False):
            raise ValueError('model_only base compensation requires mechanical waist tracking compensation')
        if config.get('sonic_mode','joint')!='planner' and (needs_odometry or config.get('waist_tracking_compensation',False)):
            raise ValueError('model_only ready/robot-frame compensation currently requires planner mode')
    planner_mode=config.get('sonic_mode','joint')=='planner'
    planner=config.get('planner',{})
    calibration_validation=None
    if config.get('calibration'):
        if config['calibration'].get('name')!='unitree_g1_dex1_tool105_motor54_v1':
            raise ValueError('unknown mechanical calibration')
        from .dex1_calibration import validate_dex1_calibration
        calibration_validation=validate_dex1_calibration(config)
    return {'mode':mode,'validated':True,'task_assists_enabled':assists,
        'official_wbc_client':False,
        'execution_backend':'sonic_planner_adapter' if planner_mode else 'sonic_joint_adapter',
        'leg_reference_source':'sonic_planner' if planner_mode else 'model',
        'arm_tracking':config.get('arm_tracking','sonic'),
        'reference_frame':config.get('reference_frame'),
        'tool_offsets':config.get('tool_offsets'),
        'calibration':config.get('calibration'),
        'calibration_validation':calibration_validation,
        'ready_gate':config.get('ready_gate'),
        'gripper_mapping':{'open_motor_value':config.get('gripper_open_value'),
                           'closed_motor_value':config.get('gripper_closed_value'),
                           'open_hand_joints':config.get('hand_open'),
                           'closed_hand_joints':config.get('hand_closed')},
        'waist_tracking_compensation':bool(config.get('waist_tracking_compensation',False)),
        'base_motion_compensation':bool(config.get('compensate_base_motion',False)),
        'action_time_scale':config.get('action_time_scale',1.),
        'align_first_ee_frame':bool(config.get('align_first_ee_frame',False)),
        'cartesian_tracking':bool(config.get('cartesian_tracking',False)),
        'action_timebase':config.get('action_timebase','observation'),
        'execute_full_chunk':bool(config.get('execute_full_chunk',False)),
        'robot_feedback_used':(['simulation_time','body_q','base_position','base_quat','initial_base_position']
                               if mode=='model_only' and config.get('robot_sensor_endpoint') else []),
        'limits':{'max_joint_velocity':config.get('max_joint_velocity'),
                  'safety_stop_file':config.get('safety_stop_file'),
                  'robot_pose_stop':({'min_base_height_m':.5,'max_base_tilt_degrees':25.}
                                     if mode=='model_only' and config.get('robot_sensor_endpoint') else None),
                  'gripper_range':config.get('gripper_range'),
                  'gripper_overshoot_tolerance':config.get('gripper_overshoot_tolerance',0.),
                  'truncate_infeasible_tail':bool(config.get('truncate_infeasible_tail',False)),
                  'max_ik_rejections':config.get('max_ik_rejections',0),
                  'reference_max_age':config.get('reference_max_age'),
                  'sensor_max_age':config.get('sensor_max_age'),
                  'max_action_latency':config.get('max_action_latency',2.),
                  'base_velocity_deadband':config.get('base_velocity_deadband'),
                  'planner':({'max_speed':planner.get('max_speed',.4),
                              'max_yaw_rate':planner.get('max_yaw_rate',.4),
                              'max_acceleration':planner.get('max_acceleration',.5),
                              'height_range':planner.get('height_range',[.5,.8]),
                              'stationary_mode':planner.get('stationary_mode',0),
                              'velocity_deadband_mps':planner.get('velocity_deadband_mps',.02),
                              'yaw_deadband_rps':planner.get('yaw_deadband_rps',.03),
                              'walking_speed_threshold_mps':planner.get('walking_speed_threshold_mps',.03),
                              'turning_rate_threshold_rps':planner.get('turning_rate_threshold_rps',.03),
                              'minimum_walking_speed_mps':planner.get('minimum_walking_speed_mps',0.),
                              'heading_tracking_gain':planner.get('heading_tracking_gain',0.),
                              'max_heading_correction_rad':planner.get('max_heading_correction_rad',.15),
                              'heading_correction_time_constant':planner.get('heading_correction_time_constant',.3),
                              'pure_turning_mode':'stationary_facing'}
                             if planner_mode else None)}}


def robot_only_feedback(sensor):
    """Do not expose contact/force or object data to the model-only executor."""
    if sensor is None:return None
    return {key:sensor[key] for key in ('simulation_time','body_q','base_position',
                                       'base_quat','initial_base_position') if key in sensor}


def robot_base_transform(sensor):
    transform=np.eye(4)
    quat=np.asarray(sensor['base_quat'],float)
    position=np.asarray(sensor['base_position'],float)
    if quat.shape!=(4,) or position.shape!=(3,) or not np.isfinite(np.r_[quat,position]).all() or not np.isclose(np.linalg.norm(quat),1,atol=1e-3):
        raise ValueError('invalid robot odometry')
    transform[:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
    transform[:3,3]=position
    return transform


def emergency_stop_check(config, robot_feedback=None):
    """Abort only; never use external audit/contact information to guide actions."""
    stop_file=config.get('safety_stop_file')
    if stop_file and Path(stop_file).exists():
        raise RuntimeError('external safety stop requested')
    if config.get('execution_mode')=='model_only' and robot_feedback is not None:
        transform=robot_base_transform(robot_feedback)
        height=float(transform[2,3])
        tilt=float(np.rad2deg(np.arccos(np.clip(transform[2,2],-1.,1.))))
        if height<=.5 or tilt>=25.:
            raise RuntimeError(f'robot pose safety stop: base height {height:.3f} m, tilt {tilt:.2f} degrees')


def compensate_model_targets(kin,nominal,measured,observation_base=None,current_base=None):
    """Track both model EE goals at measured waist/base; retain model waist.

    No hand selection, pose latch, object target or gripper adjustment occurs.
    An unreachable mechanical correction stops rather than inventing a goal.
    """
    targets=kin.forward(nominal)
    if observation_base is not None:
        transfer=np.linalg.inv(current_base)@observation_base
        targets=[transfer@target for target in targets]
    seed=np.asarray(nominal,float).copy();seed[:15]=measured[:15]
    result=kin.solve(targets,seed,position_tolerance=.01,rotation_tolerance=.05)
    result[:15]=nominal[:15]
    return result


def startup_upper_body_packet(planner, kin, state, config, dt, anchor=None, base=None):
    """Give SONIC the same measured high-arm reference held by the boot servo.

    Capture once: chasing changing measurements would conceal drift. Only the
    planner's 17 upper-body channels are encoded; legs remain under SONIC.
    """
    if anchor is None:anchor=kin.seed_q(state['body_q'],planner_legs=True)
    packet=planner.packet(anchor,np.zeros(29),[0,0,0,config['base_height']] if base is None else base,
                          state,dt,np.asarray(config['hand_open'],float))
    return packet,anchor


def load_config(path):
    def read(current, seen=()):
        current = Path(current).resolve()
        if current in seen or len(seen) >= 8:
            raise ValueError("cyclic or excessively nested base_config")
        values = yaml.safe_load(current.read_text())
        if "base_config" in values:
            parent = read(current.parent/values.pop("base_config"), (*seen,current))
            values = dict(parent, **values)
        return values
    config = read(path)
    if config.get("sonic_mode", "joint") not in ("joint", "planner"):
        raise ValueError("sonic_mode must be joint or planner")
    if config.get("action_timebase", "observation") not in ("observation", "receipt"):
        raise ValueError("action_timebase must be observation or receipt")
    retries=config.get('max_ik_rejections',0)
    if not isinstance(retries,int) or isinstance(retries,bool) or not 0<=retries<=10:
        raise ValueError('max_ik_rejections must be an integer from 0 to 10')
    if not np.isfinite(config.get("max_action_latency", 2.0)) or config.get("max_action_latency", 2.0) <= 0:
        raise ValueError("max_action_latency must be positive and finite")
    if not np.isfinite(config.get("startup_hold_seconds", 0)) or config.get("startup_hold_seconds", 0) < 0:
        raise ValueError("startup_hold_seconds must be nonnegative and finite")
    if not np.isfinite(config.get("action_time_scale", 1)) or config.get("action_time_scale", 1) <= 0:
        raise ValueError("action_time_scale must be positive and finite")
    if "ready_waist" in config:
        waist = np.asarray(config["ready_waist"], dtype=float)
        if waist.shape != (3,) or not np.isfinite(waist).all():
            raise ValueError("ready_waist must be finite (3,)")
    if 'ready_arm_seed' in config:
        seed=np.asarray(config['ready_arm_seed'],float)
        if seed.shape!=(14,) or not np.isfinite(seed).all():
            raise ValueError('ready_arm_seed must be finite (14,)')
    for key in ("model_fps", "control_fps", "inference_interval", "policy_timeout", "sensor_max_age",
                "reference_max_age", "max_joint_velocity", "base_height"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if config["control_fps"] != 50:
        raise ValueError("released SONIC controller requires control_fps=50")
    if not isinstance(config["lookahead_frames"], int) or config["lookahead_frames"] < 2:
        raise ValueError("lookahead_frames must be an integer >= 2")
    for key in ("hand_open", "hand_closed"):
        a = np.asarray(config[key], dtype=float)
        if a.shape != (2, 7) or not np.isfinite(a).all():
            raise ValueError(f"{key} must be finite (2,7)")
    if np.any(np.linalg.norm(np.asarray(config["hand_closed"]) - config["hand_open"], axis=1) == 0):
        raise ValueError("hand endpoints must differ on each side")
    if set(config["camera_mapping"]) != {"cam_left_high", "cam_left_wrist", "cam_right_wrist"}:
        raise ValueError("all three camera mappings are required")
    if "ready_pose" in config:
        if np.asarray(config["ready_pose"]).shape != (2, 6) or not np.isfinite(config["ready_pose"]).all():
            raise ValueError("ready_pose must be finite (2,6) xyz/rpy")
        for key in ("ready_transition_seconds", "ready_settle_seconds"):
            if not np.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"{key} must be positive and finite")
    limits = np.asarray(config["gripper_range"])
    opened = np.asarray(config.get("gripper_open_value", limits[:, 0]), dtype=float)
    closed = np.asarray(config.get("gripper_closed_value", limits[:, 1]), dtype=float)
    if (opened.shape != (2,) or closed.shape != (2,) or not np.isfinite([opened, closed]).all()
            or np.any(opened == closed) or np.any(np.minimum(opened, closed) < limits[:, 0])
            or np.any(np.maximum(opened, closed) > limits[:, 1])):
        raise ValueError("gripper open/closed values must be distinct endpoints inside each range")
    execution_contract(config)
    return config


async def run(config, sonic_root, max_seconds=None):
    contract=execution_contract(config)
    model_only=contract['mode']=='model_only'
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive")
    # Separate MuJoCo data: observation FK and background IK must not share mutable state.
    obs_kin = G1Kinematics(sonic_root, config["reference_frame"], config["tool_offsets"])
    action_kin = G1Kinematics(sonic_root, config["reference_frame"], config["tool_offsets"])
    adapter = ActionAdapter(action_kin, config["model_fps"], config["max_joint_velocity"], config["base_height"],
                            config["gripper_range"], config["base_velocity_deadband"],
                            config.get("ik_position_tolerance", .005), config.get("ik_rotation_tolerance", .05),
                            config.get("truncate_infeasible_tail", False), config.get("gripper_overshoot_tolerance", 0.0),
                            config.get("align_first_ee_frame", False))
    planner_mode = config.get("sonic_mode", "joint") == "planner"
    adapter.planner_mode = planner_mode
    adapter.limit_nominal_velocity=not config.get('waist_tracking_compensation',False)
    planner = PlannerMapper(**config.get("planner", {})) if planner_mode else None
    tracker = None
    if config.get("cartesian_tracking", False):
        if not planner_mode:
            raise ValueError("Cartesian tracking currently requires planner mode")
        from .tracking import CartesianTracker
        tracker = CartesianTracker(sonic_root, config["reference_frame"], config["tool_offsets"])
    grip_range = np.asarray(config["gripper_range"])
    grip_open = np.asarray(config.get("gripper_open_value", grip_range[:, 0]))
    grip_closed = np.asarray(config.get("gripper_closed_value", grip_range[:, 1]))
    context = zmq.Context()
    state_sub = LatestSubscriber(context, config["state_endpoint"], b"g1_debug")
    camera_sub = LatestSubscriber(context, config["camera_endpoint"])
    guard = tactile_sub = None
    base_hold=None
    source = MODEL_ONLY_SOURCE if model_only else SOURCE
    if model_only and config.get('robot_sensor_endpoint'):
        tactile_sub=LatestSubscriber(context,config['robot_sensor_endpoint'],b'grasp_sensors')
    if config.get('grasp_hold',{}).get('enabled',False):
        if not planner_mode:raise ValueError('grasp hold requires planner mode')
        from .grasp_hold import GraspHold
        guard=GraspHold(sonic_root,config)
        tactile_sub=LatestSubscriber(context,config['grasp_hold']['sensor_endpoint'],b'grasp_sensors')
        source='unifolm_vla_tactile_hold'
        if config.get('grasp_lift_reflex'):source='unifolm_vla_grasp_reflex'
    if config.get('base_position_hold',False):
        if guard is None:raise ValueError('base position hold requires the robot odometry sensor stream')
        from .base_hold import BasePositionHold
        base_hold=BasePositionHold(**config.get('base_position_hold_options',{}))
    if not model_only and config.get('compensate_base_motion',False) and (guard is None or not config.get('stationary_single_arm',False)):
        raise ValueError('base-frame transfer requires stationary single-arm execution and robot odometry')
    visual_selector=None
    if config.get('visual_target'):
        if guard is None:raise ValueError('visual action selection requires tactile/robot-state feedback')
        from .vision_target import VisualActionSelector
        visual_selector=VisualActionSelector(sonic_root,config)
        source='unifolm_vla_rgb_selected_reflex' if config.get('grasp_lift_reflex') else 'unifolm_vla_rgb_selected_hold'
    grasp_execution=None
    if config.get('grasp_execution'):
        if visual_selector is None:raise ValueError('explicit grasp execution requires RGB target lock')
        from .grasp_execution import GraspExecution
        grasp_execution=GraspExecution(sonic_root,config)
        source='unifolm_vla_rgb_tactile_execution'
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.setsockopt(zmq.SNDHWM, 2)
    publisher.bind(config["action_bind"])
    def publish(packet):
        # The audit process may request a stop at any time. Its file is an
        # existence-only signal; no task state is read into the executor.
        emergency_stop_check(config)
        publisher.send(packet,zmq.NOBLOCK)
    task = None
    count = {"chunks": 0, "packets": 0, "expired_chunks": 0, "rate_limited_packets": 0}
    count["model_requests"] = 0
    count['max_native_leg_overshoot_rad']=0.
    count.update(max_ik_position_error_m=0.0, max_ik_rotation_error_rad=0.0)
    count.update(truncated_chunks=0, truncated_tail_frames=0)
    count["gripper_clamped_values"] = 0
    count["max_policy_latency_seconds"] = 0.0
    count['max_camera_capture_age_seconds']=0.
    wall_started = time.time()
    epoch = time.monotonic()
    last_inference = float("-inf")
    next_inference = float("-inf")
    reference = None
    last_command = None
    last_nominal = None
    last_start = float("-inf")
    feedback_started = None
    startup_upper_anchor = None
    ready_seed = ready_q = None
    ready_gate=None
    if config.get('ready_gate',{}).get('enabled',False):
        if tactile_sub is None:raise ValueError('ready gate requires robot odometry')
        from .ready import ReadyGate
        ready_gate=ReadyGate(obs_kin,config)
    episode_id = str(time.time_ns())
    completed = False
    guard_done=False
    last_vla_grip=grip_open.copy()
    last_guard_event=0
    stationary=None
    consecutive_ik_rejections=0
    ready_base_reanchored=False
    last_visual_index=None
    visual_acquisition_started=None
    count['rejected_ik_chunks']=0
    control_trace = None
    if config.get("trace_dir") and config.get("trace_control"):
        trace_path = Path(config["trace_dir"])
        trace_path.mkdir(parents=True, exist_ok=True)
        control_trace = (trace_path.parent/"control.jsonl").open("w", buffering=1)
    write_episode(config.get("episode_state_file"), episode_id, "warming", source)
    dt = 1 / config["control_fps"]
    startup_deadline = epoch + config["policy_timeout"] + 5 + config.get("ready_transition_seconds", 0) + config.get("ready_settle_seconds", 0)
    try:
        async with PolicyClient(config["policy_url"], config["policy_timeout"]) as policy:
            LOG.info("Connected to model: %s", policy.metadata)
            while max_seconds is None or time.monotonic() - epoch < max_seconds:
                tick = time.monotonic()
                emergency_stop_check(config)
                if planner_mode and tick - last_start > .5:
                    # SONIC publishes g1_debug only in CONTROL. Bootstrap its
                    # native IDLE planner before waiting for measured feedback.
                    publish(pack_command(start=True))
                    last_start = tick
                state = state_sub.read(tick, config["sensor_max_age"])
                cameras = camera_sub.read(tick, config["sensor_max_age"])
                tactile = tactile_sub.read(tick,config['sensor_max_age']) if tactile_sub else None
                if model_only:tactile=robot_only_feedback(tactile)
                emergency_stop_check(config,tactile)
                if state is None or cameras is None or (tactile_sub is not None and tactile is None):
                    if reference is not None or tick > startup_deadline:
                        raise TimeoutError("robot state or camera stream is missing/stale")
                    await asyncio.sleep(dt)
                    continue
                if feedback_started is None:
                    feedback_started = tick
                    if base_hold:base_hold.command([0,0,0,config['base_height']],tactile)
                if planner_mode and tick - feedback_started < config.get("startup_hold_seconds", 0):
                    base=base_hold.command([0,0,0,config['base_height']],tactile) if base_hold else None
                    if config.get('startup_upper_body_hold',False):
                        # The simulator boots with elevated arms. Let the leg
                        # policy see that same upper-body goal, rather than its
                        # default hanging-arm reference during the landing.
                        packet,startup_upper_anchor=startup_upper_body_packet(
                            planner,obs_kin,state,config,dt,startup_upper_anchor,base)
                        publish(packet)
                        last_command=startup_upper_anchor.copy(),tick
                    else:
                        # Legacy landing keeps the native full-body reference.
                        publish(planner.idle_packet(state, config["base_height"], dt,base))
                    count["packets"] += 1
                    await asyncio.sleep(max(0, dt - (time.monotonic() - tick)))
                    continue
                grip_fraction = measured_grippers(state, config["hand_open"], config["hand_closed"])
                grips = model_gripper_values(grip_fraction, grip_open, grip_closed)
                raw_q=np.asarray(state['body_q'],float)
                count['max_native_leg_overshoot_rad']=max(count['max_native_leg_overshoot_rad'],
                    float(np.max(np.abs(raw_q[:12]-np.clip(raw_q[:12],obs_kin.bounds[:12,0],obs_kin.bounds[:12,1])))))
                q = obs_kin.seed_q(raw_q,planner_legs=planner_mode)
                quat = np.asarray(state["base_quat"], dtype=float)
                if planner_mode and "ready_pose" in config:
                    ready_elapsed = tick - feedback_started - config.get("startup_hold_seconds", 0)
                    duration = config["ready_transition_seconds"]
                    target_base_yaw=(planner.anchor_yaw+planner.facing
                        if planner.anchor_yaw is not None and planner.facing is not None else None)
                    # A completed gate must remain valid while waiting for the
                    # first inference. If stability is lost, keep the existing
                    # robot-only ready controller active. The pending inference
                    # is consumed normally after readiness; stale replies are
                    # rejected by the existing deadline check, never replayed.
                    if (ready_gate is not None and ready_q is not None and count['chunks']==0
                        and config.get('ready_gate',{}).get('recheck_until_first_model',False)):
                        ready_gate.update(q,ready_q,tactile,target_base_yaw=target_base_yaw)
                        if config.get('trace_dir'):
                            (Path(config['trace_dir']).parent/'ready-gate.json').write_text(json.dumps(ready_gate.metrics,indent=2)+'\n')
                    if (ready_elapsed < duration + config["ready_settle_seconds"]
                        or (ready_gate is not None and not ready_gate.completed)):
                        if ready_q is None:
                            ready_seed = q.copy()
                            ready_target_seed = q.copy()
                            if 'ready_arm_seed' in config:
                                ready_target_seed[15:]=config['ready_arm_seed']
                            if "ready_waist" in config:
                                ready_target_seed[12:15] = config["ready_waist"]
                            ready_q = obs_kin.solve([pose_matrix(p) for p in config["ready_pose"]], ready_target_seed)
                        ratio = np.clip(ready_elapsed / duration, 0, 1)
                        blend = ratio * ratio * (3 - 2 * ratio)
                        command = ready_seed + blend * (ready_q - ready_seed)
                        if ready_gate is not None and blend>=1:
                            command=ready_gate.update(q,ready_q,tactile,target_base_yaw=target_base_yaw)
                            if last_command:
                                command=last_command[0]+np.clip(command-last_command[0],
                                    -config['max_joint_velocity']*dt,config['max_joint_velocity']*dt)
                            if config.get('trace_dir'):
                                (Path(config['trace_dir']).parent/'ready-gate.json').write_text(json.dumps(ready_gate.metrics,indent=2)+'\n')
                            if ready_elapsed>config['ready_gate'].get('timeout_seconds',45):
                                raise TimeoutError(f'robot did not reach calibrated ready conditions: {ready_gate.metrics}')
                        ready_grip = np.asarray(config.get("ready_gripper_value", grip_open))
                        fraction = gripper_closed_fraction(ready_grip, grip_open, grip_closed)
                        ready_hand = np.asarray(config["hand_open"]) + fraction[:, None] * (
                            np.asarray(config["hand_closed"]) - config["hand_open"])
                        base=[0,0,0,config['base_height']]
                        if base_hold:base=base_hold.command(base,tactile)
                        ready_dq=np.zeros(29) if last_command is None else (command-last_command[0])/max(dt,tick-last_command[1])
                        packet = planner.packet(command, ready_dq, base,
                                                state, dt, ready_hand)
                        publish(packet)
                        last_command = command, tick
                        count["packets"] += 1
                        await asyncio.sleep(max(0, dt - (time.monotonic() - tick)))
                        continue
                if base_hold and ready_gate and ready_gate.completed and not ready_base_reanchored and config.get('ready_reanchor_base',False):
                    base_hold.reanchor(tactile)
                    planner.velocity[:]=0
                    ready_base_reanchored=True
                if visual_selector and (grasp_execution is None or grasp_execution.phase in ('approach','align')) and cameras.get('index')!=last_visual_index:
                    from .sync_camera import observation_state
                    try:
                        visual_selector.observe(decode_cameras(cameras,config['camera_mapping']),
                            observation_state(cameras,time.monotonic(),config['sensor_max_age']),time.monotonic())
                    except ValueError:
                        if visual_selector.target_world is not None:raise
                        if visual_acquisition_started is None:visual_acquisition_started=tick
                        if tick-visual_acquisition_started>3:raise TimeoutError('initial RGB target acquisition exceeded 3 seconds')
                    last_visual_index=cameras.get('index')
                if grasp_execution and count['chunks']>=config['grasp_execution'].get('min_vla_chunks',1) and grasp_execution.consider(tactile,visual_selector,grips,
                        ready_q[12:15] if ready_q is not None else q[12:15]):
                    if task is not None:
                        task.cancel();await asyncio.gather(task,return_exceptions=True);task=None
                    command,grip,guard_done=grasp_execution.command(tactile)
                    if last_command:
                        step=config['max_joint_velocity']*max(dt,tick-last_command[1])
                        command=last_command[0]+np.clip(command-last_command[0],-step,step)
                    dq=np.zeros(29) if last_command is None else (command-last_command[0])/max(dt,tick-last_command[1])
                    hands=np.asarray(config['hand_open'])+gripper_closed_fraction(grip,grip_open,grip_closed)[:,None]*(
                        np.asarray(config['hand_closed'])-config['hand_open'])
                    publish(planner.packet(command,dq,[0,0,0,config['base_height']],state,dt,hands))
                    count['packets']+=1;last_command=command,tick
                    if config.get('trace_dir'):
                        (Path(config['trace_dir']).parent/'grasp-execution.json').write_text(json.dumps({
                            'phase':grasp_execution.phase,'completed':guard_done,'events':grasp_execution.events,
                            'alignment_status':grasp_execution.alignment_status},indent=2)+'\n')
                    if control_trace and count['packets']%5==0:
                        control_trace.write(json.dumps({'wall_time':time.time(),'phase':'rgb_tactile_'+grasp_execution.phase,
                            'simulation_time':tactile['simulation_time'],'command_q':command.tolist(),
                            'measured_q':q.tolist(),'jaw_force_n':tactile['jaw_force_n'],'command_grippers':grip.tolist(),
                            'alignment_status':grasp_execution.alignment_status})+'\n')
                    if guard_done:break
                    await asyncio.sleep(max(0,dt-(time.monotonic()-tick)));continue
                if guard is not None and guard.holding:
                    grip,guard_done=guard.update(tactile,last_vla_grip,guard.hold_height)
                    if len(guard.events)>last_guard_event:
                        for event in guard.events[last_guard_event:]:LOG.info('Grasp feedback: %s',event)
                        last_guard_event=len(guard.events)
                        if config.get('trace_dir'):
                            (Path(config['trace_dir']).parent/'grasp-hold.json').write_text(json.dumps(guard.events,indent=2))
                    command=guard.hold_q.copy()
                    if last_command:
                        command=last_command[0]+np.clip(command-last_command[0],
                            -config['max_joint_velocity']*dt,config['max_joint_velocity']*dt)
                    dq=np.zeros(29) if last_command is None else (command-last_command[0])/dt
                    hands=np.asarray(config['hand_open'])+gripper_closed_fraction(grip,grip_open,grip_closed)[:,None]*(
                        np.asarray(config['hand_closed'])-config['hand_open'])
                    base=[0,0,0,guard.hold_height]
                    if base_hold:base=base_hold.command(base,tactile)
                    publish(planner.packet(command,dq,base,state,dt,hands))
                    count['packets']+=1
                    last_command=command,tick
                    if control_trace and count['packets']%5==0:
                        control_trace.write(json.dumps({'wall_time':time.time(),'monotonic_time':tick,'phase':guard.phase,
                            'simulation_time':tactile['simulation_time'],'command_q':command.tolist(),
                            'measured_q':q.tolist(),'jaw_force_n':tactile['jaw_force_n'],
                            'command_grippers':grip.tolist()})+'\n')
                    if guard_done:break
                    await asyncio.sleep(max(0,dt-(time.monotonic()-tick)))
                    continue
                if config.get('stationary_single_arm',False) and stationary is None:
                    if not planner_mode:raise ValueError('stationary single-arm execution requires planner mode')
                    from .execution import StationaryArmExecution
                    stationary=StationaryArmExecution(obs_kin,q,grips,config['instruction'],config['base_height'],
                        ready_q[12:15] if ready_q is not None else None)
                    if guard:guard.waist_reference=stationary.anchor[12:15].copy()
                if reference is None:
                    initial=last_command[0].copy() if ready_q is not None and last_command else q
                    reference = Reference(np.array([tick, tick + dt]), np.stack([initial, initial]), quat,
                                          np.stack([grips, grips]))
                    reference.base_commands = np.array([[0, 0, 0, config["base_height"]]] * 2)
                if task is not None and task.done():
                    try:
                        candidate = task.result()
                    except ValueError as exc:
                        if not config.get('max_ik_rejections',0) or not str(exc).startswith('IK target unreachable'):
                            raise
                        consecutive_ik_rejections+=1
                        count['rejected_ik_chunks']+=1
                        if consecutive_ik_rejections>config['max_ik_rejections']:raise
                        LOG.warning('rejecting unreachable chunk; hold last safe reference and reobserve: %s',exc)
                        candidate=None
                    if candidate is None:
                        pass
                    elif candidate.times[-1] <= tick or candidate.policy_latency > config.get("max_action_latency", 2.0):
                        # Reject expired references and replies beyond the
                        # explicit observation-to-action latency budget.
                        count["expired_chunks"] += 1
                        LOG.warning("discarding expired/over-budget action chunk (%.3fs inference+IK)", candidate.policy_latency)
                    else:
                        consecutive_ik_rejections=0
                        reference = candidate
                        if config.get("execute_full_chunk", False):
                            next_inference = float(candidate.times[-1])
                        count["chunks"] += 1
                        count["max_ik_position_error_m"] = max(count["max_ik_position_error_m"], candidate.max_ik_position_error)
                        count["max_ik_rotation_error_rad"] = max(count["max_ik_rotation_error_rad"], candidate.max_ik_rotation_error)
                        count["gripper_clamped_values"] += candidate.gripper_clamped_values
                        count["max_policy_latency_seconds"] = max(count["max_policy_latency_seconds"], candidate.policy_latency)
                        if candidate.truncated_tail_frames:
                            count["truncated_chunks"] += 1
                            count["truncated_tail_frames"] += candidate.truncated_tail_frames
                            LOG.warning("using reachable prefix; dropped %s infeasible future frames", candidate.truncated_tail_frames)
                    task = None
                settled = not planner_mode or tick - feedback_started >= config.get("startup_hold_seconds", 0)
                if (settled and (visual_selector is None or visual_selector.target_world is not None)
                        and task is None and tick >= next_inference
                        and tick - last_inference >= config["inference_interval"]):
                    if last_inference == float("-inf"):
                        write_episode(config.get("episode_state_file"), episode_id, "running", source)
                    images = decode_cameras(cameras, config["camera_mapping"])
                    policy_state,policy_grips=state,grips
                    observation_base=((robot_base_transform(tactile) if model_only else guard.base_transform(tactile))
                                      if config.get('compensate_base_motion',False) else None)
                    if config.get('synchronized_observation',False):
                        from .sync_camera import observation_state
                        policy_state=observation_state(cameras,time.monotonic(),config['sensor_max_age'])
                        policy_grips=model_gripper_values(measured_grippers(policy_state,config['hand_open'],config['hand_closed']),grip_open,grip_closed)
                        count['max_camera_capture_age_seconds']=max(count['max_camera_capture_age_seconds'],time.monotonic()-cameras['capture_monotonic'])
                        if observation_base is not None:
                            observation_base=robot_base_transform(policy_state) if model_only else guard.base_transform(policy_state)
                    if visual_selector and guard.latched_grip is None:
                        visual_selector.observe(images,policy_state,time.monotonic())
                        last_visual_index=cameras.get('index')
                    obs = make_observation(obs_kin, policy_state, images, policy_grips, config["instruction"], config.get("unnorm_key"))
                    for option in ("encoder_generation_prompt", "inference_seed"):
                        if option in config:
                            obs[option] = config[option]
                    required = [k for k in policy.metadata.get("data_keys", []) if k.startswith("observation.")]
                    if any(k not in obs for k in required):
                        raise ValueError("client cannot supply all server observation keys")
                    count["model_requests"] += 1
                    async def infer(obs=obs, q=q.copy(), quat=quat.copy(), grips=grips.copy(), started=tick,
                                    policy_state=policy_state,cameras=cameras,
                                    robot_base=observation_base,
                                    request_index=count["model_requests"]):
                        selection=None
                        if visual_selector and guard.latched_grip is None:
                            proposals=await policy.request('get_action_candidates',obs=obs,count=visual_selector.count)
                            action,selection=visual_selector.select(proposals['candidates'],policy_state,time.monotonic())
                        else:action = await policy.request("get_action", obs=obs)
                        oriented=visual_selector.constrain_orientation(action,policy_state) if visual_selector else action
                        executed_action=stationary.constrain(oriented) if stationary else oriented
                        transfer_q, transfer_quat, transfer_grips = q, quat, grips
                        if config.get("align_first_ee_frame", False):
                            latest = state_sub.read(time.monotonic(), config["sensor_max_age"])
                            if latest is None:
                                raise TimeoutError("no fresh state for cross-controller trajectory transfer")
                            transfer_q = obs_kin.seed_q(latest["body_q"],planner_legs=planner_mode)
                            transfer_quat = np.asarray(latest["base_quat"])
                            fraction = measured_grippers(latest, config["hand_open"], config["hand_closed"])
                            transfer_grips = model_gripper_values(fraction, grip_open, grip_closed)
                        if config.get("trace_dir"):
                            trace = Path(config["trace_dir"])
                            trace.mkdir(parents=True, exist_ok=True)
                            extra_trace={f'executed_{key}':value for key,value in executed_action.items()} if stationary else {}
                            if robot_base is not None:extra_trace['observed_robot_base']=robot_base
                            if selection is not None:
                                (trace/f'selection-{request_index:04d}.json').write_text(json.dumps(selection,indent=2)+'\n')
                                np.savez_compressed(trace/f'candidates-{request_index:04d}.npz',
                                    **{f'{i}.{k}':v for i,c in enumerate(proposals['candidates']) for k,v in c.items()})
                            np.savez_compressed(trace / f"reply-{time.monotonic_ns()}.npz", **action,
                                                **extra_trace,
                                                measured_q=q, measured_quat=quat,
                                                measured_grippers=grips, transfer_seed_q=transfer_q,
                                                request_time=started, reply_wall_time=time.time(), request_index=request_index,
                                                obs_left_ee=obs["observation.state.left_ee_6d"],
                                                obs_right_ee=obs["observation.state.right_ee_6d"],
                                                obs_lower_body=obs["observation.state.lower_body"],
                                                obs_body_q=np.asarray(policy_state['body_q']),
                                                obs_grippers=np.r_[obs['observation.state.left_gripper'],obs['observation.state.right_gripper']],
                                                camera_capture_monotonic=cameras.get('capture_monotonic',np.nan),
                                                camera_capture_simulation_time=cameras.get('capture_simulation_time',np.nan))
                            if config.get("trace_observations") and (request_index - 1) % config.get("trace_observation_interval", 10) == 0:
                                import cv2
                                extension = "png" if config.get("trace_lossless_observations", False) else "jpg"
                                for name in config["camera_mapping"]:
                                    cv2.imwrite(str(trace / f"obs-{request_index:04d}-{name}.{extension}"), obs[f"observation.images.{name}"])
                        converted = await asyncio.to_thread(adapter.convert, executed_action, transfer_q, transfer_quat, transfer_grips, started)
                        if stationary:converted=stationary.freeze_unused_joints(converted)
                        converted.observation_base_pose=robot_base
                        # Preserve predicted poses/grippers; only dilate their
                        # timeline when wall time outruns the physics simulator.
                        converted.times = started + (converted.times-started)*config.get("action_time_scale", 1.0)
                        converted.policy_latency = time.monotonic() - started
                        if config.get("action_timebase", "observation") == "receipt":
                            # Absolute learned goals execute after the reply is
                            # available, with seam rate limits and an explicit
                            # observation-to-action latency budget. The trace
                            # retains the original observation/request time.
                            converted.times += converted.policy_latency
                        return converted
                    task = asyncio.create_task(infer())
                    last_inference = tick
                initial_warmup = count["chunks"] == 0 and tick <= startup_deadline
                if not initial_warmup and tick - reference.times[-1] > config["reference_max_age"]:
                    raise TimeoutError("no fresh reference; stopping publication")
                packet, command, grip = reference.packet(tick, epoch, config["control_fps"], config["lookahead_frames"],
                    last_nominal if stationary and config.get('waist_tracking_compensation',False) else last_command,
                    None if config.get('waist_tracking_compensation',False) else config["max_joint_velocity"])
                nominal_command=command.copy()
                last_nominal=nominal_command,tick
                if guard is not None:
                    if visual_selector:
                        # Refresh target from the latest same-frame camera state,
                        # including while a model proposal batch is in flight.
                        if not guard.holding and guard.latched_grip is None and cameras.get('index')!=last_visual_index:
                            from .sync_camera import observation_state
                            visual_selector.observe(decode_cameras(cameras,config['camera_mapping']),
                                observation_state(cameras,time.monotonic(),config['sensor_max_age']),time.monotonic())
                            last_visual_index=cameras.get('index')
                        grip=visual_selector.gate_grip(tactile,grip,grip_open,guard.latched_grip is not None,time.monotonic())
                    last_vla_grip=grip.copy()
                    height=float(np.interp(tick,reference.times,reference.base_commands[:,3]))
                    if grasp_execution is None:grip,guard_done=guard.update(tactile,grip,height)
                    if len(guard.events)>last_guard_event:
                        for event in guard.events[last_guard_event:]:LOG.info('Grasp feedback: %s',event)
                        last_guard_event=len(guard.events)
                        if config.get('trace_dir'):
                            (Path(config['trace_dir']).parent/'grasp-hold.json').write_text(json.dumps(guard.events,indent=2))
                    if guard.holding:command=guard.hold_q.copy()
                if stationary and count['chunks']>0 and config.get('waist_tracking_compensation',False) and not (guard and guard.holding):
                    command=stationary.compensate_waist(command,q,reference.observation_base_pose,
                        guard.base_transform(tactile) if reference.observation_base_pose is not None else None)
                    if last_command:
                        step=config['max_joint_velocity']*max(dt,tick-last_command[1])
                        command=last_command[0]+np.clip(command-last_command[0],-step,step)
                elif model_only and count['chunks']>0 and config.get('waist_tracking_compensation',False):
                    command=compensate_model_targets(obs_kin,command,q,reference.observation_base_pose,
                        robot_base_transform(tactile) if reference.observation_base_pose is not None else None)
                    if last_command:
                        step=config['max_joint_velocity']*max(dt,tick-last_command[1])
                        command=last_command[0]+np.clip(command-last_command[0],-step,step)
                count["rate_limited_packets"] += int(reference.packet_limited)
                grip_fraction = gripper_closed_fraction(grip, grip_open, grip_closed)
                if planner_mode:
                    if tracker:
                        command = tracker.update(command, q, dt)
                        if last_command:
                            step = config["max_joint_velocity"]*max(dt, tick-last_command[1])
                            command = last_command[0]+np.clip(command-last_command[0], -step, step)
                    if last_command:
                        step=config['max_joint_velocity']*max(dt,tick-last_command[1])
                        limited=last_command[0]+np.clip(command-last_command[0],-step,step)
                        if not reference.packet_limited and np.any(np.abs(limited-command)>1e-8):
                            count['rate_limited_packets']+=1
                        command=limited
                    base = np.array([np.interp(tick, reference.times, reference.base_commands[:, j]) for j in range(4)])
                    if tick > reference.times[-1]:
                        base[:3] = 0  # never extend a stale walking command
                    if guard is not None and guard.holding:base=np.array([0,0,0,guard.hold_height])
                    if base_hold:base=base_hold.command(base,tactile)
                    dq = np.zeros(29) if last_command is None else (command - last_command[0]) / max(dt, tick - last_command[1])
                    hand_q = np.asarray(config["hand_open"]) + grip_fraction[:, None] * (
                        np.asarray(config["hand_closed"]) - config["hand_open"])
                    packet = planner.packet(command, dq, base, state, dt, hand_q)
                else:
                    packet = attach_sim_hands(packet, grip_fraction, config["hand_open"], config["hand_closed"])
                publish(packet)
                if control_trace and count["packets"] % 5 == 0 and (not model_only or count['chunks']>0):
                    actual_ee = obs_kin.forward(q)
                    commanded_ee = obs_kin.forward(command)
                    model_targets=obs_kin.forward(nominal_command) if model_only else None
                    if model_targets is not None and reference.observation_base_pose is not None:
                        transfer=np.linalg.inv(robot_base_transform(tactile))@reference.observation_base_pose
                        model_targets=[transfer@target for target in model_targets]
                    control_trace.write(json.dumps({"wall_time": time.time(), "monotonic_time": tick,
                        "measured_q": q.tolist(), "command_q": command.tolist(),
                        "nominal_q":nominal_command.tolist(),
                        "measured_ee": [p.tolist() for p in actual_ee],
                        "command_ee": [p.tolist() for p in commanded_ee],
                        "model_target_ee": [p.tolist() for p in model_targets] if model_targets is not None else None,
                        "measured_grippers": grips.tolist(), "command_grippers": grip.tolist(),
                        "simulation_time":tactile.get('simulation_time') if tactile is not None else None,
                        "model_only":model_only,
                        "accepted_model_chunks":count['chunks'],
                        "command_base":base.tolist() if planner_mode else None,
                        "planner_velocity":(getattr(planner,'last_command',{}).get('sent_velocity_local_mps',planner.velocity.tolist())
                                            if planner_mode else None),
                        "planner_command":getattr(planner,'last_command',None) if planner_mode else None,
                        "phase":'tactile_latched' if guard is not None and guard.latched_grip is not None else 'vla',
                        "reference_end": float(reference.times[-1])})+"\n")
                last_command = command, tick
                count["packets"] += 1
                await asyncio.sleep(max(0, dt - (time.monotonic() - tick)))
            if count["chunks"] == 0:
                raise RuntimeError("no model action chunk was completed")
            if guard is not None and not guard_done:
                raise RuntimeError('VLA did not establish and maintain the sensor-confirmed lifted grasp')
            count.update(wall_started=wall_started, wall_finished=time.time(),
                         elapsed_seconds=time.monotonic() - epoch,
                         policy_source=source,
                         execution_mode=contract['mode'],execution_contract=contract,
                         planner_modification_counts=dict(getattr(planner,'modification_counts',{})) if planner else None,
                         stationary_single_arm=bool(stationary),
                         base_motion_compensation=config.get('compensate_base_motion',False),
                         ready_base_reanchored=ready_base_reanchored,
                         ready_conditions=ready_gate.metrics if ready_gate else None,
                         waist_compensation_rejections=stationary.waist_rejections if stationary else 0,
                         visual_selection={'color':visual_selector.color,'selected_proposals':visual_selector.selected_proposals,
                            'blocked_close_packets':visual_selector.blocked_close_packets,'last_distance_m':visual_selector.last_distance} if visual_selector else None,
                         grasp_execution={'completed':grasp_execution.done,'events':grasp_execution.events} if grasp_execution else None,
                         base_position_hold={'max_error_m':base_hold.max_error,'anchor_xy':base_hold.anchor.tolist()} if base_hold else None,
                         grasp_hold={'completed':guard_done and grasp_execution is None,'events':guard.events,'lift_reflex':bool(guard.reflex),
                            'force_target_n':guard.force_target,'max_gripper_correction':guard.max_force_correction} if guard else None,
                         cartesian_tracking=bool(tracker),
                         max_tracking_error_m=tracker.max_error if tracker else None,
                         tracking_ik_rejections=tracker.rejections if tracker else 0,
                         action_time_scale=config.get("action_time_scale", 1.0),
                         arm_tracking=config.get("arm_tracking", "sonic"),
                         sonic_mode=config.get("sonic_mode", "joint"),
                         align_first_ee_frame=config.get("align_first_ee_frame", False),
                         action_timebase=config.get("action_timebase", "observation"))
            LOG.info("Simulation client finished: %s", count)
            completed = True
            return count
    except Exception as error:
        # Preserve provenance and progress for rejected/aborted trials too.
        # Missing a successful task must never erase how its actions arose.
        error.client_report=dict(count,wall_started=wall_started,wall_finished=time.time(),
            elapsed_seconds=time.monotonic()-epoch,policy_source=source,
            execution_mode=contract['mode'],execution_contract=contract,
            planner_modification_counts=dict(getattr(planner,'modification_counts',{})) if planner else None,
            arm_tracking=config.get('arm_tracking','sonic'),sonic_mode=config.get('sonic_mode','joint'))
        raise
    finally:
        if visual_selector and config.get('trace_dir'):
            (Path(config['trace_dir']).parent/'visual-target-state.json').write_text(json.dumps({
                'color':visual_selector.color,'selected_proposals':visual_selector.selected_proposals,
                'blocked_close_packets':visual_selector.blocked_close_packets,'last_distance_m':visual_selector.last_distance,
                'last_rgb_detection':visual_selector.last_detection},indent=2)+'\n')
        write_episode(config.get("episode_state_file"), episode_id, "passed" if completed else "failed", source)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if planner_mode:
            # ZMQ manager supports a real control-stop command. Repeat to allow
            # its subscriber to consume the final packet before socket close.
            for _ in range(3):
                publisher.send(pack_command(stop=True), zmq.NOBLOCK)
                await asyncio.sleep(.05)
        state_sub.close()
        camera_sub.close()
        if tactile_sub:tactile_sub.close()
        publisher.close()
        context.term()
        if control_trace:
            control_trace.close()
        # Plain SONIC --input-type zmq has keyboard control, not the manager's
        # command topic. Do not publish a misleading remote-stop command.
        LOG.warning("Reference publication stopped. %s", "SONIC manager stop sent." if planner_mode
                    else "SONIC remains operator-controlled; press O in its terminal.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("sim.yaml"))
    parser.add_argument("--simulation", action="store_true", required=True,
                        help="Use only with SONIC deploy.sh ... sim; no real-robot mode is implemented")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    def report(value):
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(value, indent=2) + "\n")
    report({"status": "running"})
    contract=None
    try:
        config = load_config(args.config)
        contract=execution_contract(config)
        if args.report:
            args.report.with_suffix(".config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        result = asyncio.run(run(config, args.sonic_root, args.max_seconds))
        report(dict(status="passed", **result))
    except KeyboardInterrupt:
        report({"status": "interrupted","execution_contract":contract,
                "execution_mode":contract['mode'] if contract else None})
        raise SystemExit(130)
    except Exception as exc:
        report(dict(getattr(exc,'client_report',{}),status="failed",error=str(exc),
                    execution_contract=contract,execution_mode=contract['mode'] if contract else None))
        raise


if __name__ == "__main__":
    main()
