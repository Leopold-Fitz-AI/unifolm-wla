"""Native SONIC locomotion planner + 17-DOF upper-body command adapter.

Planner owns the 12 leg joints. UnifoLM owns waist, arm EE goals, velocity,
yaw rate and height. This explicitly replaces the model's predicted leg angles
with SONIC's dynamically generated locomotion reference.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .protocol import UPPER_MUJOCO_IN_ISAAC, finite_array, pack_fields


def yaw(quat_wxyz):
    q = finite_array(quat_wxyz, (4,), "quaternion")
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-3):
        raise ValueError("quaternion must be normalized")
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")[2]


class PlannerMapper:
    def __init__(self, max_speed=.4, max_yaw_rate=.4, max_acceleration=.5, height_range=(.5, .8),
                 direction_resolution=0.,speed_resolution=0.,stationary_mode=0,
                 velocity_deadband_mps=.02,yaw_deadband_rps=.03,
                 walking_speed_threshold_mps=.03,turning_rate_threshold_rps=.03,
                 minimum_walking_speed_mps=0.,heading_tracking_gain=0.,
                 max_heading_correction_rad=.15,heading_correction_time_constant=.3):
        # Official mode 0 is IDLE and does not enforce requested squat height.
        # Mode 4 is IDEL_SQUAT; use explicitly for a stationary height target.
        if stationary_mode not in (0,4):raise ValueError('stationary_mode must be official IDLE=0 or IDEL_SQUAT=4')
        self.stationary_mode=stationary_mode
        self.max_speed, self.max_yaw_rate = max_speed, max_yaw_rate
        self.max_acceleration = max_acceleration
        self.height_range = height_range
        self.direction_resolution,self.speed_resolution=direction_resolution,speed_resolution
        thresholds=np.asarray([velocity_deadband_mps,yaw_deadband_rps,
                               walking_speed_threshold_mps,turning_rate_threshold_rps],float)
        if not np.isfinite(thresholds).all() or np.any(thresholds<0) or np.any(thresholds[2:]<=0):
            raise ValueError('planner deadbands must be nonnegative and motion thresholds positive')
        self.velocity_deadband,self.yaw_deadband=thresholds[:2]
        self.walking_threshold,self.turning_threshold=thresholds[2:]
        # SONIC's enum documents SLOW_WALK=1 at 0.1--0.8 m/s; its joystick
        # uses >=0.2 m/s. ZMQ passes smaller speeds straight to the network.
        # Keep zero available for reproducing old experiments, and configure
        # an explicit supported minimum rather than amplifying tiny commands.
        if (not np.isfinite(minimum_walking_speed_mps) or minimum_walking_speed_mps<0
            or minimum_walking_speed_mps>max_speed):
            raise ValueError('minimum_walking_speed_mps must be finite and within [0,max_speed]')
        self.minimum_walking_speed=minimum_walking_speed_mps
        heading_options=np.asarray([heading_tracking_gain,max_heading_correction_rad,
                                    heading_correction_time_constant],float)
        if (not np.isfinite(heading_options).all() or np.any(heading_options[:2]<0)
            or max_heading_correction_rad>np.pi or heading_correction_time_constant<=0):
            raise ValueError('heading tracking requires nonnegative gain, correction in [0,pi], and positive time constant')
        self.heading_tracking_gain=heading_tracking_gain
        self.max_heading_correction=max_heading_correction_rad
        self.heading_correction_time_constant=heading_correction_time_constant
        self.heading_correction=0.
        self.modification_counts={name:0 for name in ('velocity_deadband','yaw_deadband',
            'velocity_clipped','yaw_clipped','acceleration_limited','height_clipped','below_motion_threshold',
            'minimum_walking_speed_suppressed','turning_in_static_mode','zero_direction_suppressed','mode_switches',
            'heading_correction_packets','heading_correction_saturated','heading_rate_limited')}
        self.facing = None
        self.sent_facing = None
        self.velocity = np.zeros(2)
        self.anchor_yaw = None
        self.last_command = None

    def fields(self, q, dq, base, state, dt, hand_q):
        if not np.isfinite(dt) or dt<=0:raise ValueError('planner dt must be finite and positive')
        base = finite_array(base, (4,), "base_command")
        if self.anchor_yaw is None:
            self.anchor_yaw = yaw(state.get("init_base_quat", state["base_quat"]))
        heading = yaw(state["base_quat"]) - self.anchor_yaw
        if self.facing is None:
            self.facing = heading
            self.sent_facing=heading
        previous_facing=self.sent_facing
        yaw_rate = np.clip(base[2], -self.max_yaw_rate, self.max_yaw_rate)
        self.modification_counts['yaw_clipped']+=int(yaw_rate!=base[2])
        if abs(yaw_rate) < self.yaw_deadband:
            self.modification_counts['yaw_deadband']+=int(yaw_rate!=0)
            yaw_rate = 0.0
        self.facing += yaw_rate * dt
        # SONIC facing is in its initial-heading frame. The model's integrated
        # target remains untouched; only the emitted reference gets bounded
        # robot-orientation feedback. No position or task observations enter.
        heading_error=float(np.arctan2(np.sin(self.facing-heading),np.cos(self.facing-heading)))
        correction_target=float(np.clip(self.heading_tracking_gain*heading_error,
                                        -self.max_heading_correction,self.max_heading_correction))
        self.modification_counts['heading_correction_saturated']+=int(
            abs(self.heading_tracking_gain*heading_error)>self.max_heading_correction)
        if self.heading_tracking_gain:
            alpha=-np.expm1(-dt/self.heading_correction_time_constant)
            self.heading_correction+=float(alpha)*(correction_target-self.heading_correction)
        else:
            self.heading_correction=0.
        desired_sent_facing=self.facing+self.heading_correction
        # Both states remain unwrapped across +/-pi. Preserve the original yaw
        # speed limit on the combined feedforward + feedback reference, without
        # moving the model target to hide controller lag.
        change=desired_sent_facing-previous_facing
        step=self.max_yaw_rate*dt
        heading_rate_limited=bool(abs(change)>step+1e-12)
        self.sent_facing=(previous_facing+float(np.clip(change,-step,step))
                          if heading_rate_limited else desired_sent_facing)
        sent_facing=self.sent_facing
        sent_yaw_rate=float(yaw_rate) if not self.heading_tracking_gain else (sent_facing-previous_facing)/dt
        actual_correction=sent_facing-self.facing
        self.modification_counts['heading_correction_packets']+=int(abs(actual_correction)>1e-12)
        self.modification_counts['heading_rate_limited']+=int(heading_rate_limited)
        desired = base[:2].copy()  # checkpoint contract: local vx/vy in m/s
        if np.linalg.norm(desired) < self.velocity_deadband:
            self.modification_counts['velocity_deadband']+=int(np.any(desired!=0))
            desired[:] = 0
        norm = np.linalg.norm(desired)
        if norm > self.max_speed:
            desired *= self.max_speed / norm
            self.modification_counts['velocity_clipped']+=1
        delta = desired - self.velocity
        norm = np.linalg.norm(delta)
        if norm > self.max_acceleration * dt:
            delta *= self.max_acceleration * dt / norm
            self.modification_counts['acceleration_limited']+=1
        self.velocity += delta
        speed = float(np.linalg.norm(self.velocity))
        filtered_speed=speed
        # Planner coordinates are relative to its initial robot heading.
        rotation = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
        direction = rotation @ (self.velocity / speed) if speed > 1e-12 else np.zeros(2)
        # SONIC replans on exact float direction/speed changes. Odometry noise
        # must not restart a slow walking plan at every planner tick.
        if self.direction_resolution:direction=np.round(direction/self.direction_resolution)*self.direction_resolution
        if self.speed_resolution:speed=min(self.max_speed,round(speed/self.speed_resolution)*self.speed_resolution)
        # Gate before AND after quantization: never round an unsupported small
        # speed up into a gait. Pure turning updates facing in the static mode;
        # it does not ask the walking network to realize zero translation.
        minimum=max(self.walking_threshold,self.minimum_walking_speed)
        walking=(filtered_speed+1e-12>=self.minimum_walking_speed and speed+1e-12>=minimum)
        suppression=None
        if not walking and filtered_speed>0:
            suppression=('below_minimum_walking_speed' if filtered_speed+1e-12<self.minimum_walking_speed
                         else 'below_motion_threshold')
            key='minimum_walking_speed_suppressed' if suppression=='below_minimum_walking_speed' else suppression
            self.modification_counts[key]+=1
        direction_norm=float(np.linalg.norm(direction))
        if walking and direction_norm<=1e-12:
            walking=False;suppression='zero_direction'
            self.modification_counts['zero_direction_suppressed']+=1
        mode=1 if walking else self.stationary_mode
        if not walking:
            direction=np.zeros(2);speed=0.
            if abs(yaw_rate)>=self.turning_threshold:self.modification_counts['turning_in_static_mode']+=1
        sent_planner_velocity=direction/direction_norm*speed if walking else np.zeros(2)
        if self.last_command is not None and self.last_command['mode']!=mode:
            self.modification_counts['mode_switches']+=1
        height=np.clip(base[3], *self.height_range)
        self.modification_counts['height_clipped']+=int(height!=base[3])
        # Distinguish the filtered request from the actual packet. The upstream
        # ZMQ manager normalizes movement and substitutes speed=-1 in static
        # modes; zero here denotes the absence of a translation request.
        self.last_command={
            'requested_base_command':base.tolist(),
            'limited_velocity_local_mps':desired.tolist(),
            'filtered_velocity_local_mps':self.velocity.tolist(),
            'sent_velocity_local_mps':(rotation.T@sent_planner_velocity).tolist(),
            'sent_velocity_planner_mps':sent_planner_velocity.tolist(),
            'sent_speed_mps':float(speed),'model_yaw_rate_rps':float(yaw_rate),
            'sent_yaw_rate_rps':float(sent_yaw_rate),
            'facing_angle_rad':float(self.facing),'height_m':float(height),'mode':mode,
            'requested_facing_angle_rad':float(self.facing),'sent_facing_angle_rad':float(sent_facing),
            'measured_heading_rad':float(heading),'heading_error_rad':heading_error,
            'heading_correction_target_rad':correction_target,
            'filtered_heading_correction_rad':float(self.heading_correction),
            'heading_correction_rad':float(actual_correction),
            'desired_sent_facing_angle_rad':float(desired_sent_facing),
            'heading_rate_limited':heading_rate_limited,
            'anchor_yaw_rad':float(self.anchor_yaw),
            'requested_world_yaw_rad':float(self.anchor_yaw+self.facing),
            'sent_world_yaw_rad':float(self.anchor_yaw+sent_facing),
            'minimum_walking_speed_mps':float(self.minimum_walking_speed),
            'suppression_reason':suppression,
        }
        return {
            "mode": np.array([mode], np.int32),
            "movement": np.array([*direction, 0], np.float32),
            "facing": np.array([np.cos(sent_facing), np.sin(sent_facing), 0], np.float32),
            "speed": np.array([speed], np.float32),
            "height": np.array([height], np.float32),
            "upper_body_position": finite_array(q, (29,), "q")[UPPER_MUJOCO_IN_ISAAC].astype(np.float32),
            "upper_body_velocity": finite_array(dq, (29,), "dq")[UPPER_MUJOCO_IN_ISAAC].astype(np.float32),
            "left_hand_joints": finite_array(hand_q[0], (7,), "left hand").astype(np.float32),
            "right_hand_joints": finite_array(hand_q[1], (7,), "right hand").astype(np.float32),
        }

    def packet(self, *args, **kwargs):
        return pack_fields("planner", self.fields(*args, **kwargs))

    def idle_packet(self, state, height, dt, base=None):
        """Let SONIC own the complete landing/stand-up motion during warmup."""
        fields = self.fields(np.zeros(29), np.zeros(29), [0, 0, 0, height] if base is None else base, state, dt, np.zeros((2, 7)))
        return pack_fields("planner", {key: fields[key] for key in ("mode", "movement", "facing", "speed", "height")})
