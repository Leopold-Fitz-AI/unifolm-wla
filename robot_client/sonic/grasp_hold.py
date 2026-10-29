"""Maintain a VLA-established grasp using tactile/proprioceptive feedback.

No object localization or evaluator state is consumed. The VLA must close and
raise its hand before the pose hold engages. The explicitly labelled optional
lift-completion reflex can add a bounded vertical robot goal after that trigger.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import G1Kinematics
from .transport import gripper_closed_fraction


class GraspHold:
    def __init__(self,sonic_root,config):
        opts=config['grasp_hold']
        command=config['instruction'].casefold()
        left='left hand' in command or '左手' in command
        right='right hand' in command or '右手' in command
        if left == right:raise ValueError('grasp hold requires an explicit left/right hand instruction')
        self.hand=0 if left else 1
        self.kin=G1Kinematics(sonic_root,config['reference_frame'],
            config.get('grasp_hold_tool_offsets',config['tool_offsets']))
        self.opened=np.asarray(config['gripper_open_value'],float)
        self.closed=np.asarray(config['gripper_closed_value'],float)
        self.jaw_open=np.asarray(config['hand_open'],float)[:,:2]
        self.jaw_closed=np.asarray(config['hand_closed'],float)[:,:2]
        self.min_force=opts.get('min_jaw_force_n',.05)
        self.debounce=opts.get('contact_seconds',.08)
        self.lift=opts.get('hand_lift_m',.12)
        self.hold_seconds=opts.get('hold_sim_seconds',6.)
        self.loss_grace=opts.get('loss_grace_seconds',.2)
        self.world_hold=config.get('grasp_hold_world_frame',False)
        self.hold_position_tolerance=config.get('grasp_hold_position_tolerance_m',.005)
        if not np.isfinite(self.hold_position_tolerance) or not 0<self.hold_position_tolerance<=.02:
            raise ValueError('hold position tolerance must be in (0,.02] metres')
        force=config.get('grasp_hold_force',{})
        self.force_target=force.get('target_n',0.)
        self.force_gain=force.get('gain',3.)
        self.force_rate=force.get('max_rate',3.)
        self.force_limit=force.get('max_correction',1.5)
        if any(not np.isfinite(v) or v<0 for v in (self.force_target,self.force_gain,self.force_rate,self.force_limit)):
            raise ValueError('grasp force parameters must be finite and nonnegative')
        self.force_correction=0.
        self.max_force_correction=0.
        self.waist_reference=None
        self.reflex=config.get('grasp_lift_reflex',{})
        if self.reflex:
            trigger=self.reflex.get('trigger_lift_m',.06)
            target=self.reflex.get('target_lift_m',.12)
            extra=self.reflex.get('max_extra_lift_m',.06)
            speed=self.reflex.get('speed_mps',.04)
            if (not self.world_hold or not np.isfinite([trigger,target,extra,speed]).all()
                or trigger<=0 or target<trigger or not 0<extra<=.06 or target-trigger>extra+1e-9 or not 0<speed<=.1):
                raise ValueError('lift reflex requires world odometry, a VLA lift trigger, <=6 cm extra lift and positive bounded speed')
        self.reflex_started=None
        self.reflex_reached_since=None
        self.reflex_finished=False
        self.contact_since=None
        self.lost_since=None
        self.grasp_height=None
        self.latched_grip=None
        self.hold_q=None
        self.hold_started=None
        self.last_time=None
        self.events=[]

    @property
    def holding(self):return self.hold_q is not None

    @property
    def phase(self):
        return 'tactile_lift_reflex' if self.reflex_started is not None and not self.reflex_finished else 'tactile_hold'

    @staticmethod
    def base_transform(sensor):
        transform=np.eye(4)
        transform[:3,:3]=Rotation.from_quat(np.asarray(sensor['base_quat'])[[1,2,3,0]]).as_matrix()
        transform[:3,3]=np.asarray(sensor['base_position'],float)
        if not np.isfinite(transform).all():raise ValueError('invalid robot odometry')
        return transform

    def hold_command(self,sensor):
        """Compensate robot base motion around the measured VLA-established pose."""
        q=self.kin.seed_q(sensor['body_q'],planner_legs=True)
        targets=self.kin.forward(q)
        targets[self.hand]=np.linalg.inv(self.base_transform(sensor))@self.world_grasp_pose
        result=self.kin.solve(targets,q,position_tolerance=self.hold_position_tolerance,rotation_tolerance=.05)
        # The arm IK uses measured waist angles to reject actual base/waist
        # disturbances. The waist/unused arm keep their captured references.
        result[12:15]=self.hold_anchor[12:15]
        start=15+7*(1-self.hand)
        result[start:start+7]=self.hold_anchor[start:start+7]
        return result

    def update(self,sensor,command_grips,base_height):
        now=float(sensor['simulation_time'])
        if self.last_time is not None and now < self.last_time:raise RuntimeError('grasp sensor clock reset')
        dt=0. if self.last_time is None else min(.1,now-self.last_time)
        self.last_time=now
        forces=np.asarray(sensor['jaw_force_n'],float)
        jaw_q=np.asarray(sensor['jaw_q'],float)
        if forces.shape != (2,2) or jaw_q.shape != (2,2) or not np.isfinite([forces,jaw_q]).all():
            raise ValueError('invalid tactile data')
        fraction=gripper_closed_fraction(command_grips,self.opened,self.closed)[self.hand]
        jaw_fraction=(jaw_q-self.jaw_open)/(self.jaw_closed-self.jaw_open)
        contact=(np.all(forces[self.hand]>=self.min_force)
                 and np.max(jaw_fraction[self.hand])<.58/.65
                 and np.min(jaw_fraction[self.hand])>.03/.65)
        q=self.kin.seed_q(sensor['body_q'],planner_legs=True)
        pose=self.kin.forward(q)[self.hand]
        rotation=Rotation.from_quat(np.asarray(sensor['base_quat'])[[1,2,3,0]]).as_matrix()
        height=float((rotation@pose[:3,3])[2])
        if self.world_hold:height=float((self.base_transform(sensor)@pose)[2,3])
        if contact and (fraction>=.3 or self.latched_grip is not None):
            self.lost_since=None
            self.contact_since=now if self.contact_since is None else self.contact_since
            if self.latched_grip is None and now-self.contact_since>=self.debounce:
                self.latched_grip=float(command_grips[self.hand])
                self.grasp_height=height
                self.events.append({'event':'contact_confirmed','simulation_time':now,'robot_hand_height':height})
            if self.latched_grip is not None:
                # Preserve the strongest close request actually made by VLA.
                close_value=float(command_grips[self.hand])
                self.latched_grip=min(self.latched_grip,close_value) if self.closed[self.hand]<self.opened[self.hand] else max(self.latched_grip,close_value)
                trigger=self.reflex.get('trigger_lift_m',self.lift)
                if not self.holding and height-self.grasp_height>=trigger:
                    self.hold_q=q.copy()
                    self.hold_anchor=q.copy()
                    if self.waist_reference is not None:
                        self.hold_anchor[12:15]=self.waist_reference
                        self.hold_q[12:15]=self.waist_reference
                    if self.world_hold:self.world_grasp_pose=self.base_transform(sensor)@pose
                    self.hold_height=float(base_height)
                    if self.reflex:
                        if not self.world_hold:raise ValueError('lift reflex requires world-frame proprioception')
                        target=self.grasp_height+self.reflex.get('target_lift_m',.12)
                        delta=max(0.,target-height)
                        if delta>self.reflex.get('max_extra_lift_m',.06)+1e-6:
                            raise RuntimeError('requested grasp lift completion exceeds bounded assist')
                        self.reflex_started=now
                        self.reflex_from=self.world_grasp_pose.copy()
                        self.reflex_target=self.world_grasp_pose.copy()
                        self.reflex_target[2,3]+=delta
                        self.reflex_duration=max(.5,1.5*delta/self.reflex.get('speed_mps',.04))
                        self.hold_started=None
                        self.events.append({'event':'lift_reflex','simulation_time':now,
                            'vla_hand_lift_m':height-self.grasp_height,'assisted_lift_m':delta})
                    else:
                        self.hold_started=now
                        self.events.append({'event':'pose_hold','simulation_time':now,'robot_hand_lift_m':height-self.grasp_height})
        else:
            self.contact_since=None
            self.lost_since=now if self.lost_since is None else self.lost_since
            if now-self.lost_since>self.loss_grace:
                if self.holding:raise RuntimeError('tactile support lost during grasp hold')
                if self.latched_grip is not None:self.events.append({'event':'contact_lost','simulation_time':now})
                self.latched_grip=None
                self.grasp_height=None
                self.force_correction=0.
        result=np.asarray(command_grips,float).copy()
        if self.latched_grip is not None:
            if self.force_target:
                error=max(0.,self.force_target-float(np.min(forces[self.hand])))
                self.force_correction=min(self.force_limit,self.force_correction+min(self.force_rate,self.force_gain*error)*dt)
                self.max_force_correction=max(self.max_force_correction,self.force_correction)
            direction=np.sign(self.closed[self.hand]-self.opened[self.hand])
            result[self.hand]=np.clip(self.latched_grip+direction*self.force_correction,
                min(self.closed[self.hand],self.opened[self.hand]),max(self.closed[self.hand],self.opened[self.hand]))
        if self.reflex_started is not None and not self.reflex_finished:
            ratio=np.clip((now-self.reflex_started)/self.reflex_duration,0,1)
            blend=ratio*ratio*(3-2*ratio)
            self.world_grasp_pose=self.reflex_from.copy()
            self.world_grasp_pose[:3,3]+=blend*(self.reflex_target[:3,3]-self.reflex_from[:3,3])
            measured_world=self.base_transform(sensor)@pose
            reached=(ratio>=1 and np.linalg.norm(measured_world[:3,3]-self.reflex_target[:3,3])<=.015)
            self.reflex_reached_since=(now if self.reflex_reached_since is None else self.reflex_reached_since) if reached else None
            if self.reflex_reached_since is not None and now-self.reflex_reached_since>=.2:
                self.reflex_finished=True
                self.hold_started=now
                self.events.append({'event':'reflex_complete','simulation_time':now,
                    'measured_hand_lift_m':height-self.grasp_height})
            if now-self.reflex_started>self.reflex_duration+5 and not self.reflex_finished:
                raise RuntimeError('grasp lift completion did not reach its robot pose target')
        if self.holding and self.world_hold:self.hold_q=self.hold_command(sensor)
        done=bool(self.holding and self.hold_started is not None and now-self.hold_started>=self.hold_seconds and contact)
        return result,done
