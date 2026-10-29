"""Explicit RGB-aligned tactile close/lift execution after a VLA approach.

This controller does not consume simulator object/evaluator state. A VLA
proposal first approaches a target triangulated from RGB. An optional bounded
visual residual then aligns the hand. It freezes the achieved pose, verifies jaw
load, and raises that robot pose vertically before holding. All assist phases
are separately labelled; this is not an unassisted original-action-head trial.
"""
import numpy as np
import time
from collections import deque
from scipy.spatial.transform import Rotation

from .kinematics import G1Kinematics
from .vision_target import VisualActionSelector


class GraspExecution:
    def __init__(self,sonic_root,config):
        self.kin=G1Kinematics(sonic_root,config['reference_frame'],config['grasp_hold_tool_offsets'])
        self.options=config['grasp_execution'];self.config=config
        self.phase='approach';self.active=False;self.done=False;self.events=[]
        self.align_since=None;self.contact_since=None;self.lost_since=None;self.reached_since=None
        self.lift=float(self.options.get('lift_m',.14));self.speed=float(self.options.get('speed_mps',.04))
        if not .08<=self.lift<=.18 or not .01<=self.speed<=.06:raise ValueError('invalid explicit grasp-execution lift bounds')
        self.close_seconds=float(self.options.get('close_seconds',.8));self.hold_seconds=float(self.options.get('hold_seconds',6.))
        if self.close_seconds<=0 or self.hold_seconds<2:raise ValueError('invalid close/hold duration')
        self.closed=np.asarray(config['gripper_closed_value'],float)
        self.last_time=None
        self.selector=None
        # Optional precision gate: position alone does not mean the hand has
        # settled, nor that its jaws have reached the requested orientation.
        self.settle=self.options.get('alignment_settle',{})
        self.pose_history=deque()
        self.descent_target=None
        self.alignment_status=None
        self.bias_settle_only=self.options.get('tracking_bias_settle_only',False)
        if not isinstance(self.bias_settle_only,bool):
            raise ValueError('tracking_bias_settle_only must be a boolean')
        if self.settle:
            bounds={'position_tolerance_m':(.001,.01), 'rotation_tolerance_rad':(.01,.15),
                    'speed_mps':(.001,.03), 'window_seconds':(.1,1.),
                    'stable_seconds':(.1,1.)}
            if set(self.settle)!=set(bounds):raise ValueError('alignment_settle requires position, rotation, speed, window and stable bounds')
            for key,(low,high) in bounds.items():
                value=float(self.settle[key])
                if not np.isfinite(value) or not low<=value<=high:
                    raise ValueError(f'invalid alignment_settle {key}')

    def consider(self,sensor,selector,grips,waist_reference):
        if self.active:return True
        now=float(sensor['simulation_time'])
        if selector.target_world is None or time.monotonic()-selector.last_seen>selector.max_age:
            self.align_since=None;return False
        self.hand=selector.hand
        q=self.kin.seed_q(sensor['body_q'],planner_legs=True)
        world=selector.base(sensor)@self.kin.forward(q)[self.hand]
        local=world[:3,:3].T@(selector.target_world-world[:3,3])
        vertical=world[2,3]-selector.target_world[2]
        aligned=(np.all(np.abs(local)<[.018,.014,.018]) and -.005<vertical<.018)
        residual=float(self.options.get('visual_residual_m',0))
        if not 0<=residual<=.12:raise ValueError('visual residual must be in [0,.12] metres')
        if residual and np.linalg.norm(selector.target_world-world[:3,3])<residual:
            self.active=True;self.phase='align';self.started=now;self.world_pose=world.copy()
            yaw=np.arctan2(world[1,0],world[0,0])
            pitch=self.config['visual_target'].get('horizontal_jaw_pitch_degrees',20)
            self.world_pose[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()@Rotation.from_euler('y',pitch,degrees=True).as_matrix()
            self.anchor=q.copy();self.anchor[12:15]=waist_reference
            self.grips=np.asarray(grips,float).copy()
            self.grips[self.hand]=float(np.asarray(self.config['gripper_open_value'])[self.hand])
            self.selector=selector;self.residual_anchor=world[:3,3].copy()
            self.align_stage='raise' if self.options.get('clearance_approach',False) else 'descend'
            if self.align_stage=='descend' and self.options.get('lock_descent_target',False):
                self.descent_target=np.asarray(selector.target_world,float).copy()
            self.tracking_bias=np.zeros(3)
            self.hover_z=max(world[2,3],selector.target_world[2]+.06)
            self.events.append({'phase':'align','simulation_time':now,'max_visual_residual_m':residual,
                                'initial_rgb_error_m':(selector.target_world-world[:3,3]).tolist()})
            return True
        self.align_since=(now if self.align_since is None else self.align_since) if aligned else None
        if self.align_since is None or now-self.align_since<self.options.get('align_seconds',.1):return False
        self.active=True;self.phase='close';self.started=now;self.world_pose=world.copy();self.grasp_from=world.copy()
        self.anchor=q.copy();self.anchor[12:15]=waist_reference
        self.grips=np.asarray(grips,float).copy();self.close_from=self.grips.copy()
        self.events.append({'phase':'close','simulation_time':now,'rgb_alignment_error_local_m':local.tolist(),
                            'hand_world_position':world[:3,3].tolist()})
        return True

    def command(self,sensor):
        now=float(sensor['simulation_time'])
        if self.last_time is not None and now<self.last_time:raise RuntimeError('execution sensor clock reset')
        dt=.02 if self.last_time is None else min(.05,max(0,now-self.last_time))
        self.last_time=now
        q=self.kin.seed_q(sensor['body_q'],planner_legs=True)
        forces=np.asarray(sensor['jaw_force_n'],float)[self.hand]
        if forces.shape!=(2,) or not np.isfinite(forces).all():raise ValueError('invalid execution jaw forces')
        contact=bool(np.all(forces>=self.options.get('min_jaw_force_n',.5)))
        if self.phase=='align':
            selector=self.selector
            if time.monotonic()-selector.last_seen>selector.max_age:raise RuntimeError('RGB target stale during visual alignment')
            target=np.asarray(selector.target_world,float)
            if self.descent_target is not None:
                if np.linalg.norm(target-self.descent_target)>.015:
                    raise RuntimeError('RGB target moved after descent lock')
                target=self.descent_target
            goal=target+np.array([0,0,self.options.get('alignment_height_m',.008)])
            if np.linalg.norm(goal-self.residual_anchor)>float(self.options['visual_residual_m'])+.015:
                raise RuntimeError('visual residual exceeded bounded approach workspace')
            actual=VisualActionSelector.base(sensor)@self.kin.forward(q)[self.hand]
            if self.align_stage=='raise':
                goal=np.r_[self.residual_anchor[:2],self.hover_z]
                if actual[2,3]>=self.hover_z-.008:
                    self.align_stage='hover';self.tracking_bias[:]=0
                    self.events.append({'phase':'align_hover','simulation_time':now})
            elif self.align_stage=='hover':
                goal=np.r_[selector.target_world[:2],self.hover_z]
                if np.linalg.norm(actual[:2,3]-goal[:2])<.006 and abs(actual[2,3]-goal[2])<.01:
                    self.align_stage='descend';self.tracking_bias[:]=0
                    self.events.append({'phase':'align_descend','simulation_time':now})
                    if self.options.get('lock_descent_target',False):
                        # Use the unobstructed hover estimate during the final
                        # descent, instead of chasing finger-occluded boxes.
                        # Live RGB freshness and target-motion guards still run.
                        self.descent_target=target.copy()
                        self.events[-1]['rgb_target_world']=target.tolist()
            bias_limit=float(self.options.get('tracking_bias_limit_m',0))
            if not 0<=bias_limit<=.02:raise ValueError('tracking bias must be in [0,.02] metres')
            # During a rate-limited move, goal - actual includes normal travel
            # lag. Integrating it produces a bias in the direction of motion
            # and makes the hand overshoot, then wait for that bias to unwind.
            # Compare against the *corrected* endpoint so an established bias
            # does not itself disable subsequent steady-state compensation.
            bias_ready=(np.linalg.norm(self.world_pose[:3,3]-(goal+self.tracking_bias))<.002
                        if self.bias_settle_only else np.linalg.norm(self.world_pose[:3,3]-goal)<.025)
            if bias_ready:
                self.tracking_bias+=.5*dt*(goal-actual[:3,3])
                self.tracking_bias*=min(1,bias_limit/max(np.linalg.norm(self.tracking_bias),1e-12))
            error=goal+self.tracking_bias-self.world_pose[:3,3];length=np.linalg.norm(error)
            self.world_pose[:3,3]+=error*min(1,.02*dt/max(length,1e-12))
            local=actual[:3,:3].T@(target-actual[:3,3])
            vertical=actual[2,3]-target[2]
            aligned=self.align_stage=='descend' and np.all(np.abs(local)<[.010,.009,.015]) and 0<vertical<.018
            if self.settle:
                if not self.pose_history or now>self.pose_history[-1][0]:
                    self.pose_history.append((now,actual[:3,3].copy()))
                window=self.settle['window_seconds']
                while len(self.pose_history)>1 and now-self.pose_history[1][0]>=window:
                    self.pose_history.popleft()
                span=now-self.pose_history[0][0]
                # Path length catches oscillation that endpoint displacement
                # would incorrectly classify as a stationary hand.
                points=np.asarray([p for _,p in self.pose_history])
                speed=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()/max(span,1e-9))
                rotation_error=float(Rotation.from_matrix(self.world_pose[:3,:3].T@actual[:3,:3]).magnitude())
                goal_position=target+np.array([0,0,self.options.get('alignment_height_m',.008)])
                aligned=bool(aligned and span>=window and speed<=self.settle['speed_mps']
                             and np.max(np.abs(actual[:3,3]-goal_position))<=self.settle['position_tolerance_m']
                             and rotation_error<=self.settle['rotation_tolerance_rad'])
                self.alignment_status={'goal_error_m':(goal_position-actual[:3,3]).tolist(),
                    'speed_mps':speed,'rotation_error_rad':rotation_error,'stage':self.align_stage,
                    'rgb_age_seconds':time.monotonic()-selector.last_seen,'aligned':aligned,
                    'tracking_bias_m':self.tracking_bias.tolist(),'bias_integrating':bool(bias_ready)}
            self.align_since=(now if self.align_since is None else self.align_since) if aligned else None
            if self.align_since is not None and now-self.align_since>=self.settle.get('stable_seconds',.2):
                self.phase='close';self.started=now;self.world_pose=actual.copy();self.close_from=self.grips.copy()
                self.events.append({'phase':'close','simulation_time':now,'rgb_alignment_error_local_m':local.tolist(),
                                    'visual_residual_applied_m':(actual[:3,3]-self.residual_anchor).tolist(),
                                    'robot_tracking_bias_m':self.tracking_bias.tolist(),
                                    'hand_world_position':actual[:3,3].tolist()})
                if self.settle:self.events[-1].update(measured_speed_mps=speed,rotation_error_rad=rotation_error)
            elif now-self.started>self.options.get('alignment_timeout_seconds',12):raise RuntimeError('bounded visual alignment did not converge')
        elif self.phase=='close':
            fraction=np.clip((now-self.started)/self.close_seconds,0,1)
            self.grips[self.hand]=self.close_from[self.hand]+fraction*(self.closed[self.hand]-self.close_from[self.hand])
            self.contact_since=(now if self.contact_since is None else self.contact_since) if contact else None
            if fraction>=1 and self.contact_since is not None and now-self.contact_since>=.25:
                # Capture the actually achieved hand pose after loaded closure,
                # rather than assuming that the requested pose was reached.
                self.grasp_from=VisualActionSelector.base(sensor)@self.kin.forward(q)[self.hand]
                self.world_pose=self.grasp_from.copy();self.phase='lift';self.lift_started=now
                self.duration=max(.5,1.5*self.lift/self.speed)
                self.events.append({'phase':'lift','simulation_time':now,'assist_lift_m':self.lift,
                                    'jaw_forces_n':forces.tolist()})
            elif now-self.started>4:raise RuntimeError('visual-aligned closure did not establish bilateral jaw load')
        else:
            self.lost_since=(now if self.lost_since is None else self.lost_since) if not contact else None
            if self.lost_since is not None and now-self.lost_since>.2:raise RuntimeError('bilateral support lost during explicit grasp execution')
            if self.phase=='lift':
                ratio=np.clip((now-self.lift_started)/self.duration,0,1);blend=ratio*ratio*(3-2*ratio)
                self.world_pose=self.grasp_from.copy();self.world_pose[2,3]+=blend*self.lift
                actual=VisualActionSelector.base(sensor)@self.kin.forward(q)[self.hand]
                reached=ratio>=1 and np.linalg.norm(actual[:3,3]-self.world_pose[:3,3])<.015
                self.reached_since=(now if self.reached_since is None else self.reached_since) if reached else None
                if self.reached_since is not None and now-self.reached_since>=.2:
                    self.phase='hold';self.hold_started=now
                    self.events.append({'phase':'hold','simulation_time':now,'measured_hand_lift_m':float(actual[2,3]-self.grasp_from[2,3])})
                elif now-self.lift_started>self.duration+5:raise RuntimeError('explicit lift failed to reach measured robot target')
            elif self.phase=='hold' and now-self.hold_started>=self.hold_seconds and contact:
                self.done=True
        targets=self.kin.forward(q)
        targets[self.hand]=np.linalg.inv(VisualActionSelector.base(sensor))@self.world_pose
        result=self.kin.solve(targets,q,position_tolerance=.01,rotation_tolerance=.05)
        result[12:15]=self.anchor[12:15]
        start=15+7*(1-self.hand);result[start:start+7]=self.anchor[start:start+7]
        return result,self.grips.copy(),self.done
