"""Validate robot-only reset/ready conditions before starting frozen VLA."""
import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import pose_matrix


class ReadyGate:
    def __init__(self,kin,config):
        self.kin=kin
        self.targets=[pose_matrix(p) for p in config['ready_pose']]
        self.options=config.get('ready_gate',{})
        self.yaw_tolerance=self.options.get('yaw_tolerance_rad')
        self.angular_speed_tolerance=self.options.get('angular_speed_tolerance_rps')
        for name,value in (('yaw_tolerance_rad',self.yaw_tolerance),
                           ('angular_speed_tolerance_rps',self.angular_speed_tolerance)):
            if value is not None and (not np.isfinite(value) or value<=0):
                raise ValueError(f'{name} must be positive and finite')
        self.check_orientation=self.yaw_tolerance is not None or self.angular_speed_tolerance is not None
        self.consistent_posture='ready_arm_seed' in config
        self.since=None
        self.previous=None
        self.speed=float('inf')
        self.angular_speed=float('inf')
        self.completed=False
        self.metrics={}

    def update(self,q,target_q,sensor,*,target_base_yaw=None):
        now=float(sensor['simulation_time'])
        position=np.asarray(sensor['base_position'],float)
        origin=np.asarray(sensor['initial_base_position'],float)
        if (not np.isfinite(now) or position.shape!=(3,) or origin.shape!=(3,)
                or not np.isfinite(np.r_[position,origin]).all()):
            raise ValueError('ready gate requires finite robot time and base positions')
        rotation=None
        base_yaw=yaw_error=None
        if self.check_orientation:
            quat=np.asarray(sensor.get('base_quat'),float)
            if (quat.shape!=(4,) or not np.isfinite(quat).all()
                    or not np.isclose(np.linalg.norm(quat),1.,atol=1e-3)):
                raise ValueError('ready orientation gate requires a valid base_quat')
            rotation=Rotation.from_quat(quat[[1,2,3,0]])
            matrix=rotation.as_matrix()
            if np.hypot(matrix[0,0],matrix[1,0])<1e-6:
                raise ValueError('ready gate cannot determine yaw from vertical base heading')
            base_yaw=float(np.arctan2(matrix[1,0],matrix[0,0]))
            if self.yaw_tolerance is not None:
                if target_base_yaw is None or np.ndim(target_base_yaw)!=0 or not np.isfinite(target_base_yaw):
                    raise ValueError('ready yaw gate requires finite target_base_yaw')
                target_base_yaw=float(target_base_yaw)
                yaw_error=float(abs(np.arctan2(np.sin(base_yaw-target_base_yaw),np.cos(base_yaw-target_base_yaw))))
        clock_restarted=self.previous is not None and now<self.previous[0]
        if clock_restarted:
            self.since=None
            self.completed=False
            self.speed=self.angular_speed=float('inf')
            self.previous=None
        if self.previous is not None and now>self.previous[0]:
            self.speed=float(np.linalg.norm(position[:2]-self.previous[1][:2])/(now-self.previous[0]))
            if rotation is not None:
                self.angular_speed=float((self.previous[2].inv()*rotation).magnitude()/(now-self.previous[0]))
        elif self.previous is not None and now==self.previous[0]:
            # Repeating an unchanged fresh snapshot is normal. Changed pose at
            # zero elapsed simulation time cannot establish measured stability.
            if not np.allclose(position,self.previous[1],rtol=0,atol=1e-10):
                self.speed=float('inf')
            if rotation is not None and (self.previous[2].inv()*rotation).magnitude()>1e-10:
                self.angular_speed=float('inf')
        speed=self.speed
        if self.previous is None or now>self.previous[0]:self.previous=now,position.copy(),rotation
        actual=self.kin.forward(q)
        pos_error=max(float(np.linalg.norm(a[:3,3]-t[:3,3])) for a,t in zip(actual,self.targets))
        rot_error=max(float(Rotation.from_matrix(t[:3,:3].T@a[:3,:3]).magnitude()) for a,t in zip(actual,self.targets))
        waist_error=float(np.max(np.abs(q[12:15]-target_q[12:15])))
        base_error=float(np.linalg.norm(position[:2]-origin[:2]))
        try:
            seed=q.copy()
            if self.consistent_posture:seed[15:]=target_q[15:]
            command=self.kin.solve(self.targets,seed,position_tolerance=.01,rotation_tolerance=.05)
        except ValueError as exc:
            if not str(exc).startswith('IK target unreachable'):raise
            command=target_q.copy()
        command[:15]=target_q[:15]
        valid=(pos_error<self.options.get('position_tolerance_m',.012)
               and rot_error<self.options.get('rotation_tolerance_rad',.08)
               and waist_error<self.options.get('waist_tolerance_rad',.05)
               and base_error<self.options.get('base_tolerance_m',.05)
               and speed<self.options.get('speed_tolerance_mps',.08))
        if self.yaw_tolerance is not None:valid=valid and yaw_error<=self.yaw_tolerance
        if self.angular_speed_tolerance is not None:valid=valid and self.angular_speed<self.angular_speed_tolerance
        self.since=(now if self.since is None else self.since) if valid else None
        self.completed=bool(self.since is not None and now-self.since>=self.options.get('stable_sim_seconds',.5))
        self.metrics={'simulation_time':now,'ready':self.completed,'eef_position_error_m':pos_error,
                      'eef_rotation_error_rad':rot_error,'waist_error_rad':waist_error,
                      'base_error_m':base_error,'base_speed_mps':speed if np.isfinite(speed) else None,
                      'base_yaw_rad':base_yaw,'target_base_yaw_rad':target_base_yaw if self.yaw_tolerance is not None else None,
                      'base_yaw_error_rad':yaw_error,
                      'base_angular_speed_rps':self.angular_speed if self.check_orientation and np.isfinite(self.angular_speed) else None,
                      'yaw_tolerance_rad':self.yaw_tolerance,'angular_speed_tolerance_rps':self.angular_speed_tolerance,
                      'speed_tolerance_mps':self.options.get('speed_tolerance_mps',.08),
                      'stable_sim_seconds':self.options.get('stable_sim_seconds',.5),
                      'clock_restarted':clock_restarted}
        return command
