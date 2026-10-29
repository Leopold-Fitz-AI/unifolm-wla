"""Station keeping through SONIC velocity commands and robot odometry only."""
import numpy as np
from scipy.spatial.transform import Rotation


class BasePositionHold:
    def __init__(self,gain=1.5,max_speed=.12,engage_m=.02,release_m=.008,min_speed=.04):
        self.gain,self.max_speed=gain,max_speed
        self.engage,self.release,self.min_speed=engage_m,release_m,min_speed
        self.anchor=None
        self.active=False
        self.max_error=0.

    def reanchor(self,sensor):
        self.anchor=np.asarray(sensor['base_position'],float)[:2].copy()
        self.active=False

    def command(self,base,sensor):
        position=np.asarray(sensor['base_position'],float)
        quat=np.asarray(sensor['base_quat'],float)
        if position.shape!=(3,) or not np.isfinite(position).all():raise ValueError('invalid robot odometry')
        if self.anchor is None:self.anchor=np.asarray(sensor.get('initial_base_position',position),float)[:2].copy()
        error=self.anchor-position[:2]
        distance=float(np.linalg.norm(error))
        self.max_error=max(self.max_error,distance)
        if distance>self.engage:self.active=True
        elif distance<self.release:self.active=False
        result=np.asarray(base,float).copy()
        result[:2]=0
        if self.active and distance>0:
            velocity=error/distance*np.clip(self.gain*distance,self.min_speed,self.max_speed)
            yaw=Rotation.from_quat(quat[[1,2,3,0]]).as_euler('xyz')[2]
            rotation=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
            result[:2]=rotation@velocity
        return result
