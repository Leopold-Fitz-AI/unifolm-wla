"""Simulation-only parallel-jaw driver; position in metres, force in newtons.

The same planner hand fields carry joint positions. This driver supplies the
linear actuator gains/effort limit appropriate to sliding fingers. It neither
reads objects nor produces grasp/lift targets.
"""
import numpy as np


class ParallelGripperServo:
    def __init__(self, env, kp=250., kd=3., force_limit=8.,closed_limit=.045):
        if not np.isfinite(kp) or not 100<=kp<=500 or not 0<force_limit<=8:
            raise ValueError('gripper gains require kp in [100,500] and force limit at most 8 N')
        self.env, self.kp, self.kd, self.force_limit = env, kp, kd, force_limit
        self.closed_limit=closed_limit
        m = env.mj_model
        self.joints = [[m.joint(f'{side}_hand_thumb_{i}_joint').id for i in (0,1)]
                       for side in ('left','right')]
        for indices in (env.left_hand_index, env.right_hand_index):
            env.torque_limit[indices[:2]-1] = force_limit

    def forces(self, native):
        bridge = self.env.unitree_bridge
        if bridge is None or not bridge.low_cmd:return native
        m, d = self.env.mj_model, self.env.mj_data
        result = native.copy()
        for side,cmd in enumerate((bridge.left_hand_cmd,bridge.right_hand_cmd)):
            for finger,joint in enumerate(self.joints[side]):
                desired = np.clip(cmd.motor_cmd[finger].q,0.,self.closed_limit)
                force = self.kp*(desired-d.qpos[m.jnt_qposadr[joint]])-self.kd*d.qvel[m.jnt_dofadr[joint]]
                result[7*side+finger] = np.clip(force,-self.force_limit,self.force_limit)
        return result
