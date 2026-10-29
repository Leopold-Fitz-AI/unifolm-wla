"""Simulation diagnostic: raw planner arm references -> PD + robot gravity.

SONIC drives legs, and optionally waist. This is an explicit alternative upper-body execution
path, not the SONIC neural arm policy and not a task/vision controller.
"""
import json
import time

import numpy as np
import zmq

from .protocol import HEADER_SIZE, UPPER_MUJOCO_IN_ISAAC


def planner_fields(raw):
    if not raw.startswith(b'planner'):
        raise ValueError('expected planner topic')
    start = len(b'planner')
    header = json.loads(raw[start:start+HEADER_SIZE].rstrip(b'\0'))
    if header['v'] != 1 or header['endian'] != 'le':
        raise ValueError('unsupported planner protocol')
    cursor, fields = start+HEADER_SIZE, {}
    for field in header['fields']:
        dtype = {'f32':'<f4', 'i32':'<i4', 'u8':'u1'}[field['dtype']]
        count = int(np.prod(field['shape']))
        value = np.frombuffer(raw, dtype=dtype, count=count, offset=cursor).reshape(field['shape'])
        cursor += value.nbytes
        fields[field['name']] = value
    if cursor != len(raw):
        raise ValueError('truncated/trailing planner payload')
    return fields


class ArmReferenceServo:
    def __init__(self, env, endpoint, waist=True, startup_arm_q=None):
        self.env = env
        self.reference = None
        self.reference_velocity=np.zeros(29)
        self.received_at = float('-inf')
        self.first_joint=12 if waist else 15
        self.startup_reference=None
        self.startup_log_next=0.
        if startup_arm_q is not None:
            # Scene initialization only. Later motion remains torque driven;
            # neither reset nor a stale client may teleport the robot here.
            import mujoco
            arms=np.asarray(startup_arm_q,float)
            model,data=env.mj_model,env.mj_data
            if data.time!=0 or arms.shape!=(14,) or not np.isfinite(arms).all():
                raise ValueError('startup arms require finite 14 joints at simulation time zero')
            joints=np.asarray(env.body_joint_index[15:29])
            limits=model.jnt_range[joints]
            if np.any(arms<limits[:,0]) or np.any(arms>limits[:,1]):
                raise ValueError('startup arms exceed robot joint limits')
            data.qpos[model.jnt_qposadr[joints]]=arms
            data.qvel[model.jnt_dofadr[joints]]=0
            mujoco.mj_forward(model,data)
            self.startup_reference=np.zeros(29)
            # The SDK publishes zero gains before startup. Holding only arms
            # leaves the unpowered waist free to fold, dropping both hands
            # despite perfect arm-joint tracking. Keep the initial waist pose
            # until the first client upper-body reference, then hand it back
            # to SONIC in arms_pd mode.
            self.startup_reference[12:15]=data.qpos[model.jnt_qposadr[np.asarray(env.body_joint_index[12:15])]]
            self.startup_reference[15:]=arms
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'planner')
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)

    def torques(self, native):
        try:
            fields = planner_fields(self.socket.recv(zmq.NOBLOCK))
        except zmq.Again:
            pass
        else:
            if 'upper_body_position' in fields:
                upper = fields['upper_body_position']
                if upper.shape != (17,) or not np.isfinite(upper).all():
                    raise ValueError('invalid arm reference')
                q = np.zeros(29)
                q[UPPER_MUJOCO_IN_ISAAC] = upper
                velocity=fields.get('upper_body_velocity',np.zeros(17))
                if velocity.shape!=(17,) or not np.isfinite(velocity).all():
                    raise ValueError('invalid arm reference velocity')
                self.reference_velocity=np.zeros(29)
                self.reference_velocity[UPPER_MUJOCO_IN_ISAAC]=velocity
                self.reference, self.received_at = q, time.monotonic()
        startup=self.reference is None and getattr(self,'startup_reference',None) is not None
        if not startup and (self.reference is None or time.monotonic()-self.received_at > .3):
            return native
        model, data = self.env.mj_model, self.env.mj_data
        result = native.copy()
        for i in range(12 if startup else self.first_joint, 29):
            joint = self.env.body_joint_index[i]
            qa, va = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
            if startup:
                # Before the SDK controller starts, low_cmd gains may be zero.
                # Use its configured native gains, still subject to the same
                # upstream motor effort clipping in DefaultEnv.sim_step.
                kp,kd=self.env.robot.MOTOR_KP[i],self.env.robot.MOTOR_KD[i]
                target=self.startup_reference[i]
            else:
                motor = self.env.unitree_bridge.low_cmd.motor_cmd[i]
                kp,kd,target=motor.kp,motor.kd,self.reference[i]
            # Use native motor gains and upstream effort clipping. qfrc_bias
            # contains robot gravity/Coriolis, not object/contact support forces.
            result[i] = (kp*(target-data.qpos[qa])
                         +kd*(self.reference_velocity[i]-data.qvel[va])+data.qfrc_bias[va])
        if getattr(self,'startup_reference',None) is not None and data.time<=1. and data.time>=self.startup_log_next:
            self.startup_log_next=float(data.time)+.1
            arms=np.asarray(self.env.body_joint_index[15:29])
            print('UNIFOLM_STARTUP_ARMS '+json.dumps({'simulation_time':float(data.time),'startup':startup,
                'waist_reference':self.startup_reference[12:15].tolist(),
                'waist_measured':data.qpos[model.jnt_qposadr[np.asarray(self.env.body_joint_index[12:15])]].tolist(),
                'reference':(self.startup_reference if startup else self.reference)[15:].tolist(),
                'measured':data.qpos[model.jnt_qposadr[arms]].tolist(),'torque':result[15:].tolist()}),flush=True)
        return result

    def close(self):
        self.socket.close()
        self.context.term()
