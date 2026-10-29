"""Simulated jaw force/proprioception sensors, without object identities/poses."""
import msgpack
import mujoco
import numpy as np
import zmq

from .kinematics import JOINT_NAMES


def jaw_forces(model, data):
    sensors = {}
    for hand, side in enumerate(('left','right')):
        for finger in (0,1):
            for part in ('finger','pad'):
                name = f'{side}_grasp_{part}_{finger}'
                geom = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
                if geom >= 0: sensors[geom] = (hand,finger)
    forces = np.zeros((2,2))
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = [sensors.get(int(g)) for g in (contact.geom1,contact.geom2)]
        # Calibrate out contact between the gripper's own two empty jaws.
        if all(p is not None for p in pair) and pair[0][0] == pair[1][0]:
            continue
        if not any(p is not None for p in pair):continue
        mujoco.mj_contactForce(model,data,index,force)
        for sensor in pair:
            if sensor is not None:forces[sensor] += max(0.,force[0])
    return forces


class TactilePublisher:
    def __init__(self,env,port):
        self.env=env
        self.context=zmq.Context()
        self.socket=self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER,0)
        self.socket.setsockopt(zmq.SNDHWM,1)
        self.socket.bind(f'tcp://127.0.0.1:{port}')
        m=env.mj_model
        self.qadr=[m.jnt_qposadr[m.joint(n).id] for n in JOINT_NAMES]
        self.hand_adr=[[m.jnt_qposadr[m.joint(f'{side}_hand_thumb_{i}_joint').id] for i in (0,1)]
                       for side in ('left','right')]
        self.last=-1.
        self.index=0
        self.initial_position=env.mj_data.qpos[:3].copy()

    def publish(self):
        m,d=self.env.mj_model,self.env.mj_data
        now=float(d.time)
        if now >= self.last and now-self.last < .02:return
        self.last=now
        packet={'index':self.index,'simulation_time':now,'jaw_force_n':jaw_forces(m,d).tolist(),
                'jaw_q':d.qpos[self.hand_adr].tolist(),'body_q':d.qpos[self.qadr].tolist(),
                'base_position':d.xpos[m.body('pelvis').id].tolist(),
                'initial_base_position':self.initial_position.tolist(),
                'base_quat':d.xquat[m.body('pelvis').id].tolist()}
        self.index+=1
        self.socket.send(b'grasp_sensors'+msgpack.packb(packet,use_bin_type=True),zmq.NOBLOCK)

    def close(self):
        self.socket.close()
        self.context.term()
