"""Atomic rendered RGB + measured robot state; no evaluator/object state.

One snapshot is captured while physics is paused for rendering. A bounded worker
encodes that immutable snapshot and publishes once per actual render, retaining
capture time rather than making old images appear fresh at publication time.
"""
import queue
import threading
import time

import cv2
import msgpack
import numpy as np
import zmq

from .kinematics import JOINT_NAMES


class SynchronizedCameraPublisher:
    def __init__(self,env,port):
        self.env=env
        m=env.mj_model
        self.body=m.body('pelvis').id
        self.qadr=[m.jnt_qposadr[m.joint(n).id] for n in JOINT_NAMES]
        self.hand_adr=[]
        for side in ('left','right'):
            joints=[i for i in range(m.njnt) if (m.joint(i).name or '').startswith(f'{side}_hand_')]
            if len(joints)!=7:raise ValueError('synchronized observation requires seven SDK joints per hand')
            self.hand_adr.append(m.jnt_qposadr[joints])
        self.pending=queue.Queue(maxsize=1)
        self.ready=threading.Event()
        self.stop=threading.Event()
        self.error=None
        self.index=0
        self.thread=threading.Thread(target=self._worker,args=(port,),daemon=True)
        self.thread.start()
        if not self.ready.wait(3):raise TimeoutError('camera publisher did not initialize')
        if self.error:raise RuntimeError('camera publisher initialization failed') from self.error

    def capture(self,frames):
        if self.error:raise RuntimeError('camera publisher failed') from self.error
        d=self.env.mj_data
        state={'body_q':d.qpos[self.qadr].tolist(),'base_quat':d.xquat[self.body].tolist(),
               'base_position':d.xpos[self.body].tolist(),
               'left_hand_q':d.qpos[self.hand_adr[0]].tolist(),
               'right_hand_q':d.qpos[self.hand_adr[1]].tolist()}
        images={k:np.ascontiguousarray(frames[k+'_image']).copy()
                for k in ('ego_view','left_wrist','right_wrist')}
        snapshot={'index':self.index,'capture_monotonic':time.monotonic(),
                  'capture_simulation_time':float(d.time),'robot_state':state}
        self.index+=1
        try:self.pending.get_nowait()
        except queue.Empty:pass
        self.pending.put_nowait((images,snapshot))

    def _worker(self,port):
        context=zmq.Context();socket=context.socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER,0);socket.setsockopt(zmq.SNDHWM,1)
        try:
            socket.bind(f'tcp://127.0.0.1:{port}')
            self.ready.set()
            while not self.stop.is_set():
                try:images,packet=self.pending.get(timeout=.1)
                except queue.Empty:continue
                packet['images']={}
                for key,rgb in images.items():
                    ok,encoded=cv2.imencode('.jpg',rgb[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,95])
                    if not ok:raise RuntimeError('camera JPEG encoding failed')
                    packet['images'][key]=encoded.tobytes()
                packet['timestamps']={k:packet['capture_monotonic'] for k in images}
                socket.send(msgpack.packb(packet,use_bin_type=True),zmq.NOBLOCK)
        except BaseException as exc:
            self.error=exc;self.ready.set()
        finally:
            socket.close();context.term()

    def close(self):
        self.stop.set();self.thread.join(timeout=3)
        if self.thread.is_alive():raise RuntimeError('camera publisher did not stop')


def observation_state(packet,now,max_age):
    """Fail closed on missing or stale capture metadata; never substitute state."""
    captured=packet.get('capture_monotonic')
    if captured is None or not np.isfinite(captured) or not 0<=now-captured<=max_age:
        raise TimeoutError('synchronized camera capture missing/stale')
    state=packet.get('robot_state')
    if not isinstance(state,dict):raise ValueError('camera packet has no synchronized robot state')
    for key,shape in (('body_q',(29,)),('base_quat',(4,)),('base_position',(3,)),
                      ('left_hand_q',(7,)),('right_hand_q',(7,))):
        value=np.asarray(state.get(key),float)
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError('invalid synchronized '+key)
    return state
