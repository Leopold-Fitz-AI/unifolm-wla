import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml
import zmq

from robot_client.sonic.kinematics import JOINT_NAMES
from robot_client.sonic.launch_sim import prepare_scene
from robot_client.sonic.sync_camera import SynchronizedCameraPublisher,observation_state
from robot_client.sonic.transport import LatestSubscriber,decode_cameras


def test_render_packet_keeps_matching_robot_state_and_rgb_after_physics_advances(tmp_path):
    root=os.environ.get('SONIC_ROOT')
    if not root:pytest.skip('set SONIC_ROOT for real robot layout')
    root=Path(root);scene=tmp_path/'scene.xml'
    task=yaml.safe_load(Path('robot_client/sonic/tabletop_dex1_visible.yaml').read_text())
    prepare_scene(root,root/'gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml',scene,task=task)
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    adr=model.jnt_qposadr[model.joint(JOINT_NAMES[0]).id]
    data.qpos[adr]=.1;mujoco.mj_forward(model,data)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    context=zmq.Context();sub=LatestSubscriber(context,f'tcp://127.0.0.1:{port}')
    publisher=SynchronizedCameraPublisher(SimpleNamespace(mj_model=model,mj_data=data),port)
    try:
        rgb=np.zeros((48,64,3),np.uint8);rgb[:,:,0]=250
        frames={name+'_image':rgb for name in ('ego_view','left_wrist','right_wrist')}
        # Establish PUB/SUB with real warmup frames rather than guessing when
        # an asynchronous TCP reconnect/subscription handshake has completed.
        deadline=time.monotonic()+2;warmup=None
        while warmup is None and time.monotonic()<deadline:
            publisher.capture(frames)
            time.sleep(.01)
            warmup=sub.read(time.monotonic(),.5)
        assert warmup is not None,repr(publisher.error)
        wanted=publisher.index
        publisher.capture(frames)
        # The encoder may still be pending. Physics and render caches can change.
        data.qpos[adr]=.7;rgb[:]=[0,0,250]
        deadline=time.monotonic()+2;packet=None
        while (packet is None or packet['index']!=wanted) and time.monotonic()<deadline:
            packet=sub.read(time.monotonic(),.5);time.sleep(.005)
        assert packet is not None
        assert packet['index']==wanted
        state=observation_state(packet,time.monotonic(),.5)
        assert state['body_q'][0]==pytest.approx(.1)
        assert 'objects' not in state and 'objects' not in packet
        image=decode_cameras(packet,{'cam_left_high':'ego_view'})['cam_left_high']
        assert image[:,:,2].mean()>240 and image[:,:,0].mean()<5
        # No new render: encoding/publishing must not refresh capture freshness.
        assert sub.read(time.monotonic()+1,.5) is None
        with pytest.raises(TimeoutError,match='capture missing/stale'):
            observation_state(packet,time.monotonic()+1,.5)
    finally:
        publisher.close();sub.close();context.term()


def test_synchronized_mode_refuses_missing_capture_metadata():
    with pytest.raises(TimeoutError,match='capture missing/stale'):
        observation_state({'robot_state':{}},time.monotonic(),.5)
