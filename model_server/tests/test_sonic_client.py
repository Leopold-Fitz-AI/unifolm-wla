"""Cross-check wire bytes against upstream and exercise actual G1 FK/IK.

Set SONIC_ROOT to the pinned NVlabs checkout to run the G1 integration cases.
No training stack, model weights or GPU are needed for this test file.
"""

import asyncio
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from robot_client.sonic.adapter import ActionAdapter, Reference, make_observation
from robot_client.sonic.kinematics import G1Kinematics
from robot_client.sonic.protocol import HEADER_SIZE, ISAAC_TO_MUJOCO, MUJOCO_TO_ISAAC, pack_reference
from robot_client.sonic.protocol import UPPER_MUJOCO_IN_ISAAC, pack_fields, pack_command
from robot_client.sonic.planner import PlannerMapper
from robot_client.sonic.smoke import decode_packet, fixture_action, smoke
from robot_client.sonic.transport import attach_sim_hands, decode_cameras


@pytest.fixture
def sonic_root():
    root = os.environ.get("SONIC_ROOT")
    if not root:
        pytest.skip("set SONIC_ROOT to run upstream compatibility and actual G1 tests")
    return Path(root)


@pytest.fixture
def kin(sonic_root):
    return G1Kinematics(sonic_root)


def test_wire_bytes_match_current_upstream_builder(sonic_root):
    path = sonic_root / "gear_sonic/utils/teleop/zmq/zmq_planner_sender.py"
    spec = importlib.util.spec_from_file_location("sonic_wire_builder", path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    q = np.tile(np.arange(29, dtype=float), (5, 1)) / 30
    dq = q / 2
    quat = np.tile([1, 0, 0, 0], (5, 1))
    indices = np.arange(100, 105)
    expected = upstream.pack_pose_message({
        "joint_pos": q[:, MUJOCO_TO_ISAAC].astype("<f4"),
        "joint_vel": dq[:, MUJOCO_TO_ISAAC].astype("<f4"),
        "body_quat": quat.astype("<f4"),
        "frame_index": indices.astype("<i8"),
    }, version=1)
    assert pack_reference(q, dq, quat, indices) == expected
    header_path = sonic_root / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"
    match = re.search(r"mujoco_to_isaaclab\s*=\s*\{([^}]+)\}", header_path.read_text())
    np.testing.assert_array_equal(MUJOCO_TO_ISAAC, [int(v) for v in re.findall(r"\d+", match[1])])
    fields = PlannerMapper().fields(q[0], dq[0], [.2, -.1, .1, .74],
                                    {"base_quat": [1, 0, 0, 0], "init_base_quat": [1, 0, 0, 0]},
                                    .02, np.zeros((2, 7)))
    expected = upstream.build_planner_message(int(fields['mode'][0]), fields['movement'], fields['facing'],
        float(fields['speed'][0]), float(fields['height'][0]), fields['upper_body_position'],
        fields['upper_body_velocity'], fields['left_hand_joints'], fields['right_hand_joints'])
    assert pack_fields("planner", fields) == expected
    assert pack_command(start=True) == upstream.build_command_message(True, False, True)
    match = re.search(r"upper_body_joint_isaaclab_order_in_mujoco_index\s*=\s*\{([^}]+)\}", header_path.read_text())
    np.testing.assert_array_equal(UPPER_MUJOCO_IN_ISAAC, [int(v) for v in re.findall(r"\d+", match[1])])


def test_packet_absolute_joint_order_and_1280_header():
    q = np.tile(np.arange(29), (3, 1)).astype(float)
    packet = pack_reference(q, q * 0, np.tile([1, 0, 0, 0], (3, 1)), np.arange(3))
    assert HEADER_SIZE == 1280
    arrays = decode_packet(packet)
    np.testing.assert_array_equal(arrays["joint_pos"][:, ISAAC_TO_MUJOCO], q)
    assert packet[4:5] == b"{"


def test_reject_nonfinite_and_invalid_frame_timeline():
    q = np.zeros((3, 29))
    quat = np.tile([1, 0, 0, 0], (3, 1))
    with pytest.raises(ValueError, match="consecutive"):
        pack_reference(q, q, quat, [1, 2, 4])
    q[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pack_reference(q, q, quat, np.arange(3))


def test_30_to_50_hz_and_absolute_frame_alignment():
    q = np.zeros((3, 29))
    q[:, 0] = [0, 1 / 30, 2 / 30]
    ref = Reference(10 + np.arange(3) / 30, q, np.array([1, 0, 0, 0]), np.zeros((3, 2)))
    packet, _, _ = ref.packet(10.02, 10, lookahead=4)
    arrays = decode_packet(packet)
    np.testing.assert_array_equal(arrays["frame_index"], [1, 2, 3, 4])
    sampled, _, _ = ref.sample(np.array([10.02, 10.04, 11]))
    np.testing.assert_allclose(sampled[:, 0], [.02, .04, 2 / 30])


def test_real_g1_ik_preserves_legs_and_recovers_tools(sonic_root):
    kin = G1Kinematics(sonic_root, tool_offsets=[[.04, .01, 0, .1, 0, 0], [.03, 0, .02, 0, .1, 0]])
    seed = np.zeros(29)
    seed[[18, 25]] = .5
    desired = seed.copy()
    desired[[15, 22]] += .04
    desired[12] = .1
    target = kin.forward(desired)
    seed[:15] = desired[:15]
    solved = kin.solve(target, seed)
    np.testing.assert_array_equal(solved[:15], desired[:15])
    for actual, expected in zip(kin.forward(solved), target):
        np.testing.assert_allclose(actual, expected, atol=.005)


def test_unreachable_target_fails(kin):
    targets = kin.forward(np.zeros(29))
    targets[0][:3, 3] += 10
    with pytest.raises(ValueError, match="unreachable"):
        kin.solve(targets, np.zeros(29))


def test_soft_limit_measurement_seed_keeps_strict_output_limits(kin):
    q = np.zeros(29)
    q[0] = kin.bounds[0, 1] + .001
    with pytest.raises(ValueError):
        kin.check_q(q)
    assert kin.seed_q(q)[0] == kin.bounds[0, 1]
    q[0] += .1
    with pytest.raises(ValueError, match="soft-limit"):
        kin.seed_q(q)


def test_planner_arm_seed_does_not_reject_native_leg_measurement(kin):
    q=np.zeros(29)
    q[10]=kin.bounds[10,1]+.1
    clipped=kin.seed_q(q,planner_legs=True)
    assert clipped[10]==kin.bounds[10,1]
    # Legs do not affect pelvis-relative arm FK. Actual VLA leg observations
    # still come from the untouched state packet, not this optimization seed.
    for measured,seed in zip(kin.forward(q),kin.forward(clipped)):
        np.testing.assert_allclose(measured,seed,atol=1e-9)
    q[20]=kin.bounds[20,1]+.1
    with pytest.raises(ValueError,match='soft-limit'):
        kin.seed_q(q,planner_legs=True)


def test_observation_rot6d_and_lower_body_order(kin):
    q = np.zeros(29)
    q[[12, 18, 25]] = [.2, .5, .5]
    images = {k: np.zeros((12, 16, 3), np.uint8) for k in ("cam_left_high", "cam_left_wrist", "cam_right_wrist")}
    obs = make_observation(kin, {"body_q": q}, images, [.2, .8], "test", "dataset")
    for side, pose in zip(("left", "right"), kin.forward(q)):
        rot6d = obs[f"observation.state.{side}_ee_6d"][3:]
        np.testing.assert_allclose(rot6d.reshape(2, 3).T, pose[:3, :2], atol=1e-6)
    np.testing.assert_array_equal(obs["observation.state.lower_body"], q[:15].astype(np.float32))
    assert obs["unnorm_key"] == "dataset"


@pytest.mark.parametrize("error", ["nan", "length", "base", "gripper"])
def test_action_validation(kin, error):
    q = np.zeros(29)
    q[[18, 25]] = .5
    action = fixture_action(kin, q)
    adapter = ActionAdapter(kin)
    if error == "nan":
        action["action.left_ee_rpy"][0, 0, 0] = np.nan
    elif error == "length":
        action["action.lower_body"] = action["action.lower_body"][:, :-1]
    elif error == "base":
        action["action.base_command"][0, 0, 0] = .2
    elif error == "gripper":
        action["action.left_gripper"][0, 0, 0] = 10
    with pytest.raises(ValueError):
        adapter.convert(action, q, [1, 0, 0, 0], [0, 0], 100)


def test_first_step_and_arrival_seams_are_rate_limited(kin):
    q = np.zeros(29)
    q[[18, 25]] = .5
    ref = ActionAdapter(kin, max_joint_velocity=.01).convert(fixture_action(kin, q), q, [1, 0, 0, 0], [0, 0], 10)
    assert np.max(np.abs(np.diff(ref.q, axis=0))) * 30 <= .010001
    assert ref.limited_frames > 0
    # Simulate inference completing 0.4 seconds after the observation.
    packet, command, _ = ref.packet(10.4, 10, previous=(q, 10.38), max_velocity=.01)
    assert np.max(np.abs(command - q)) <= .000201
    arrays = decode_packet(packet)
    assert np.max(np.abs(np.diff(arrays['joint_pos'], axis=0))) * 50 <= .010001


def test_infeasible_future_tail_keeps_prefix_and_original_deadline(kin):
    q = np.zeros(29)
    q[[18, 25]] = .5
    action = fixture_action(kin, q)
    action['action.left_ee_rpy'][0, 20:, 0] = 10
    adapter = ActionAdapter(kin, truncate_infeasible_tail=True)
    ref = adapter.convert(action, q, [1, 0, 0, 0], [0, 0], 100)
    assert ref.truncated_tail_frames == 10
    assert len(ref.q) == len(ref.grippers) == len(ref.base_commands) == 21
    assert ref.times[-1] == pytest.approx(100 + 20 / 30)


def test_first_frame_alignment_keeps_predicted_relative_motion(kin):
    from robot_client.sonic.kinematics import pose_matrix
    q = np.zeros(29)
    q[[18, 25]] = .5
    action = fixture_action(kin, q)
    action['action.left_ee_rpy'][0, :, 2] += .01
    adapter = ActionAdapter(kin, max_joint_velocity=100, align_first_ee_frame=True)
    ref = adapter.convert(action, q, [1, 0, 0, 0], [0, 0], 100)
    observed = kin.forward(q)[0]
    first = pose_matrix(action['action.left_ee_rpy'][0, 0])
    last = pose_matrix(action['action.left_ee_rpy'][0, -1])
    np.testing.assert_allclose(kin.forward(ref.q[1])[0], observed, atol=.001)
    np.testing.assert_allclose(kin.forward(ref.q[-1])[0], observed @ np.linalg.inv(first) @ last, atol=.005)


def test_small_gripper_regression_overshoot_is_bounded_and_counted(kin):
    q = np.zeros(29)
    q[[18, 25]] = .5
    action = fixture_action(kin, q)
    action['action.left_gripper'][0, 0, 0] = -.006
    adapter = ActionAdapter(kin, gripper_overshoot_tolerance=.05)
    ref = adapter.convert(action, q, [1, 0, 0, 0], [0, 0], 100)
    assert ref.grippers[1, 0] == 0
    assert ref.gripper_clamped_values == 1


def test_planner_owns_legs_and_accepts_base_velocity(kin):
    q = np.zeros(29)
    q[[18, 25]] = .5
    action = fixture_action(kin, q)
    action['action.base_command'][0, :, 0] = .3
    action['action.lower_body'][0, :, :12] = 99  # never used as leg goals in planner mode
    adapter = ActionAdapter(kin)
    adapter.planner_mode = True
    ref = adapter.convert(action, q, [1, 0, 0, 0], [0, 0], 10)
    np.testing.assert_allclose(ref.q[:, :12], np.broadcast_to(q[:12], (31, 12)))
    assert ref.base_commands[-1, 0] == pytest.approx(.3)


def test_stationary_height_uses_official_squat_mode_without_walking():
    from robot_client.sonic.planner import PlannerMapper
    state={'base_quat':[1,0,0,0]}
    mapper=PlannerMapper(stationary_mode=4,max_speed=0)
    fields=mapper.fields(np.zeros(29),np.zeros(29),[0,0,0,.69],state,.02,np.zeros((2,7)))
    assert fields['mode'].tolist()==[4]
    assert fields['height'][0]==pytest.approx(.69)
    assert fields['speed'][0]==0
    np.testing.assert_equal(fields['movement'],[0,0,0])
    assert PlannerMapper().fields(np.zeros(29),np.zeros(29),[0,0,0,.75],state,.02,np.zeros((2,7)))['mode'][0]==0
    with pytest.raises(ValueError,match='stationary_mode'):PlannerMapper(stationary_mode=12)


def test_planner_local_velocity_rotation_limits_and_acceleration():
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_euler('z', np.pi / 2).as_quat()
    state = {'base_quat': xyzw[[3, 0, 1, 2]], 'init_base_quat': [1, 0, 0, 0]}
    mapper = PlannerMapper()
    fields = mapper.fields(np.arange(29), np.zeros(29), [10, 0, 10, 2], state, .1, np.zeros((2, 7)))
    assert fields['speed'][0] == pytest.approx(.05)  # 0.5 m/s^2 over 0.1 s
    np.testing.assert_allclose(fields['movement'], [0, 1, 0], atol=1e-6)
    assert fields['height'][0] == pytest.approx(.8)
    np.testing.assert_array_equal(fields['upper_body_position'], np.arange(29)[UPPER_MUJOCO_IN_ISAAC])
    idle = mapper.idle_packet(state, .75, .02)
    assert b'"upper_body_position"' not in idle
    assert b'"left_hand_joints"' not in idle


def test_planner_bootstraps_before_waiting_for_feedback(monkeypatch):
    # Native SONIC emits g1_debug only after CONTROL starts. Waiting for that
    # stream before sending start would deadlock both peers at startup.
    import importlib
    client = importlib.import_module("robot_client.sonic.run")
    sent = []
    class Publisher:
        def setsockopt(self, *args): pass
        def bind(self, *args): pass
        def send(self, packet, *args): sent.append(packet)
        def close(self): pass
    class Context:
        def socket(self, *args): return Publisher()
        def term(self): pass
    class Subscriber:
        def __init__(self, *args): pass
        def read(self, *args):
            assert pack_command(start=True) in sent
            return None
        def close(self): pass
    class Policy:
        metadata = {}
        def __init__(self, *args): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    monkeypatch.setattr(client.zmq, "Context", Context)
    monkeypatch.setattr(client, "LatestSubscriber", Subscriber)
    monkeypatch.setattr(client, "PolicyClient", Policy)
    monkeypatch.setattr(client, "G1Kinematics", lambda *args: None)
    config = client.load_config(Path(client.__file__).with_name("sim.yaml"))
    with pytest.raises(RuntimeError, match="no model action chunk"):
        asyncio.run(client.run(config, Path("unused"), max_seconds=.03))
    assert sent[-1] == pack_command(stop=True)


def test_camera_bgr_and_missing_view():
    import cv2
    image = np.full((12, 16, 3), [10, 50, 180], np.uint8)
    _, encoded = cv2.imencode(".jpg", image)
    result = decode_cameras({"images": {"head": encoded.tobytes()}}, {"cam_left_high": "head"})
    np.testing.assert_allclose(result["cam_left_high"][0, 0], [10, 50, 180], atol=2)
    with pytest.raises(KeyError):
        decode_cameras({"images": {}}, {"cam_left_wrist": "wrist"})


def test_sim_hand_values_are_applied_to_current_frame():
    packet = pack_reference(np.zeros((3, 29)), np.zeros((3, 29)), np.tile([1, 0, 0, 0], (3, 1)), np.arange(3))
    arrays = decode_packet(attach_sim_hands(packet, [.25, .75], np.zeros((2, 7)), np.ones((2, 7))))
    np.testing.assert_allclose(arrays["left_hand_joints"], .25)
    np.testing.assert_allclose(arrays["right_hand_joints"], .75)


def test_loopback_real_sockets_and_g1_kinematics(sonic_root):
    report = asyncio.run(smoke(sonic_root, seconds=2.5))
    assert report["chunks"] >= 2
    assert report["received_packets"] > 20
    assert report["max_joint_excursion_rad"] > .001


def test_unreachable_chunk_is_rejected_and_fresh_observation_recovers(sonic_root,monkeypatch):
    original=ActionAdapter.convert
    attempted=0
    def intermittent(self,*args,**kwargs):
        nonlocal attempted
        attempted+=1
        if attempted==1:raise ValueError('IK target unreachable within tolerances: fixture')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(ActionAdapter,'convert',intermittent)
    report=asyncio.run(smoke(sonic_root,seconds=2.5,config_overrides={'max_ik_rejections':2}))
    assert report['rejected_ik_chunks']==1
    assert report['chunks']>=2
    assert report['received_packets']>20


def test_full_chunk_executes_late_motion_before_replanning(sonic_root, monkeypatch):
    import importlib
    fixture = importlib.import_module('robot_client.sonic.smoke')
    from robot_client.sonic.adapter import pose_to_rpy
    original = fixture.fixture_action
    def late_motion(kin, q0, frames=30):
        action = original(kin, q0, frames)
        poses = []
        for i in range(frames):
            q = q0.copy()
            q[15] += .05 if i >= 25 else 0
            poses.append([pose_to_rpy(t) for t in kin.forward(q)])
        poses = np.asarray(poses)
        action['action.left_ee_rpy'] = poses[None, :, 0]
        action['action.right_ee_rpy'] = poses[None, :, 1]
        return action
    monkeypatch.setattr(fixture, 'fixture_action', late_motion)
    report = asyncio.run(fixture.smoke(sonic_root, seconds=2.8,
                                      config_overrides={'execute_full_chunk': True}))
    assert report['chunks'] >= 2
    # The only nonzero goal is in the final 5 frames. Replanning every .35 s
    # would starve it indefinitely despite a valid one-second model forecast.
    assert report['max_joint_excursion_rad'] > .04


def test_cartesian_feedback_reduces_a_fixed_tracking_bias(sonic_root):
    from robot_client.sonic.tracking import CartesianTracker
    tracker = CartesianTracker(sonic_root, 'pelvis', [[0]*6]*2)
    kin = G1Kinematics(sonic_root)
    target = np.zeros(29)
    target[[18,25]] = .5
    plant_bias = np.zeros(29)
    plant_bias[[15,22]] = -.06
    measured = target+plant_bias
    goal = np.array([p[:3,3] for p in kin.forward(target)])
    before = np.linalg.norm(np.array([p[:3,3] for p in kin.forward(measured)])-goal)
    for _ in range(200):
        command = tracker.update(target, measured, .02)
        measured = kin.seed_q(command+plant_bias)
    after = np.linalg.norm(np.array([p[:3,3] for p in kin.forward(measured)])-goal)
    assert after < before*.3
    assert np.max(np.abs(tracker.integral)) <= .04


@pytest.mark.parametrize('first_joint',[12,15])
def test_upper_servo_preserves_native_leg_torques(sonic_root,first_joint):
    from types import SimpleNamespace
    from robot_client.sonic.arm_servo import ArmReferenceServo, planner_fields
    from robot_client.sonic.kinematics import JOINT_NAMES
    kin = G1Kinematics(sonic_root)
    q = np.zeros(29)
    q[[18,25]] = .5
    kin.forward(q)
    desired = q.copy()
    desired[12:] += .02
    velocity=np.zeros(29)
    velocity[12:]=.3
    fields = PlannerMapper().fields(desired, velocity, [0,0,0,.75],
                                     {'base_quat':[1,0,0,0]}, .02, np.zeros((2,7)))
    raw = pack_fields('planner', fields)
    np.testing.assert_allclose(planner_fields(raw)['upper_body_position'], desired[UPPER_MUJOCO_IN_ISAAC])
    servo = ArmReferenceServo.__new__(ArmReferenceServo)
    servo.reference, servo.received_at = None, float('-inf')
    servo.first_joint=first_joint
    servo.socket = SimpleNamespace(recv=lambda *a: raw)
    servo.env = SimpleNamespace(mj_model=kin.model, mj_data=kin.data,
        body_joint_index=[kin.model.joint(n).id for n in JOINT_NAMES],
        unitree_bridge=SimpleNamespace(low_cmd=SimpleNamespace(motor_cmd=[SimpleNamespace(kp=40,kd=1)]*29)))
    native = np.arange(29, dtype=float)
    result = servo.torques(native)
    np.testing.assert_array_equal(result[:first_joint], native[:first_joint])
    for i in range(first_joint,29):
        va = kin.model.jnt_dofadr[servo.env.body_joint_index[i]]
        assert result[i] == pytest.approx(1.1+kin.data.qfrc_bias[va], abs=1e-5)


def test_inference_geometry_does_not_import_training_readers():
    subprocess.run([sys.executable, "-c", (
        "import sys; "
        "from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES; "
        "assert SLICES['height'].start == 41; "
        "assert not any(m in sys.modules for m in ('torch', 'lerobot', 'datasets', 'pyarrow'))"
    )], check=True, cwd=Path(__file__).resolve().parents[2])
