"""Launch the official SONIC MuJoCo/DDS simulator with three real render views.

Run in SONIC's Linux simulation environment with this project on PYTHONPATH.
Generated MJCF stays in a temporary folder; the upstream checkout is untouched.
"""

import argparse
from pathlib import Path
import sys
import tempfile
import json
import threading
import time


def bgr_render_cache(frames):
    """MuJoCo renders RGB; the upstream OpenCV JPEG publisher expects BGR."""
    import numpy as np
    return {name: np.ascontiguousarray(rgb[:, :, ::-1]) for name, rgb in frames.items()}


def task_video_due(task, suspended, finalized, now, previous, fps):
    """Keep initialization visible when requested, using the same frame clock."""
    return (not finalized and (task.get('record_from_initialization',False) or not suspended)
            and now-previous>=1/fps-1e-6)


def prepare_scene(sonic_root, scene_path, output, overview=False, task=None):
    import mujoco
    scene_path = Path(scene_path).resolve()
    spec = mujoco.MjSpec.from_file(str(scene_path))
    # Resolving these is necessary when saving the generated XML elsewhere.
    spec.compiler.meshdir = str((scene_path.parent / spec.compiler.meshdir).resolve())
    spec.compiler.texturedir = str((scene_path.parent / spec.compiler.texturedir).resolve())
    cameras = {
        "unifolm_head": ("torso_link", [0.06, 0, 0.40]),
        "unifolm_left_wrist": ("left_wrist_yaw_link", [0.04, 0, 0.04]),
        "unifolm_right_wrist": ("right_wrist_yaw_link", [0.04, 0, 0.04]),
    }
    # These are simulation fixture mounts, not the real training camera calibration.
    for name, (body, pos) in cameras.items():
        quat = [0.5, 0.5, -0.5, -0.5]
        if task:
            from scipy.spatial.transform import Rotation
            if body != "torso_link":
                pos = [.10, 0, .12]  # clear the palm mesh, look toward fingertips/table
            base = Rotation.from_quat([.5, -.5, -.5, .5])
            rot = Rotation.from_euler("y", 45 if body == "torso_link" else 60, degrees=True) * base
            quat = rot.as_quat()[[3, 0, 1, 2]]
        fovy = (82.5 if body == "torso_link" else 58.0) if task else 75
        mount = (task or {}).get("camera_mounts", {}).get(name, {})
        if mount:
            from scipy.spatial.transform import Rotation
            pos = mount.get("position", pos)
            if "pitch_degrees" in mount:
                base = Rotation.from_quat([.5, -.5, -.5, .5])
                rot = Rotation.from_euler("y", mount["pitch_degrees"], degrees=True)*base
                quat = rot.as_quat()[[3, 0, 1, 2]]
            if 'quaternion_wxyz' in mount:quat=mount['quaternion_wxyz']
            fovy = mount.get("fovy", fovy)
        spec.body(body).add_camera(name=name, pos=pos, quat=quat, fovy=fovy)
    if overview:
        import numpy as np
        from scipy.spatial.transform import Rotation
        pos = np.array([1.8, -1.5, 1.4] if task else [2.5, -2, 1.6])
        z = pos - ([.2, 0, .75] if task else [0, 0, .7])
        z /= np.linalg.norm(z)
        x = np.cross([0, 0, 1], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        xyzw = Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()
        spec.worldbody.add_camera(name="unifolm_overview", pos=pos, quat=xyzw[[3, 0, 1, 2]], fovy=45 if task else 55)
    spec.compile()  # fail before launching DDS if meshes are missing/LFS pointers
    output = Path(output)
    output.write_text(spec.to_xml())
    if task:
        from .grasp import tabletop_scene
        tabletop_scene(output, task)
    mujoco.MjModel.from_xml_path(str(output))  # verify relocated asset paths
    result = {
        key: {"height": 336, "width": 448, "mjcf_name": camera}
        for key, camera in zip(("ego_view", "left_wrist", "right_wrist"), cameras)
    }
    if overview:
        result["overview"] = {"height": 384, "width": 512, "mjcf_name": "unifolm_overview"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic-root", required=True, type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--overview", action="store_true", help="Add a fourth external camera for validation snapshots")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--domain-id", type=int, default=73)
    parser.add_argument("--release-file", type=Path, help="Create this file to release the simulated suspension")
    parser.add_argument("--metrics-file", type=Path, help="Write physical simulator measurements as JSONL")
    parser.add_argument("--task-config", type=Path, help="Tabletop task YAML; uses physical two-jaw grippers")
    parser.add_argument("--task-output", type=Path, help="Task scores, physics trace and video directory")
    parser.add_argument("--episode-file", type=Path, help="Policy lifecycle marker; freezes scoring before shutdown")
    parser.add_argument("--rgbd-port", type=int, help="Publish head RGB-D plus robot calibration on loopback")
    parser.add_argument("--arm-tracking", choices=("sonic", "joint_pd", "arms_pd"), default="sonic")
    parser.add_argument("--tactile-port", type=int, help="Publish jaw forces and robot proprioception only")
    args = parser.parse_args()
    task = None
    if args.task_config:
        import yaml
        task = yaml.safe_load(args.task_config.read_text())
        task["arm_tracking"] = args.arm_tracking
        if not args.task_output or not args.episode_file:
            parser.error("tabletop tasks require --task-output and --episode-file")
    if args.rgbd_port and not task:
        parser.error("RGB-D publishing currently requires --task-config")
    if task and task.get('startup_arm_q') is not None and args.arm_tracking not in ('joint_pd','arms_pd'):
        parser.error('startup_arm_q requires the explicit simulation arm PD mode')
    sys.path.insert(0, str(args.sonic_root.resolve()))
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
    from gear_sonic.utils.mujoco_sim.simulator_factory import init_channel
    config = SimLoopConfig(interface="sim", enable_offscreen=True,
                           enable_onscreen=not args.headless, enable_image_publish=True)
    wbc = config.load_wbc_yaml()
    if task and 'camera_fps' in task:
        fps=float(task['camera_fps'])
        if not 1<=fps<=60:raise ValueError('camera_fps must be between 1 and 60')
        wbc['IMAGE_DT']=1/fps
    wbc["DOMAIN_ID"] = args.domain_id
    if wbc["ENV_TYPE"] != "sim":
        raise ValueError("only simulation is supported")
    with tempfile.TemporaryDirectory(prefix="unifolm-sonic-scene-") as tmp:
        scene = Path(tmp) / "scene.xml"
        cameras = prepare_scene(args.sonic_root, args.sonic_root / wbc["ROBOT_SCENE"], scene, args.overview, task)
        wbc["ROBOT_SCENE"] = str(scene)
        init_channel(wbc)
        simulator = BaseSimulator(config=wbc, env_name="default", camera_configs=cameras,
                                  onscreen=not args.headless, offscreen=True, enable_image_publish=True)
        if task and task.get('gripper_type') in ('parallel','dex1_cad'):
            from .parallel_gripper import ParallelGripperServo
            hand_servo = ParallelGripperServo(simulator.sim_env,
                kp=task.get('gripper_servo_kp',250),
                closed_limit=.0445 if task.get('gripper_type')=='dex1_cad' else .045)
            native_hand_forces = simulator.sim_env.compute_hand_torques
            simulator.sim_env.compute_hand_torques = lambda: hand_servo.forces(native_hand_forces())
        task_monitor = None
        tactile = None
        if args.tactile_port:
            from .tactile import TactilePublisher
            tactile = TactilePublisher(simulator.sim_env,args.tactile_port)
        arm_servo = None
        if args.arm_tracking in ('joint_pd','arms_pd'):
            from .arm_servo import ArmReferenceServo
            arm_servo = ArmReferenceServo(simulator.sim_env, "tcp://127.0.0.1:15556",waist=args.arm_tracking=='joint_pd',
                                          startup_arm_q=(task or {}).get('startup_arm_q'))
            native_torques = simulator.sim_env.compute_body_torques
            simulator.sim_env.compute_body_torques = lambda: arm_servo.torques(native_torques())
        video = None
        sync_camera = None
        if task and task.get('synchronized_cameras',False):
            from .sync_camera import SynchronizedCameraPublisher
            sync_camera=SynchronizedCameraPublisher(simulator.sim_env,args.camera_port)
        rgbd = None
        if args.rgbd_port:
            from .rgbd import RGBDPublisher
            rgbd = RGBDPublisher(simulator.sim_env, args.rgbd_port)
        if task:
            from .grasp import GraspMonitor
            import numpy as np
            # Upstream initializes this as an integer array; assigning .82 in
            # place would truncate it to zero and pull the robot into the floor.
            simulator.sim_env.elastic_band.point = np.array([0., 0., task.get("band_anchor_height", .82)])
            task_monitor = GraspMonitor(simulator.sim_env, task, args.task_output, args.episode_file)
            original_step = simulator.sim_env.sim_step
            def task_step():
                original_step()
                if tactile:tactile.publish()
                task_monitor.sample()
            simulator.sim_env.sim_step = task_step
            original_render = simulator.sim_env.update_render_caches
            last_video_time = -1.0
            video_fps=float(task.get('camera_fps',10))
            def render_and_record():
                nonlocal video, last_video_time
                frames = original_render()
                if sync_camera:sync_camera.capture(frames)
                if rgbd:
                    rgbd.publish(frames["ego_view_image"])
                now = simulator.sim_env.mj_data.time
                if task_video_due(task,simulator.sim_env.elastic_band.enable,task_monitor.finalized,
                                  now,last_video_time,video_fps):
                    import cv2
                    last_video_time = now
                    if "overview_image" in frames:
                        rgb = frames["overview_image"]
                        if video is None:
                            video = cv2.VideoWriter(str(args.task_output / "overview.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (rgb.shape[1], rgb.shape[0]))
                            if not video.isOpened():
                                raise RuntimeError("could not create task video")
                            (args.task_output/'video-timing.json').write_text(json.dumps({
                                'first_frame_simulation_time':float(now),'fps':video_fps,
                                'record_from_initialization':bool(task.get('record_from_initialization',False)),
                                'suspended_at_first_frame':bool(simulator.sim_env.elastic_band.enable)},indent=2)+'\n')
                        video.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    if int(now * 10) % 50 == 0:
                        for key, rgb in frames.items():
                            cv2.imwrite(str(args.task_output / f"{now:07.2f}-{key}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                return frames
            simulator.sim_env.update_render_caches = render_and_record
            if task.get('record_from_initialization',False):
                # Capture the initialized pose at t=0, before the first physics
                # step. The existing snapshot path also saves 0000.00-*.jpg.
                render_and_record()
        def monitor():
            released = False
            while simulator._running:
                if args.release_file and args.release_file.exists() and not released:
                    simulator.sim_env.elastic_band.enable = False
                    print("UNIFOLM_SIM: suspension released", flush=True)
                    released = True
                if args.metrics_file:
                    data = simulator.sim_env.mj_data
                    record = {"wall_time": time.time(), "simulation_time": float(data.time),
                              "base_position": data.qpos[:3].tolist(),
                              "base_height": float(data.qpos[2]),
                              "base_quat_wxyz": data.qpos[3:7].tolist(),
                              "suspension_enabled": bool(simulator.sim_env.elastic_band.enable)}
                    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
                    with args.metrics_file.open("a") as f:
                        f.write(json.dumps(record) + "\n")
                time.sleep(0.2)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        try:
            # The upstream factory catches all exceptions and returns success;
            # keep failures visible to this task's CLI and evaluator instead.
            if sync_camera is None:
                simulator.start_image_publish_subprocess(camera_port=args.camera_port)
                image_process = simulator.sim_env.image_publish_process
                original_publish = image_process.update_shared_memory
                image_process.update_shared_memory = lambda frames: original_publish(bgr_render_cache(frames))
            if task and task.get('realtime_clock',False):
                from .realtime import run_realtime
                run_realtime(simulator)
            else:
                simulator.start()
        finally:
            if args.task_output and hasattr(simulator,'realtime_report'):
                (args.task_output/'clock.json').write_text(json.dumps(simulator.realtime_report,indent=2)+'\n')
            if task_monitor:
                task_monitor.close()
            if video:
                video.release()
            if rgbd:
                rgbd.close()
            if arm_servo:
                arm_servo.close()
            if tactile:
                tactile.close()
            if sync_camera:
                sync_camera.close()


if __name__ == "__main__":
    main()
