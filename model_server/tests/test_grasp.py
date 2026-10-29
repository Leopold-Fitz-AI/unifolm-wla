import os
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from robot_client.sonic.grasp import GraspScore, enable_task_contacts
from robot_client.sonic.launch_sim import prepare_scene, bgr_render_cache
from robot_client.sonic.run import load_config
from robot_client.sonic.transport import model_gripper_values, gripper_closed_fraction
from robot_client.sonic.rgbd import locate_cube
from model_server.grounding import parse_grounding


def test_dex1_open_close_polarity_and_feedback_roundtrip():
    opened, closed = np.array([5.5, 5.5]), np.zeros(2)
    np.testing.assert_allclose(model_gripper_values([0, 1], opened, closed), [5.5, 0])
    np.testing.assert_allclose(gripper_closed_fraction([5.5, 0], opened, closed), [0, 1])
    values = model_gripper_values([.3, .7], opened, closed)
    np.testing.assert_allclose(gripper_closed_fraction(values, opened, closed), [.3, .7])


def test_mujoco_rgb_to_jpeg_roundtrip_preserves_object_color():
    import cv2
    rgb = np.full((20, 20, 3), [220, 30, 10], np.uint8)
    bgr = bgr_render_cache({"head_image": rgb})["head_image"]
    _, jpeg = cv2.imencode(".jpg", bgr)
    observed = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    # These are precisely the BGR bytes the model server converts back to RGB.
    np.testing.assert_allclose(cv2.cvtColor(observed, cv2.COLOR_BGR2RGB)[10, 10], rgb[10, 10], atol=2)


def test_grounding_parser_requires_valid_semantic_box():
    g = parse_grounding('```json\n{"color":"red","hand":"right","bbox":[100,200,300,400]}\n```')
    assert g['color'] == 'red' and g['hand'] == 'right'
    for text in ('no object', '{"error":"target_not_visible"}',
                 '{"color":"red","hand":"right","bbox":[300,200,100,400]}'):
        with pytest.raises(ValueError): parse_grounding(text)


def test_native_model_points_reject_separated_objects():
    result = parse_grounding('[{"point_2d":[690,640]},{"point_2d":[710,650]}]')
    assert result['point_2d'] == [700, 645]
    with pytest.raises(ValueError, match='separated objects'):
        parse_grounding('[{"point_2d":[100,200]},{"point_2d":[800,650]}]')


def test_rgbd_target_uses_model_box_and_sensor_depth():
    bgr = np.zeros((40, 60, 3), np.uint8)
    bgr[15:25, 35:45] = [0, 0, 230]
    depth = np.full((40, 60), .5, dtype='<f4')
    transform = np.eye(4)
    transform[:3, 3] = [.2, 0, .6]
    packet = {'shape':[40,60], 'bgr':bgr.tobytes(), 'depth_f32':depth.tobytes(),
              'camera_to_pelvis':transform.tolist(), 'intrinsics':[100,100,29.5,19.5], 'up_pelvis':[0,0,1]}
    result = locate_cube(packet, {'color':'red','bbox':[550,300,800,700]})
    np.testing.assert_allclose(result['center_pelvis'], [.25, 0, .075], atol=.003)
    with pytest.raises(ValueError):
        locate_cube(packet, {'color':'blue','bbox':[550,300,800,700]})


def test_grasp_with_wrong_hand_does_not_pass_instruction():
    score = GraspScore('red_cube', required_hand='left')
    for t in (0,1,2,3):
        assert not score.update(supported_sample(t))


def supported_sample(time):
    return {"simulation_time": time, "base_height": .78, "base_tilt_deg": 3,
            "suspension_enabled": False,
            "objects": {"red_cube": {"bottom_above_table": .10, "jaw_contacts": {"right": [0, 1]}, "table_contact": False},
                        "blue_cube": {"bottom_above_table": 0, "jaw_contacts": {}, "table_contact": True}}}


def test_scoring_needs_sustained_contact_and_rejects_reset():
    score = GraspScore("red_cube")
    assert not score.update(supported_sample(1))
    assert not score.update(supported_sample(2))
    assert score.update(supported_sample(3.01))
    assert not score.update(supported_sample(0))
    assert score.invalid_reset


@pytest.mark.parametrize("invalid", ["airborne", "one_jaw", "table", "suspended", "wrong_object"])
def test_no_false_success_from_lift_alone(invalid):
    score = GraspScore("red_cube")
    for time in (0, 1, 2, 3):
        sample = supported_sample(time)
        red = sample["objects"]["red_cube"]
        if invalid == "airborne": red["jaw_contacts"] = {}
        if invalid == "one_jaw": red["jaw_contacts"] = {"right": [0]}
        if invalid == "table": red["table_contact"] = True
        if invalid == "suspended": sample["suspension_enabled"] = True
        if invalid == "wrong_object": sample["objects"]["blue_cube"]["bottom_above_table"] = .1
        assert not score.update(sample)


@pytest.mark.parametrize('parallel',[False,True,'dex1_cad'])
def test_tabletop_has_free_objects_and_preserves_sdk_layout(tmp_path,parallel):
    sonic = os.environ.get("SONIC_ROOT")
    if not sonic:
        pytest.skip("set SONIC_ROOT for the real scene physics test")
    sonic = Path(sonic)
    task = yaml.safe_load(Path("robot_client/sonic/tabletop.yaml").read_text())
    if parallel:task.update(gripper_type='parallel',jaw_root_x=.1,grasp_center_x=.15)
    if parallel=='dex1_cad':task.update(gripper_type='dex1_cad',grasp_center_x=.16009)
    scene = tmp_path / "tabletop.xml"
    prepare_scene(sonic, sonic / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml", scene, True, task)
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert model.nu == 43 and model.nq == 64
    assert model.neq == 0
    np.testing.assert_array_equal(model.actuator_trnid[:, 0], np.arange(1, 44))
    if parallel:
        for side in ('left','right'):
            for i in (0,1):
                joint=model.joint(f'{side}_hand_thumb_{i}_joint').id
                actuator=list(model.actuator_trnid[:,0]).index(joint)
                assert model.jnt_type[joint]==mujoco.mjtJoint.mjJNT_SLIDE
                np.testing.assert_equal(model.actuator_ctrlrange[actuator],[-8,8])
    enable_task_contacts(model, task)
    # Broadphase uses body masks cached at compilation, not only geom masks.
    assert model.body_conaffinity[model.body('red_cube').id] & 1
    assert model.body_conaffinity[model.body('task_table').id] & 1
    # This layout/settling test has no standing controller. Keep its falling
    # unpowered robot from knocking the free objects off the table.
    task_bodies=[0,model.body('task_table').id]+[model.body(o['name']).id for o in task['objects']]
    robot_bodies=~np.isin(np.arange(model.nbody),task_bodies)
    robot_geoms=robot_bodies[model.geom_bodyid]
    model.geom_contype[robot_geoms]=0
    model.geom_conaffinity[robot_geoms]=0
    model.body_contype[robot_bodies]=0
    model.body_conaffinity[robot_bodies]=0
    data = mujoco.MjData(model)
    # Controller FK and full dynamics must describe the same physical wrist;
    # floating-base motion is factored out in the pelvis-relative VLA contract.
    from robot_client.sonic.kinematics import G1Kinematics, JOINT_NAMES
    from scipy.spatial.transform import Rotation
    kin=G1Kinematics(sonic)
    q=np.zeros(29)
    q[[3,9]]=.2
    q[[18,25]]=.6
    q[14]=.12
    data.qpos[:7]=[.1,-.1,.78,*Rotation.from_euler('xyz',[.03,-.02,.1]).as_quat()[[3,0,1,2]]]
    for name,value in zip(JOINT_NAMES,q):data.qpos[model.jnt_qposadr[model.joint(name).id]]=value
    mujoco.mj_forward(model,data)
    base=np.eye(4)
    base[:3,:3]=data.xmat[model.body('pelvis').id].reshape(3,3)
    base[:3,3]=data.xpos[model.body('pelvis').id]
    for side,expected in zip(('left','right'),kin.forward(q)):
        wrist=np.eye(4)
        wrist[:3,:3]=data.xmat[model.body(f'{side}_wrist_yaw_link').id].reshape(3,3)
        wrist[:3,3]=data.xpos[model.body(f'{side}_wrist_yaw_link').id]
        np.testing.assert_allclose(np.linalg.inv(base)@wrist,expected,atol=1e-5)
    for _ in range(400):
        mujoco.mj_step(model, data)
    for obj in task["objects"]:
        assert model.joint(f"{obj['name']}_free").type == mujoco.mjtJoint.mjJNT_FREE
        assert data.xpos[model.body(obj["name"]).id, 2] == pytest.approx(task["table_height"]+obj["half_size"], abs=.002)
    config = load_config("robot_client/sonic/grasp_client.yaml")
    assert config["gripper_open_value"] == [5.5, 5.5]
    assert config["planner"]["max_speed"] == 0


def _gripper_bench_model(tmp_path,parallel=False,contact_solver=None):
    """Isolated gripper diagnostic, NOT learned-policy or SONIC task success.

    A vertical position servo supplies the arm trajectory; the two jaws remain
    dynamic and torque driven. The cube rises only through contact/friction.
    """
    sonic = os.environ.get("SONIC_ROOT")
    if not sonic:
        pytest.skip("set SONIC_ROOT for the physical gripper diagnostic")
    from scipy.spatial.transform import Rotation
    task = yaml.safe_load(Path("robot_client/sonic/tabletop.yaml").read_text())
    if parallel:task.update(gripper_type='parallel',jaw_root_x=.1,grasp_center_x=.15)
    if parallel=='dex1_cad':task.update(gripper_type='dex1_cad',grasp_center_x=.16009)
    if contact_solver:task['contact_solver']=contact_solver
    full = tmp_path / "full.xml"
    prepare_scene(Path(sonic), Path(sonic)/"gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml", full, task=task)
    source = ET.parse(full).getroot()
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep="0.002", iterations="80")
    if contact_solver:
        for key in ('cone','impratio'):root.find('option').set(key,source.find('option').get(key))
    if parallel=='dex1_cad':
        asset=ET.SubElement(root,'asset')
        for mesh in source.findall('asset/mesh'):
            if mesh.get('name','').startswith('unifolm_dex1_'):asset.append(copy.deepcopy(mesh))
    world = ET.SubElement(root, "worldbody")
    rotation = Rotation.from_euler("y", np.pi/3)
    wrist_pos = np.array([*task["objects"][0]["xy"], task["table_height"]+.125]) - rotation.apply([task.get('grasp_center_x',.14), 0, 0])
    hand = ET.SubElement(world, "body", name="bench_mount", gravcomp="1",
        pos=" ".join(map(str, wrist_pos)), quat=" ".join(map(str, rotation.as_quat()[[3,0,1,2]])))
    ET.SubElement(hand, "inertial", pos="0 0 0", mass="5", diaginertia=".1 .1 .1")
    ET.SubElement(hand, "joint", name="bench_lift", type="slide",
                  axis=" ".join(map(str, rotation.inv().apply([0,0,1]))))
    wrist = source.find(".//body[@name='right_wrist_yaw_link']")
    for element in wrist:
        if element.get("name", "").startswith("right_grasp_") or element.get('name')=='right_dex1_base':
            hand.append(copy.deepcopy(element))
    for name in ("task_table", "red_cube", "blue_cube"):
        body = copy.deepcopy(source.find(f".//body[@name='{name}']"))
        world.append(body)
    actuators = ET.SubElement(root, "actuator")
    for i in (0, 1):
        ET.SubElement(actuators, "motor", joint=f"right_hand_thumb_{i}_joint", ctrllimited="true",
                      ctrlrange="-8 8" if parallel else "-1.4 1.4")
    ET.SubElement(actuators, "position", joint="bench_lift", kp="3000", kv="100")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, task


def test_dex1_mesh_transforms_match_official_urdf_through_full_stroke(tmp_path):
    """Compare every moving part against an independently imported URDF.

    In particular Link2_3 has +29.4 mm Z, unlike Link1_3's -1 mm.
    A lift-only test cannot detect an asymmetric/misplaced contact pad.
    """
    from scipy.spatial.transform import Rotation
    from robot_client.sonic import dex1_cad
    folder=Path(dex1_cad.__file__).with_name('assets')/'dex1_1'
    urdf=ET.parse(folder/'dex1_1.urdf').getroot()
    urdf.find('mujoco/compiler').set('meshdir',str(folder.resolve()))
    reference=mujoco.MjModel.from_xml_string(ET.tostring(urdf,encoding='unicode'))
    rd=mujoco.MjData(reference)
    model,_=_gripper_bench_model(tmp_path,'dex1_cad');data=mujoco.MjData(model)
    rotation=Rotation.from_euler('z',-90,degrees=True).as_matrix()
    body=model.body('bench_mount').id
    def vertices(m,d,g):
        mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh])
        return m.mesh_vert[start:start+count]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
    for close in (0,.022,.0445):
        rd.qpos[:]=close-.02
        for i in (0,1):data.qpos[model.jnt_qposadr[model.joint(f'right_hand_thumb_{i}_joint').id]]=close
        mujoco.mj_forward(reference,rd);mujoco.mj_forward(model,data)
        for i,number in enumerate((2,1)):
            for part,index in (('carriage',1),('finger',2),('pad',3)):
                name=f'Link{number}_{index}'
                rg=next(g for g in range(reference.ngeom) if reference.mesh(reference.geom_dataid[g]).name==name)
                expected=vertices(reference,rd,rg)@rotation.T+[.04,0,0]
                actual=vertices(model,data,model.geom(f'right_grasp_{part}_{i}').id)
                actual=(actual-data.xpos[body])@data.xmat[body].reshape(3,3)
                np.testing.assert_allclose(actual.min(0),expected.min(0),atol=2e-6)
                np.testing.assert_allclose(actual.max(0),expected.max(0),atol=2e-6)


def test_tip_grasp_creep_reduced_by_contact_solver_and_release_stays_free(tmp_path):
    """Reproduce observed tip grasp/load; distinguish creep from weak friction.

    Both cases use identical jaws, forces, coefficients, free cube and motion.
    Neither uses the NoSlip postprocessor or an object attachment.
    """
    from scipy.spatial.transform import Rotation
    rotation=Rotation.from_matrix([[.297,.949,-.103],[-.828,.202,-.523],[-.476,.240,.846]])
    drops=[]
    for index,settings in enumerate((None,{'cone':'elliptic','impratio':10})):
        folder=tmp_path/str(index);folder.mkdir()
        model,task=_gripper_bench_model(folder,True,settings)
        body=model.body('bench_mount').id
        model.body_quat[body]=rotation.as_quat()[[3,0,1,2]]
        model.body_pos[body]=np.r_[task['objects'][0]['xy'],task['table_height']+.125]-rotation.apply([.19,0,0])
        model.jnt_axis[model.joint('bench_lift').id]=rotation.inv().apply([0,0,1])
        assert model.opt.noslip_iterations==0 and model.neq==0
        data=mujoco.MjData(model);red=model.body('red_cube').id
        heights={}
        for step in range(9000):
            now=step*.002
            if step==500:enable_task_contacts(model,task)
            data.ctrl[2]=-.10*np.clip(now-1,0,1)+.12*np.clip(now-3,0,1)
            for i in (0,1):
                j=model.joint(f'right_hand_thumb_{i}_joint').id
                desired=.045 if 2<=now<15 else 0
                data.ctrl[i]=np.clip(250*(desired-data.qpos[model.jnt_qposadr[j]])-3*data.qvel[model.jnt_dofadr[j]],-8,8)
            mujoco.mj_step(model,data)
            if step in (2500,6500):heights[now]=float(data.xpos[red,2])
        drops.append(heights[5.]-heights[13.])
        assert data.xpos[red,2]==pytest.approx(task['table_height']+.025,abs=.003)
    assert drops[0]>.01
    assert abs(drops[1])<.003


@pytest.mark.parametrize('parallel',[False,True,'dex1_cad'])
def test_physical_jaws_lift_and_release_without_object_attachment(tmp_path,parallel):
    from robot_client.sonic.tactile import jaw_forces
    model, task = _gripper_bench_model(tmp_path,parallel)
    data = mujoco.MjData(model)
    red = model.body("red_cube").id
    red_geom = model.geom("red_cube_geom").id
    supported = []
    tactile_support=[]
    for step in range(4000):
        t = step * .002
        if step == 500:
            # Exercise runtime restoration, including cached broadphase masks.
            enable_task_contacts(model, task)
        data.ctrl[2] = -.10*np.clip((t-1)/1, 0, 1) + .12*np.clip((t-3)/1, 0, 1)
        desired = (.0445 if parallel=='dex1_cad' else .045 if parallel else .65) if 2 <= t < 6 else 0
        for i in (0, 1):
            j = model.joint(f"right_hand_thumb_{i}_joint").id
            kp,kd,limit=(250,3,8) if parallel else (1.5,.1,1.4)
            data.ctrl[i] = np.clip(kp*(desired-data.qpos[model.jnt_qposadr[j]])-kd*data.qvel[model.jnt_dofadr[j]], -limit, limit)
        mujoco.mj_step(model, data)
        if 4 <= t < 6:
            jaws = set()
            for c in data.contact:
                if red_geom not in (c.geom1, c.geom2): continue
                other = c.geom2 if c.geom1 == red_geom else c.geom1
                name = model.geom(other).name or ""
                for i in (0, 1):
                    if name in (f"right_grasp_finger_{i}", f"right_grasp_pad_{i}"): jaws.add(i)
            bottom = data.xpos[red, 2] - .025 - task["table_height"]
            supported.append(bottom > .08 and len(jaws) == 2)
            tactile_support.append(np.all(jaw_forces(model,data)[1]>.05))
    assert sum(supported) / len(supported) > .98
    assert sum(tactile_support) / len(tactile_support) > .98
    assert model.neq == 0
    # Opening removes support and the unconstrained cube falls back to the table.
    assert data.xpos[red, 2] == pytest.approx(task["table_height"]+.025, abs=.003)


@pytest.mark.parametrize('task_name,kp,tolerance',[
    ('tabletop_rgb_green_firm.yaml',250,.002),
    ('tabletop_rgb_green_guarded.yaml',400,.004),
])
def test_dex1_loaded_oblique_grasp_has_small_creep_and_releases(tmp_path,task_name,kp,tolerance):
    """Real CAD, observed oblique wrist/load, free cube and dynamic jaws.

    The mount has a servoed degree of freedom so automatic parent collision
    filtering matches the robot; welding it to world creates spurious palm/
    carriage self contacts. This is a contact regression, not VLA success.
    """
    from scipy.spatial.transform import Rotation
    config=yaml.safe_load((Path('robot_client/sonic')/task_name).read_text())
    model,task=_gripper_bench_model(tmp_path,'dex1_cad',config['contact_solver'])
    rotation=Rotation.from_euler('xyz',[-1.977796,20.263323,-63.419162],degrees=True)
    mount=model.body('bench_mount').id
    tcp=np.r_[task['objects'][0]['xy'],task['table_height']+.15]
    model.body_quat[mount]=rotation.as_quat()[[3,0,1,2]]
    model.body_pos[mount]=tcp-rotation.apply([.16009,0,0])
    model.jnt_axis[model.joint('bench_lift').id]=rotation.inv().apply([0,0,1])
    red=model.body('red_cube').id;geom=model.geom('red_cube_geom').id
    model.geom_size[geom]=.02;model.body_mass[red]=.06;model.body_inertia[red]=.06*.04**2/6
    data=mujoco.MjData(model)
    free=model.jnt_qposadr[model.joint('red_cube_free').id]
    data.qpos[free:free+3]=tcp+rotation.apply([.01010901,-.00010179,-.00982946])
    joints=[model.joint(f'right_hand_thumb_{i}_joint').id for i in (0,1)]
    adr=model.jnt_qposadr[joints];dof=model.jnt_dofadr[joints]
    data.qpos[adr]=[.02411306,.02430725]
    enable_task_contacts(model,task)
    assert model.opt.noslip_iterations==0 and model.neq==0
    heights={}
    for step in range(7000):
        data.ctrl[:2]=np.clip(kp*((.0445 if step<6000 else 0)-data.qpos[adr])-3*data.qvel[dof],-8,8)
        data.ctrl[2]=0
        mujoco.mj_step(model,data)
        if step in (1000,5999,6999):heights[step]=float(data.xpos[red,2])
    assert abs(heights[1000]-heights[5999])<tolerance
    assert heights[5999]-heights[6999]>.08
    assert heights[6999]==pytest.approx(task['table_height']+.02,abs=.003)
