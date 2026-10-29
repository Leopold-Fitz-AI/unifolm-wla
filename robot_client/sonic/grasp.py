"""Physical tabletop fixture and independent grasp scoring for SONIC.

No object pose is sent to the learned policy. Objects have free joints and are
held only by contact/friction; there are no welds or attachment shortcuts.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROBOT_COLLISION_MANIFEST='unifolm_robot_collision_geoms'


def robot_geom_ids(model):
    """Robot membership is structural, independent of mutable collision masks."""
    pelvis=model.body('pelvis').id
    bodies=set()
    for body in range(model.nbody):
        ancestor=body
        while ancestor:
            if ancestor==pelvis:
                bodies.add(body);break
            ancestor=int(model.body_parentid[ancestor])
    return np.flatnonzero(np.isin(model.geom_bodyid,list(bodies)))


def tabletop_scene(path, task):
    """Keep the SDK's 29+7+7 joint/actuator order while replacing Dex3 fingers.

    The first two hand channels drive opposed jaws. Five unloaded joints
    preserve the upstream DDS layout. Fixtures include simple proxies and
    official Dex1 CAD with an explicitly assumed wrist mounting transform.
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    # Explicit numerical contact model, kept in the scene/report for A/B tests.
    # Harder tangential impedance reduces soft-contact creep without raising
    # friction coefficients or adding object constraints/forces.
    contact_solver=task.get('contact_solver',{})
    if contact_solver:
        cone=contact_solver.get('cone','elliptic')
        impratio=float(contact_solver.get('impratio',10))
        if set(contact_solver)-{'cone','impratio'} or cone not in ('elliptic','pyramidal') or not np.isfinite(impratio) or not 1<=impratio<=100:
            raise ValueError('contact_solver requires cone elliptic/pyramidal and impratio in [1,100]')
        option=root.find('option')
        if option is None:option=ET.SubElement(root,'option')
        option.set('cone',cone)
        option.set('impratio',str(impratio))
    jaw_root_x = float(task.get("jaw_root_x", .08))
    gripper_type = task.get('gripper_type', 'hinged')
    if gripper_type not in ('hinged','parallel','dex1_cad'):
        raise ValueError('gripper_type must be hinged, parallel or dex1_cad')
    parallel = gripper_type in ('parallel','dex1_cad')
    if gripper_type=='dex1_cad':
        from .dex1_cad import add_dex1_assets,add_dex1_hand
        add_dex1_assets(root)
    parallel_joints = set()
    for side in ("left", "right"):
        wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        names = [j.get("name") for j in wrist.findall(".//joint") if j.get("name", "").startswith(f"{side}_hand_")]
        if len(names) != 7 or any(not n.startswith(f"{side}_hand_") for n in names):
            raise ValueError("unexpected upstream hand joint layout")
        for body in list(wrist.findall("body")):
            wrist.remove(body)
        # The Dex3 palm reaches x=.128 m; keeping its collision mesh would
        # occupy a 5 cm object's intended grasp volume around TCP x=.14 m.
        for geom in list(wrist.findall("geom")):
            if geom.get("mesh") == f"{side}_hand_palm_link":
                wrist.remove(geom)
        if gripper_type=='dex1_cad':
            parallel_joints.update(add_dex1_hand(wrist,side,names,task.get('dex1_mount_x',.04),
                task.get('dex1_finger_rgba','.89804 .91765 .92941 1')))
            continue
        ET.SubElement(wrist, "geom", name=f"{side}_grasp_palm", type="box",
                      pos="0.06 0 0", size="0.022 0.04 0.022", mass="0", rgba="0.7 0.7 0.75 1")
        for i, name in enumerate(names):
            body = ET.SubElement(wrist, "body", name=f"{side}_grasp_link_{i}",
                                 pos=f"{jaw_root_x} {(.055 if i == 0 else -.055) if i < 2 else 0} 0")
            if i < 2:
                if parallel:
                    parallel_joints.add(name)
                    ET.SubElement(body, 'joint', name=name, type='slide', axis=f'0 {-1 if i == 0 else 1} 0',
                                  range='0 0.05', damping='1', armature='0.002',
                                  frictionloss='0.01', actuatorfrcrange='-8 8')
                else:
                    ET.SubElement(body, "joint", name=name, axis=f"0 0 {-1 if i == 0 else 1}",
                                  range="0 0.8", damping="0.05", armature="0.002",
                                  frictionloss="0.01", actuatorfrcrange="-1.4 1.4")
                ET.SubElement(body, "geom", name=f"{side}_grasp_finger_{i}", type="capsule",
                              fromto="0 0 0 0.08 0 0", size="0.008", mass="0.045",
                              rgba=task.get("finger_rgba", "0.7 0.7 0.75 1"), friction="1.2 0.005 0.0001", condim="4")
                ET.SubElement(body, "geom", name=f"{side}_grasp_pad_{i}", type="box",
                              pos=f"0.05 {-.006 if i == 0 else .006} 0", size="0.025 0.004 0.016",
                              mass="0.005", rgba="0.12 0.12 0.12 1", friction="2.0 0.005 0.0001", condim="4")
            else:
                ET.SubElement(body, "inertial", pos="0 0 0", mass="0.002", diaginertia="1e-6 1e-6 1e-6")
                ET.SubElement(body, "joint", name=name, axis="0 0 1", range="-0.0001 0.0001",
                              damping="0.1", armature="0.002", actuatorfrcrange="-0.1 0.1")
    if parallel:
        for actuator in root.find('actuator'):
            if actuator.get('joint') in parallel_joints:
                actuator.set('ctrllimited', 'true')
                actuator.set('ctrlrange', '-8 8')
    world = root.find("worldbody")
    if task.get("calibrated_lighting", False):
        headlight = root.find("visual/headlight")
        if headlight is not None:
            headlight.set("diffuse", ".15 .15 .15")
            headlight.set("ambient", ".45 .45 .45")
            headlight.set("specular", "0 0 0")
        for light in world.findall("light"):
            light.set("diffuse", ".15 .15 .15")
            light.set("ambient", ".15 .15 .15")
            light.set("specular", "0 0 0")
    # Upstream's red floor-level COM site is a debug overlay, not a task object.
    marker = world.find("site[@name='com_marker']")
    if marker is not None:
        world.remove(marker)
    floor = world.find("geom[@name='floor']")
    if floor is not None:
        floor.attrib.pop("material", None)
        floor.set("rgba", ".6 .6 .6 1")
    for texture in root.findall(".//texture[@type='skybox']"):
        texture.set("rgb1", ".8 .8 .8")
        texture.set("rgb2", ".65 .65 .65")
    height = float(task["table_height"])
    contacts_from_start=task.get('robot_contacts_from_start',False)
    if not isinstance(contacts_from_start,bool):
        raise ValueError('robot_contacts_from_start must be boolean')
    table_affinity='3' if contacts_from_start else '2'
    object_affinity='7' if contacts_from_start else '6'
    table = ET.SubElement(world, "body", name="task_table")
    ET.SubElement(table, "geom", name="task_table_top", type="box", pos=f"0.50 0 {height-.025}",
                  size="0.32 0.46 0.025", rgba="0.78 0.76 0.70 1", friction="0.8 0.005 0.0001",
                  contype="4", conaffinity=table_affinity)
    for x in (.25, .75):
        for y in (-.38, .38):
            ET.SubElement(table, "geom", type="box", pos=f"{x} {y} {(height-.05)/2}",
                          size=f"0.025 0.025 {(height-.05)/2}", rgba="0.32 0.32 0.34 1", contype="4", conaffinity=table_affinity)
    for obj in task["objects"]:
        pos = [*obj["xy"], height + obj["half_size"] + .001]
        body = ET.SubElement(world, "body", name=obj["name"], pos=" ".join(map(str, pos)))
        ET.SubElement(body, "freejoint", name=f"{obj['name']}_free")
        size = " ".join([str(obj["half_size"])] * 3)
        ET.SubElement(body, "geom", name=f"{obj['name']}_geom", type="box", size=size,
                      mass=str(obj["mass"]), rgba=" ".join(map(str, obj["rgba"])),
                      friction="1.2 0.005 0.0001", condim="4", solref="0.008 1", contype="2", conaffinity=object_affinity)
    # Persist the physical collider inventory at scene generation, before any
    # runtime code can disable a mask. A zero-mask visual geom is allowed, but
    # constructing an auditor after disabling physical colliders must not turn
    # those colliders into an empty/valid expected set.
    compiled=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    robot=robot_geom_ids(compiled)
    physical=robot[(compiled.geom_contype[robot]!=0)|(compiled.geom_conaffinity[robot]!=0)]
    if not len(physical):raise ValueError('scene has no physical robot collision geoms')
    custom=root.find('custom')
    if custom is None:custom=ET.SubElement(root,'custom')
    for item in list(custom):
        if item.get('name')==ROBOT_COLLISION_MANIFEST:custom.remove(item)
    ET.SubElement(custom,'numeric',name=ROBOT_COLLISION_MANIFEST,data=' '.join(map(str,physical)))
    tree.write(path, encoding="unicode")


class GraspScore:
    """A lift is valid only with bilateral jaw contact and sustained support."""
    def __init__(self, target, lift_height=.08, hold_seconds=2.0, required_hand=None):
        self.target = target
        self.lift_height = lift_height
        self.hold_seconds = hold_seconds
        self.since = None
        self.last_time = None
        self.success = False
        self.invalid_reset = False
        self.wrong_object = False
        self.best_hold = 0.0
        self.required_hand = required_hand

    def update(self, sample):
        now = sample["simulation_time"]
        if self.last_time is not None and now < self.last_time:
            self.invalid_reset = True
        self.last_time = now
        target = sample["objects"][self.target]
        self.wrong_object |= any(o["bottom_above_table"] > self.lift_height
                                 for name, o in sample["objects"].items() if name != self.target)
        hands = (self.required_hand,) if self.required_hand else ("left", "right")
        both_jaws = any(set(target["jaw_contacts"].get(hand, [])) == {0, 1} for hand in hands)
        valid = (not sample["suspension_enabled"] and sample["base_height"] > .5
                 and sample["base_tilt_deg"] < 25 and target["bottom_above_table"] >= self.lift_height
                 and both_jaws and not target["table_contact"] and not self.invalid_reset)
        if valid:
            self.since = now if self.since is None else self.since
            self.best_hold = max(self.best_hold, now - self.since)
            self.success |= self.best_hold >= self.hold_seconds and not self.wrong_object
        else:
            self.since = None
        return self.success and not self.invalid_reset and not self.wrong_object


def enable_task_contacts(model, task):
    """Restore both geom masks and MuJoCo's compiled body broadphase masks."""
    table = model.body("task_table").id
    model.geom_conaffinity[model.geom_bodyid == table] = 3
    model.body_conaffinity[table] = 3
    for obj in task["objects"]:
        model.geom_conaffinity[model.geom(f"{obj['name']}_geom").id] = 7
        model.body_conaffinity[model.body(obj["name"]).id] = 7


class TableCollisionAudit:
    """Independent every-physics-step table/robot audit, including boot/ready.

    Uses actual contacts rather than TCP height. The report is evaluator-only;
    it is never published to the robot or policy. Disabled collisions cannot be
    interpreted as zero penetration.
    """
    def __init__(self,model,task):
        self.model=model
        self.enabled_from_start=bool(task.get('robot_contacts_from_start',False))
        self.limit=float(task.get('table_penetration_limit_m',.001))
        if not np.isfinite(self.limit) or not 0<self.limit<=.005:
            raise ValueError('table_penetration_limit_m must be in (0,.005]')
        self.table_body=model.body('task_table').id
        self.table_geoms=set(np.flatnonzero(model.geom_bodyid==self.table_body).tolist())
        self.robot_geoms=set(robot_geom_ids(model).tolist())
        manifest=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_NUMERIC,ROBOT_COLLISION_MANIFEST)
        values=np.array([],float)
        if manifest>=0:
            start=model.numeric_adr[manifest];size=model.numeric_size[manifest]
            values=np.asarray(model.numeric_data[start:start+size]).copy()
        self.manifest_valid=bool(len(values) and np.isfinite(values).all() and
                                 np.all(values==np.floor(values)) and
                                 len(set(values))==len(values) and
                                 set(values).issubset(self.robot_geoms))
        self.required_robot_geoms=values.astype(int) if self.manifest_valid else np.array([],int)
        self.steps=0;self.disabled_steps=0;self.max_penetration=0.;self.worst=None
        self.first_violation=None;self.last_time=None;self.clock_reset=False
        self.max_gap=0.;self.phases={};self.invalid_contacts=0

    def update(self,data,phase):
        now=float(data.time)
        if self.last_time is not None:
            self.clock_reset |= now<self.last_time
            self.max_gap=max(self.max_gap,now-self.last_time)
        self.last_time=now;self.steps+=1
        model=self.model
        robot=self.required_robot_geoms
        table=np.asarray(sorted(self.table_geoms),int)
        body=model.geom_bodyid[robot]
        # Verify both narrowphase geom masks and compiled body broadphase
        # masks. Checking only the table side would miss disabled robot jaws.
        geom_pairs=((model.geom_contype[robot,None]&model.geom_conaffinity[table]) |
                    (model.geom_contype[table]&model.geom_conaffinity[robot,None]))
        body_pairs=((model.body_contype[body]&model.body_conaffinity[self.table_body]) |
                    (model.body_contype[self.table_body]&model.body_conaffinity[body]))
        contact_enabled=not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT))
        masks_enabled=bool(self.manifest_valid and contact_enabled and len(table) and
                           np.all(geom_pairs!=0) and np.all(body_pairs!=0))
        self.disabled_steps+=not masks_enabled
        penetration=0.;pair=None
        for index in range(data.ncon):
            contact=data.contact[index]
            first,second=int(contact.geom1),int(contact.geom2)
            if ((first in self.table_geoms and second in self.robot_geoms) or
                (second in self.table_geoms and first in self.robot_geoms)):
                distance=float(contact.dist)
                if not np.isfinite(distance):
                    self.invalid_contacts+=1;continue
                depth=max(0.,-distance)
                if depth>penetration:
                    penetration=depth;pair=[model.geom(first).name,model.geom(second).name]
        state=self.phases.setdefault(phase,{'checked_steps':0,'max_penetration_m':0.,'disabled_steps':0})
        state['checked_steps']+=1;state['disabled_steps']+=int(not masks_enabled)
        state['max_penetration_m']=max(state['max_penetration_m'],penetration)
        event={'simulation_time':now,'phase':phase,'penetration_m':penetration,'geom_pair':pair}
        if penetration>self.max_penetration:
            self.max_penetration=penetration;self.worst=event
        if penetration>self.limit and self.first_violation is None:self.first_violation=event

    def report(self):
        coverage=bool(self.steps>0 and not self.clock_reset and
                      self.max_gap<=float(self.model.opt.timestep)*1.5+1e-9)
        return {'passed':bool(self.enabled_from_start and not self.disabled_steps and not self.invalid_contacts and coverage
                              and self.max_penetration<=self.limit),
                'enabled_from_start':self.enabled_from_start,'checked_physics_steps':self.steps,
                'disabled_steps':self.disabled_steps,'complete_step_coverage':coverage,
                'maximum_step_gap_seconds':self.max_gap,'clock_reset':self.clock_reset,
                'invalid_contacts':self.invalid_contacts,
                'collider_manifest_valid':self.manifest_valid,
                'required_robot_collision_geoms':len(self.required_robot_geoms),
                'penetration_limit_m':self.limit,'maximum_penetration_m':self.max_penetration,
                'worst_contact':self.worst,'first_violation':self.first_violation,'phases':self.phases}


class GraspMonitor:
    def __init__(self, env, task, directory, episode_file=None):
        self.env, self.task = env, task
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.score = GraspScore(task["target"], task.get("lift_height", .08), task.get("hold_seconds", 2), task.get("required_hand"))
        self.last_sample = -1.0
        self.max_lift = {o["name"]: 0.0 for o in task["objects"]}
        self.samples = 0
        self.episode_file = Path(episode_file) if episode_file else None
        self.episode = {}
        self.started = False
        self.finalized = False
        self.stream = (self.directory / "physics.jsonl").open("w")
        self.table_audit=TableCollisionAudit(env.mj_model,task)
        # Include the initialized pose, before the first integration step.
        mujoco.mj_forward(env.mj_model,env.mj_data)
        self.table_audit.update(env.mj_data,'initialization')

    def sample(self):
        if self.finalized:
            return
        if self.episode_file and self.episode_file.exists():
            self.episode = json.loads(self.episode_file.read_text())
        if self.episode.get("status") == "running":
            # The standard SDK boot pose has hands below the table. Disable
            # robot/table collision only during reset positioning; object/table
            # contact stays active. Restore ALL table contacts before any policy
            # action is evaluated. No object pose or force is modified.
            if not self.task.get('robot_contacts_from_start',False):
                enable_task_contacts(self.env.mj_model, self.task)
            self.started = True
        # This precedes the 50 Hz recording throttle and completion handling:
        # a transient penetration between recorded samples must remain a failure.
        self.table_audit.update(self.env.mj_data,self.episode.get('phase',self.episode.get('status','initialization')))
        if self.started and self.episode.get("status") in ("passed", "failed"):
            self.finalized = True
            self.stream.flush()
            self.report()
            return
        model, data = self.env.mj_model, self.env.mj_data
        time = float(data.time)
        if time >= self.last_sample and time - self.last_sample < .02:
            return
        self.last_sample = time
        q = data.qpos[3:7].copy()
        import time as wall_clock
        record = {"wall_time": wall_clock.time(), "simulation_time": time, "base_height": float(data.qpos[2]),
                  "base_position": data.qpos[:3].tolist(), "base_quat_wxyz": q.tolist(),
                  "base_tilt_deg": float(np.rad2deg(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2), -1, 1)))),
                  "suspension_enabled": bool(self.env.elastic_band.enable), "objects": {}}
        record["hands"] = {}
        for side in ("left", "right"):
            wrist = model.body(f"{side}_wrist_yaw_link").id
            tcp = data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ [self.task.get("grasp_center_x", .14), 0, 0]
            record["hands"][side] = {"tcp_position": tcp.tolist(),
                "wrist_rotation": data.xmat[wrist].tolist(),
                "jaw_q": [float(data.qpos[model.jnt_qposadr[model.joint(f"{side}_hand_thumb_{i}_joint").id]]) for i in (0, 1)]}
        for obj in self.task["objects"]:
            geom = model.geom(f"{obj['name']}_geom").id
            body = model.body(obj["name"]).id
            # Project the oriented cube half extents onto world vertical.
            extent = obj["half_size"] * np.abs(data.xmat[body].reshape(3, 3)[2]).sum()
            bottom = float(data.xpos[body, 2] - extent - self.task["table_height"])
            jaws, table = {"left": set(), "right": set()}, False
            for i in range(data.ncon):
                contact = data.contact[i]
                if geom not in (contact.geom1, contact.geom2):
                    continue
                other = contact.geom2 if contact.geom1 == geom else contact.geom1
                name = model.geom(other).name or ""
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                if force[0] < .02:
                    continue
                table |= name == "task_table_top"
                for side in jaws:
                    for finger in (0, 1):
                        if name in (f"{side}_grasp_finger_{finger}", f"{side}_grasp_pad_{finger}"):
                            jaws[side].add(finger)
            record["objects"][obj["name"]] = {"position": data.xpos[body].tolist(),
                "bottom_above_table": bottom, "jaw_contacts": {s: sorted(v) for s, v in jaws.items()},
                "table_contact": table}
            self.max_lift[obj["name"]] = max(self.max_lift[obj["name"]], bottom)
        record["success"] = self.score.update(record) if self.started else False
        self.samples += 1
        self.stream.write(json.dumps(record) + "\n")
        if self.samples % 10 == 0:
            self.stream.flush()
            self.report()

    def report(self):
        achieved = self.score.success and not self.score.invalid_reset and not self.score.wrong_object
        collision=self.table_audit.report()
        collision_required=self.task.get('robot_contacts_from_start',False)
        result = {"success": achieved and self.episode.get("status") == "passed" and (not collision_required or collision['passed']),
                  "grasp_achieved": achieved, "episode": self.episode,
                  "target": self.task["target"], "instruction": self.task["instruction"],
                  "required_hand": self.score.required_hand,
                  "max_bottom_above_table_m": self.max_lift, "best_supported_hold_seconds": self.score.best_hold,
                  "wrong_object_lifted": self.score.wrong_object, "simulation_reset": self.score.invalid_reset,
                  "samples": self.samples, "lift_threshold_m": self.score.lift_height,
                  "hold_threshold_seconds": self.score.hold_seconds,
                  "gripper": "two opposed physical jaws, driven through first two SDK hand channels",
                  "gripper_type": self.task.get('gripper_type', 'hinged'),
                  "realtime_clock": self.task.get('realtime_clock', False),
                  "arm_tracking": self.task.get("arm_tracking", "sonic"),
                  "table_robot_collision": "enabled from initialization" if collision_required else "legacy: disabled during reset/ready positioning",
                  "object_robot_collision": "enabled from initialization" if collision_required else "legacy: disabled during reset/ready positioning",
                  "table_collision_audit":collision,
                  "attachments_or_welds": False}
        (self.directory / "task-report.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    def close(self):
        self.report()
        self.stream.close()
