"""Contact audit must include initialization and transient ready collisions."""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

from robot_client.sonic.grasp import TableCollisionAudit, GraspMonitor
from robot_client.sonic.launch_sim import prepare_scene


def fixture(enabled=True):
    model=mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".005"/>
      <custom><numeric name="unifolm_robot_collision_geoms" data="0"/></custom>
      <worldbody>
        <body name="pelvis" pos="0 0 1.04"><freejoint/>
          <geom name="left_grasp_finger_0" type="sphere" size=".02" mass=".1" contype="1" conaffinity="1"/>
          <geom name="visual_only" type="sphere" size=".021" mass="0" group="2" contype="0" conaffinity="0"/>
        </body>
        <body name="task_table"><geom name="task_table_top" type="box" pos="0 0 .975" size=".5 .5 .025"
          contype="4" conaffinity="{3 if enabled else 2}"/></body>
        <body name="red_cube" pos=".2 0 .999"><freejoint/>
          <geom name="red_cube_geom" type="sphere" size=".02" mass=".1" contype="2" conaffinity="6"/>
        </body>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    audit=TableCollisionAudit(model,{'robot_contacts_from_start':enabled})
    return model,data,audit


def test_audit_catches_initial_penetration_and_ignores_object_table_contacts():
    model,data,audit=fixture()
    audit.update(data,'initialization')
    assert audit.report()['passed']  # Cube/table contact is allowed.
    data.qpos[2]=1.018
    mujoco.mj_forward(model,data)
    audit.update(data,'initialization')
    report=audit.report()
    assert not report['passed']
    assert report['maximum_penetration_m']==pytest.approx(.002)
    assert report['worst_contact']['phase']=='initialization'
    assert set(report['worst_contact']['geom_pair'])=={'task_table_top','left_grasp_finger_0'}


def test_ready_collision_between_logged_frames_remains_a_failure():
    model,data,audit=fixture()
    audit.update(data,'initialization')
    monitor=GraspMonitor.__new__(GraspMonitor)
    monitor.env=SimpleNamespace(mj_model=model,mj_data=data)
    monitor.episode_file=None;monitor.episode={'status':'warming','phase':'ready'}
    monitor.task={'robot_contacts_from_start':True}
    monitor.finalized=False;monitor.started=False;monitor.last_sample=0
    monitor.table_audit=audit
    data.time=.005;data.qpos[2]=1.018
    mujoco.mj_forward(model,data);monitor.sample()
    data.time=.010;data.qpos[2]=1.04
    mujoco.mj_forward(model,data);monitor.sample()
    report=audit.report()
    assert report['checked_physics_steps']==3
    assert report['complete_step_coverage']
    assert not report['passed']
    assert report['first_violation']['simulation_time']==.005
    assert report['phases']['ready']['max_penetration_m']==pytest.approx(.002)
    assert monitor.last_sample==0  # Neither step reaches the 50 Hz log.


def test_disabled_contacts_are_not_misreported_as_collision_free():
    model,data,audit=fixture(False)
    data.qpos[2]=1.018;mujoco.mj_forward(model,data)
    audit.update(data,'initialization')
    report=audit.report()
    assert report['maximum_penetration_m']==0
    assert report['disabled_steps']==1 and not report['passed']


@pytest.mark.parametrize('time',[.02,-.005])
def test_missing_physics_steps_or_reset_invalidates_coverage(time):
    _,data,audit=fixture()
    audit.update(data,'initialization');data.time=time;audit.update(data,'warming')
    assert not audit.report()['complete_step_coverage']
    assert not audit.report()['passed']


@pytest.mark.parametrize('enabled',[False,True])
def test_compiled_scene_starts_with_requested_robot_contacts(tmp_path,enabled):
    sonic=os.environ.get('SONIC_ROOT')
    if not sonic:pytest.skip('SONIC_ROOT required for complete CAD scene')
    sonic=Path(sonic)
    task=yaml.safe_load(Path('robot_client/sonic/tabletop_rgb_green_guarded.yaml').read_text())
    task['robot_contacts_from_start']=enabled
    scene=tmp_path/'scene.xml'
    prepare_scene(sonic,sonic/'gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml',scene,task=task)
    model=mujoco.MjModel.from_xml_path(str(scene))
    for name in ('task_table','red_cube','green_cube','yellow_cube'):
        body=model.body(name).id
        assert bool(model.body_conaffinity[body]&1)==enabled
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom]==body:assert bool(model.geom_conaffinity[geom]&1)==enabled
    audit=TableCollisionAudit(model,task)
    assert audit.manifest_valid and len(audit.required_robot_geoms)>0
    if enabled:
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        audit.update(data,'initialization')
        assert audit.report()['disabled_steps']==0


@pytest.mark.parametrize('kind',['geom','body','global'])
@pytest.mark.parametrize('before_constructor',[True,False])
def test_robot_side_or_global_disable_cannot_pass(kind,before_constructor):
    model,data,audit=fixture()
    if not before_constructor:
        audit.update(data,'initialization')
        assert audit.report()['passed']
        data.time=.005
    robot=model.geom('left_grasp_finger_0').id
    if kind=='geom':
        model.geom_contype[robot]=0;model.geom_conaffinity[robot]=0
    elif kind=='body':
        body=model.geom_bodyid[robot]
        model.body_contype[body]=0;model.body_conaffinity[body]=0
    else:model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    if before_constructor:
        audit=TableCollisionAudit(model,{'robot_contacts_from_start':True})
    mujoco.mj_forward(model,data)
    audit.update(data,'initialization' if before_constructor else 'running')
    assert audit.report()['required_robot_collision_geoms']==1
    assert audit.report()['disabled_steps']==1
    assert not audit.report()['passed']


def test_missing_collider_manifest_fails_closed():
    model,data,_=fixture()
    # The compiled inventory must survive a fully disabled startup. Conversely,
    # an empty inventory is insufficient evidence, even when no contacts exist.
    model.numeric_size[0]=0
    audit=TableCollisionAudit(model,{'robot_contacts_from_start':True})
    audit.update(data,'initialization')
    assert not audit.report()['collider_manifest_valid']
    assert not audit.report()['passed']


def test_safe_monitor_does_not_reenable_masks_before_inspecting_them():
    model,data,audit=fixture()
    audit.update(data,'initialization')
    monitor=GraspMonitor.__new__(GraspMonitor)
    monitor.env=SimpleNamespace(mj_model=model,mj_data=data)
    monitor.episode_file=None;monitor.episode={'status':'running'}
    monitor.task={'robot_contacts_from_start':True}
    monitor.finalized=False;monitor.started=False;monitor.last_sample=0
    monitor.table_audit=audit
    table=model.body('task_table').id
    model.body_conaffinity[table]=2
    model.geom_conaffinity[model.geom('task_table_top').id]=2
    data.time=.005;mujoco.mj_forward(model,data)
    monitor.sample()
    assert model.body_conaffinity[table]==2
    assert audit.report()['disabled_steps']==1
    assert not audit.report()['passed']


def test_collider_manifest_does_not_change_dynamics(tmp_path):
    sonic=os.environ.get('SONIC_ROOT')
    if not sonic:pytest.skip('SONIC_ROOT required for complete CAD scene')
    sonic=Path(sonic)
    task=yaml.safe_load(Path('robot_client/sonic/tabletop_rgb_green_safe.yaml').read_text())
    scene=tmp_path/'scene.xml'
    prepare_scene(sonic,sonic/'gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml',scene,task=task)
    audited=mujoco.MjModel.from_xml_path(str(scene))
    tree=ET.parse(scene);custom=tree.getroot().find('custom')
    custom.remove(custom.find("numeric[@name='unifolm_robot_collision_geoms']"))
    comparison=tmp_path/'without-manifest.xml';tree.write(comparison,encoding='unicode')
    original=mujoco.MjModel.from_xml_path(str(comparison))
    for field in ('nq','nv','nu','ngeom','nmesh','nbody','neq'):
        assert getattr(audited,field)==getattr(original,field)
    for field in ('geom_contype','geom_conaffinity','body_contype','body_conaffinity',
                  'geom_pos','geom_quat','geom_size','mesh_vert','body_mass','body_inertia',
                  'jnt_range','actuator_gainprm','qpos0'):
        np.testing.assert_array_equal(getattr(audited,field),getattr(original,field))
    left,right=mujoco.MjData(audited),mujoco.MjData(original)
    for _ in range(50):
        mujoco.mj_step(audited,left);mujoco.mj_step(original,right)
    np.testing.assert_array_equal(left.qpos,right.qpos)
    np.testing.assert_array_equal(left.qvel,right.qvel)
