"""Official Unitree Dex1-1 mesh geometry in the existing 7-channel SDK slots.

The wrist adapter is explicit; meshes/kinematic offsets are from Unitree's
BSD-licensed dex1_1 URDF. Two coupled commands are sent to independent sliding
finger dynamics; five inertial dummy joints preserve SONIC's Dex3 SDK layout.
"""
from pathlib import Path
import xml.etree.ElementTree as ET


def add_dex1_assets(root):
    folder=Path(__file__).with_name('assets')/'dex1_1'
    asset=root.find('asset')
    if asset is None:asset=ET.SubElement(root,'asset')
    for name in ('base_link','Link1_1','Link1_2','Link1_3','Link2_1','Link2_2','Link2_3'):
        path=folder/f'{name}.STL'
        if not path.is_file():raise FileNotFoundError(f'missing official Dex1 asset: {path}')
        ET.SubElement(asset,'mesh',name=f'unifolm_dex1_{name}',file=str(path.resolve()))


def add_dex1_hand(wrist,side,names,mount_x=.04,finger_rgba='.89804 .91765 .92941 1'):
    # URDF +Y is forward and +X opens the opposing fingers. Rotate to G1 +X
    # forward, retaining the native wrist yaw frame used by the learned EE.
    base=ET.SubElement(wrist,'body',name=f'{side}_dex1_base',pos=f'{mount_x} 0 0',
                       quat='0.7071067811865476 0 0 -0.7071067811865476')
    ET.SubElement(base,'geom',name=f'{side}_grasp_palm',type='mesh',mesh='unifolm_dex1_base_link',
                  mass='.11',rgba='.79216 .81961 .93333 1')
    joints=[]
    for i,name in enumerate(names):
        if i<2:
            number=2 if i==0 else 1
            sign=-1 if number==2 else 1
            # Encode the URDF open endpoint q=-.02 as SDK position zero.
            body=ET.SubElement(base,'body',name=f'{side}_grasp_link_{i}',
                pos=f'{sign*.06055} .0593 {sign*.0152}')
            ET.SubElement(body,'joint',name=name,type='slide',axis=f'{-sign} 0 0',
                range='0 .0445',damping='1',armature='.002',frictionloss='.01',actuatorfrcrange='-8 8')
            joints.append(name)
            ET.SubElement(body,'geom',name=f'{side}_grasp_carriage_{i}',type='mesh',
                mesh=f'unifolm_dex1_Link{number}_1',mass='.0084398',rgba='.79216 .81961 .93333 1')
            ET.SubElement(body,'geom',name=f'{side}_grasp_finger_{i}',type='mesh',
                mesh=f'unifolm_dex1_Link{number}_2',pos=f'{-sign*.013} .013 0',mass='.028354',
                rgba=finger_rgba,friction='1.2 .005 .0001',condim='4')
            ET.SubElement(body,'geom',name=f'{side}_grasp_pad_{i}',type='mesh',
                mesh=f'unifolm_dex1_Link{number}_3',pos=f'{-sign*.0155165} .038043 {-.001 if number==1 else .0294}',
                mass='.004583',rgba='.29804 .29804 .29804 1',friction='2 .005 .0001',condim='4')
        else:
            body=ET.SubElement(base,'body',name=f'{side}_grasp_link_{i}')
            ET.SubElement(body,'inertial',pos='0 0 0',mass='.002',diaginertia='1e-6 1e-6 1e-6')
            ET.SubElement(body,'joint',name=name,axis='0 0 1',range='-.0001 .0001',
                damping='.1',armature='.002',actuatorfrcrange='-.1 .1')
    return joints
