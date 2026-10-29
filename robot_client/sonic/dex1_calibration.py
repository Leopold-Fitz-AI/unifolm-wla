"""Verified tool/motor conventions; no object state or task policy logic."""
import numpy as np

TOOL_OFFSETS = [[.105, 0., 0., 0., 0., 0.], [.105, 0., 0., 0., 0., 0.]]
MOTOR_OPEN_RAD = 5.4
MOTOR_CLOSED_RAD = 0.
JAW_STROKE_M = .0445


def validate_dex1_calibration(config):
    """Require the documented endpoints when this calibration is selected.

    0 is the official encoder closed endpoint. Collision-mesh clearance is
    not an encoder zero measurement and must not be used to change that value.
    Normalization/action-range limits may extend beyond the physical endpoint;
    execution saturates those requests at the declared mechanical limits.
    """
    expected = {
        'tool_offsets': TOOL_OFFSETS,
        'gripper_open_value': [MOTOR_OPEN_RAD] * 2,
        'gripper_closed_value': [MOTOR_CLOSED_RAD] * 2,
        'hand_open': [[0.] * 7] * 2,
        'hand_closed': [[JAW_STROKE_M, JAW_STROKE_M, 0., 0., 0., 0., 0.]] * 2,
    }
    for key, value in expected.items():
        actual = np.asarray(config.get(key), dtype=float)
        reference = np.asarray(value)
        if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.allclose(actual, reference, rtol=0., atol=1e-8):
            raise ValueError(f'verified Dex1 calibration requires {key}={value}')
    return {
        'name': 'unitree_g1_dex1_tool105_motor54_v1',
        'tool_definition': 'dataset end-effector frame expressed in pelvis; not CAD base_link or pad center',
        'tool_offsets': TOOL_OFFSETS,
        'tool_reference': 'unitree_ros@deeb607f3a8945a394f53068c28b3a88dde7ff06 g1_29dof_mode_15_with_dex1_1.urdf plus official joint/EE observations',
        'execution_kinematics': 'actual SONIC URDF retained; training reference model is calibration-only',
        'motor_unit': 'radian after reduction gear',
        'motor_closed_rad': MOTOR_CLOSED_RAD, 'motor_open_rad': MOTOR_OPEN_RAD,
        'official_nominal_open_aperture_m': .09, 'simulated_jaw_stroke_m': JAW_STROKE_M,
        'motor_reference': 'https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_hand_unitree.py',
        'geometry_limitations': 'Existing independent Dex1 CAD and assumed 40 mm mount retained. Its pad gap is about 90.063 mm open and 1.063 mm at the closed stop; mesh clearance is not an encoder zero calibration. Linear simulated aperture gain is approximate.',
    }
