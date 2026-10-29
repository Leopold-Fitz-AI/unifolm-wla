"""Official observations, held-out postures, and declared motor endpoints."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from robot_client.sonic.calibrate_tools import REFERENCE_URDF, fit_observations
from robot_client.sonic.dex1_calibration import TOOL_OFFSETS, validate_dex1_calibration

FIXTURE = Path(__file__).with_name('fixtures') / 'g1_dex1_observations.npz'


def observations():
    return dict(np.load(FIXTURE))


def test_official_observations_identify_105mm_tool_on_both_sides():
    result = fit_observations(observations())
    np.testing.assert_allclose(result['tool_offsets'], TOOL_OFFSETS, atol=2e-8)
    for residual in result['residuals']:
        assert residual['validation_pairs'] >= 40
        assert residual['validation_position_max_m'] < 3e-8
        assert residual['validation_rotation_max_rad'] < 2e-7
    assert 'pad center' in result['tool_definition']


def test_wrong_wrist_model_cannot_be_hidden_in_an_average_tcp(tmp_path):
    tree = ET.parse(REFERENCE_URDF)
    # The actual SONIC model has 46 mm here; the dataset model uses 51 mm.
    tree.getroot().find("joint[@name='left_wrist_yaw_joint']/origin").set('xyz', '.046 0 0')
    wrong = tmp_path / 'wrong-wrist.urdf'
    tree.write(wrong)
    with pytest.raises(ValueError, match='do not support a rigid tool'):
        fit_observations(observations(), wrong)
    diagnostic = fit_observations(observations(), wrong, enforce=False)
    assert not diagnostic['residuals'][0]['rigid_tool_supported']
    assert diagnostic['residuals'][0]['validation_position_max_m'] > .001


def test_held_out_pose_error_is_rejected():
    rows = observations()
    rows['left_ee_pose_gripper_base'][1::2, 2] += .005
    with pytest.raises(ValueError, match='do not support a rigid tool'):
        fit_observations(rows)


def test_reference_sources_have_pinned_hashes():
    manifest = json.loads(REFERENCE_URDF.with_name('SOURCE.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256(REFERENCE_URDF.with_name(name).read_bytes()).hexdigest() == expected
    source = json.loads(FIXTURE.with_suffix('.source.json').read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == source['fixture_sha256']


def calibrated_config():
    return {'tool_offsets': copy.deepcopy(TOOL_OFFSETS), 'gripper_open_value': [5.4, 5.4],
            'gripper_closed_value': [0., 0.], 'hand_open': [[0.] * 7] * 2,
            'hand_closed': [[.0445, .0445, 0., 0., 0., 0., 0.]] * 2,
            'gripper_range': [[0., 5.5], [0., 5.5]]}


def test_normalization_bound_does_not_override_physical_motor_endpoint():
    config = calibrated_config()
    report = validate_dex1_calibration(config)
    assert report['motor_closed_rad'] == 0 and report['motor_open_rad'] == 5.4
    assert 'approximate' in report['geometry_limitations']
    assert config['gripper_range'] == [[0., 5.5], [0., 5.5]]


@pytest.mark.parametrize('key,value', [
    ('gripper_open_value', [5.5, 5.5]),
    ('gripper_closed_value', [.064, .064]),
    ('tool_offsets', [[.10619, .00605, -.00018, 0, 0, 0], [.1087, -.00497, .00027, 0, 0, 0]]),
])
def test_wrong_calibration_or_mesh_derived_encoder_zero_fails_closed(key, value):
    config = calibrated_config()
    config[key] = value
    with pytest.raises(ValueError, match=key):
        validate_dex1_calibration(config)
