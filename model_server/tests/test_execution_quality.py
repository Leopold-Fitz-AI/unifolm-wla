import json
import numpy as np
from robot_client.sonic.execution_quality import measure_window


def records(speed=.1,actual_speed=.1,yaw=np.pi/2):
    times=np.arange(0.,4.01,.1)
    control=[{'simulation_time':float(t),'command_base':[speed,0,0,.75],
              'planner_velocity':[speed,0]} for t in times]
    physics=[{'simulation_time':float(t),'base_position':[np.cos(yaw)*actual_speed*t,
        np.sin(yaw)*actual_speed*t,.75],'base_quat_wxyz':[np.cos(yaw/2),0,0,np.sin(yaw/2)],
        'base_tilt_deg':0.} for t in times]
    return control,physics


def test_body_velocity_is_integrated_in_world_heading():
    result=measure_window(*records())
    np.testing.assert_allclose(result['actual_base_displacement_m'],[0,.4,0],atol=1e-10)
    assert result['maximum_error_to_model_xy_m']<1e-10


def test_stationary_command_does_not_hide_real_drift():
    result=measure_window(*records(speed=0.,actual_speed=.02,yaw=0),seconds=3.)
    assert result['full_requested_window']
    json.dumps(result,allow_nan=False)
    np.testing.assert_allclose(result['final_error_to_model_xy_m'],[.06,0])
    assert abs(result['maximum_error_to_planner_xy_m']-.06)<1e-10


def test_short_and_reset_recordings_do_not_become_full_windows():
    control,physics=records()
    result=measure_window(control[:15],physics,3.)
    assert not result['full_requested_window']
    control[5]['simulation_time']=0.
    assert not measure_window(control,physics)['available']
