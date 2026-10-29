import numpy as np
import pytest

from robot_client.sonic.base_hold import BasePositionHold
from robot_client.sonic.planner import PlannerMapper


def sensor(x=0.,y=0.,yaw=0.):
    return {'base_position':[x,y,.75],'base_quat':[np.cos(yaw/2),0,0,np.sin(yaw/2)]}


@pytest.mark.parametrize('yaw',[0.,np.pi/2,-np.pi/2])
def test_base_hold_commands_towards_anchor_in_robot_frame(yaw):
    hold=BasePositionHold()
    hold.command([0,0,0,.75],sensor())
    base=hold.command([0,0,0,.75],sensor(x=-.1,yaw=yaw))
    world=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])@base[:2]
    np.testing.assert_allclose(world,[.12,0],atol=1e-9)
    assert base[3]==.75
    assert hold.max_error==pytest.approx(.1)


def test_position_hysteresis_and_planner_actually_execute_correction():
    hold=BasePositionHold()
    hold.command([0,0,0,.75],sensor())
    assert not hold.command([0,0,0,.75],sensor(.01))[:2].any()
    corrected=hold.command([0,0,0,.75],sensor(.03))
    assert corrected[0]<0
    planner=PlannerMapper(max_speed=.12,max_yaw_rate=0)
    state=sensor()
    for _ in range(20):fields=planner.fields(np.zeros(29),np.zeros(29),corrected,state,.02,np.zeros((2,7)))
    assert fields['mode'][0]==1
    assert fields['movement'][0]<0
    assert fields['speed'][0]>=.03
    assert hold.command([0,0,0,.75],sensor(.012))[0]<0
    assert not hold.command([0,0,0,.75],sensor(.005))[:2].any()


def test_robot_reset_origin_and_quantization_reject_odometry_jitter():
    hold=BasePositionHold()
    data=dict(sensor(3.1,2),initial_base_position=[3,2,.75])
    assert hold.command([0,0,0,.75],data)[0]<0
    np.testing.assert_array_equal(hold.anchor,[3,2])
    planner=PlannerMapper(max_speed=.25,direction_resolution=.05,speed_resolution=.02)
    state=sensor()
    fields=[]
    for epsilon in [1e-5,-1e-5,2e-5,-2e-5]*10:
        fields.append(planner.fields(np.zeros(29),np.zeros(29),[.2,epsilon,0,.75],state,.02,np.zeros((2,7))))
    for f in fields[-10:]:
        np.testing.assert_array_equal(f['movement'],[1,0,0])
        assert f['speed'][0]==pytest.approx(.2)


def test_reanchor_holds_measured_ready_station_without_motion():
    hold=BasePositionHold()
    hold.command([0,0,0,.75],sensor())
    ready=sensor(.03,-.02)
    hold.reanchor(ready)
    np.testing.assert_array_equal(hold.command([0,0,0,.75],ready)[:2],[0,0])
    assert not hold.active
    assert hold.command([0,0,0,.75],sensor(.06,-.02))[0]<0
