import numpy as np
import pytest
import json

from robot_client.sonic.planner import PlannerMapper


def command(mapper,base):
    return mapper.fields(np.zeros(29),np.zeros(29),base,{'base_quat':[1,0,0,0]},.1,np.zeros((2,7)))


def test_model_only_small_base_commands_are_not_silently_zeroed():
    baseline=PlannerMapper(stationary_mode=4)
    native=command(baseline,[.01,0,.01,.75])
    assert native['mode'][0]==4 and native['speed'][0]==0
    assert baseline.modification_counts['velocity_deadband']==1
    assert baseline.modification_counts['yaw_deadband']==1
    literal=PlannerMapper(velocity_deadband_mps=0,yaw_deadband_rps=0,
        walking_speed_threshold_mps=1e-5,turning_rate_threshold_rps=1e-5)
    sent=command(literal,[.01,0,.01,.75])
    assert sent['mode'][0]==1
    assert sent['speed'][0]==pytest.approx(.01)
    assert sent['facing'][1]==pytest.approx(np.sin(.001))
    assert not any(literal.modification_counts.values())


def test_execution_limits_are_applied_and_counted():
    mapper=PlannerMapper(max_speed=.15,max_yaw_rate=.3,max_acceleration=.5,height_range=(.65,.8))
    sent=command(mapper,[1,0,1,1])
    assert sent['speed'][0]==pytest.approx(.05)
    assert sent['height'][0]==pytest.approx(.8)
    for key in ('velocity_clipped','yaw_clipped','acceleration_limited','height_clipped'):
        assert mapper.modification_counts[key]==1


@pytest.mark.parametrize('options',[{'velocity_deadband_mps':-1},
    {'yaw_deadband_rps':float('nan')},{'walking_speed_threshold_mps':0},
    {'turning_rate_threshold_rps':float('inf')}])
def test_invalid_thresholds_rejected(options):
    with pytest.raises(ValueError,match='deadbands'):PlannerMapper(**options)


def supported_mapper(**kwargs):
    return PlannerMapper(stationary_mode=4,max_speed=.15,
        velocity_deadband_mps=0,yaw_deadband_rps=0,
        walking_speed_threshold_mps=1e-5,turning_rate_threshold_rps=1e-5,
        minimum_walking_speed_mps=.1,**kwargs)


@pytest.mark.parametrize('speed',[1e-6,.001,.01,.099])
def test_unsupported_small_velocity_does_not_activate_or_amplify_gait(speed):
    mapper=supported_mapper()
    for _ in range(20):
        sent=command(mapper,[speed,0,0,.75])
        assert sent['mode'][0]==4 and sent['speed'][0]==0
        np.testing.assert_array_equal(sent['movement'],0)
        np.testing.assert_array_equal(mapper.last_command['sent_velocity_local_mps'],[0,0])
    assert mapper.last_command['filtered_velocity_local_mps'][0]==pytest.approx(speed)
    assert mapper.last_command['suppression_reason']=='below_minimum_walking_speed'
    assert mapper.modification_counts['minimum_walking_speed_suppressed']==20


@pytest.mark.parametrize('velocity',[[.1,0],[.15,0],[-.12,0],[0,.1],[.09,.09]])
def test_supported_model_velocities_remain_executable_without_direction_rewrite(velocity):
    mapper=supported_mapper()
    for _ in range(20):sent=command(mapper,[*velocity,0,.75])
    assert sent['mode'][0]==1
    assert .1<=sent['speed'][0]<=.150001
    np.testing.assert_allclose(mapper.last_command['sent_velocity_local_mps'],velocity,atol=1e-8)
    assert mapper.last_command['suppression_reason'] is None


def test_quantization_cannot_round_unsupported_velocity_up_to_a_gait():
    mapper=supported_mapper(speed_resolution=.01)
    for _ in range(20):sent=command(mapper,[.099,0,0,.75])
    assert sent['mode'][0]==4 and sent['speed'][0]==0
    assert mapper.last_command['suppression_reason']=='below_minimum_walking_speed'


def test_acceleration_ramp_never_sends_subminimum_walking_speeds():
    mapper=supported_mapper()
    modes=[]
    for request in [.15]*10+[0.]*10:
        sent=mapper.fields(np.zeros(29),np.zeros(29),[request,0,0,.75],
                           {'base_quat':[1,0,0,0]},.02,np.zeros((2,7)))
        modes.append(int(sent['mode'][0]))
        if sent['mode'][0]==1:assert sent['speed'][0]>=.1-1e-8
        else:assert sent['speed'][0]==0
    assert 1 in modes and modes[0]==modes[-1]==4
    assert mapper.modification_counts['mode_switches']==2


def test_pure_turn_updates_facing_and_height_without_requesting_translation():
    mapper=supported_mapper()
    for height in [.69,.71]:
        sent=command(mapper,[0,0,.2,height])
        assert sent['mode'][0]==4 and sent['speed'][0]==0
        np.testing.assert_array_equal(sent['movement'],[0,0,0])
        assert sent['height'][0]==pytest.approx(height)
    assert sent['facing'][1]==pytest.approx(np.sin(.04))
    assert mapper.last_command['sent_yaw_rate_rps']==.2
    assert mapper.modification_counts['turning_in_static_mode']==2


def test_stationary_packet_never_contains_residual_translation_below_mode_threshold():
    mapper=PlannerMapper(stationary_mode=4,velocity_deadband_mps=0,walking_speed_threshold_mps=.03)
    sent=command(mapper,[.02,0,0,.75])
    assert sent['mode'][0]==4 and sent['speed'][0]==0
    np.testing.assert_array_equal(sent['movement'],0)
    assert mapper.last_command['filtered_velocity_local_mps']==pytest.approx([.02,0])


def test_execution_trace_separates_local_and_planner_velocity_and_serializes():
    mapper=supported_mapper()
    state={'base_quat':[np.sqrt(.5),0,0,np.sqrt(.5)],'init_base_quat':[1,0,0,0]}
    for _ in range(20):
        mapper.fields(np.zeros(29),np.zeros(29),[.15,0,0,.75],state,.1,np.zeros((2,7)))
    trace=mapper.last_command
    np.testing.assert_allclose(trace['sent_velocity_local_mps'],[.15,0],atol=1e-10)
    np.testing.assert_allclose(trace['sent_velocity_planner_mps'],[0,.15],atol=1e-10)
    assert trace['requested_base_command']==[.15,0,0,.75]
    json.dumps(trace,allow_nan=False)


def test_initialization_idle_packet_uses_static_height_and_no_translation():
    mapper=supported_mapper()
    mapper.idle_packet({'base_quat':[1,0,0,0]},.69,.02)
    assert mapper.last_command['mode']==4
    assert mapper.last_command['height_m']==.69
    assert mapper.last_command['sent_velocity_local_mps']==[0.,0.]


@pytest.mark.parametrize('minimum',[-.1,float('nan'),float('inf'),.16])
def test_invalid_supported_speed_interval_rejected(minimum):
    with pytest.raises(ValueError,match='minimum_walking_speed'):
        PlannerMapper(max_speed=.15,minimum_walking_speed_mps=minimum)


def heading_command(mapper,actual_yaw,model_rate=0.,*,anchor=0.,dt=.02):
    state={'base_quat':[np.cos(actual_yaw/2),0,0,np.sin(actual_yaw/2)],
           'init_base_quat':[np.cos(anchor/2),0,0,np.sin(anchor/2)]}
    return mapper.fields(np.zeros(29),np.zeros(29),[0,0,model_rate,.75],state,dt,np.zeros((2,7)))


@pytest.mark.parametrize('drift',[-.05,.05])
def test_heading_feedback_opposes_measured_drift_in_the_initial_heading_frame(drift):
    mapper=supported_mapper(heading_tracking_gain=2.)
    heading_command(mapper,1.2,anchor=1.2)
    sent=heading_command(mapper,1.2+drift,anchor=1.2)
    trace=mapper.last_command
    assert trace['heading_error_rad']==pytest.approx(-drift)
    assert np.sign(trace['heading_correction_rad'])==-np.sign(drift)
    assert trace['anchor_yaw_rad']==pytest.approx(1.2)
    assert trace['requested_world_yaw_rad']==pytest.approx(1.2)
    assert mapper.facing==pytest.approx(0.)
    np.testing.assert_allclose(sent['facing'][:2],[np.cos(trace['sent_facing_angle_rad']),
                                                np.sin(trace['sent_facing_angle_rad'])],atol=1e-7)
    assert sent['mode'][0]==4 and sent['speed'][0]==0


def test_heading_feedback_uses_shortest_error_across_pi_without_target_reset():
    mapper=supported_mapper(heading_tracking_gain=2.)
    heading_command(mapper,np.pi-.01)
    previous=mapper.sent_facing
    heading_command(mapper,-np.pi+.01)
    trace=mapper.last_command
    assert mapper.facing==pytest.approx(np.pi-.01)
    assert trace['heading_error_rad']==pytest.approx(-.02)
    assert -.01<trace['heading_correction_rad']<0
    assert abs(mapper.sent_facing-previous)<.01


def test_heading_correction_is_saturated_smoothed_and_rate_limited():
    mapper=supported_mapper(heading_tracking_gain=2.,max_yaw_rate=.3)
    heading_command(mapper,0.)
    sent=[];filtered=[]
    for _ in range(100):
        heading_command(mapper,-1.)
        trace=mapper.last_command;sent.append(mapper.sent_facing);filtered.append(trace['filtered_heading_correction_rad'])
        assert abs(trace['heading_correction_rad'])<=.15+1e-12
        assert abs(trace['sent_yaw_rate_rps'])<=.3+1e-12
        assert trace['heading_correction_target_rad']==.15
    assert filtered[0]==pytest.approx(.15*(1-np.exp(-.02/.3)))
    assert sent[0]==pytest.approx(.006)
    assert sent[-1]>.149
    assert mapper.modification_counts['heading_correction_saturated']==100
    assert mapper.modification_counts['heading_rate_limited']>0


def test_model_turning_target_continues_while_feedback_and_output_saturate():
    mapper=supported_mapper(heading_tracking_gain=2.,max_yaw_rate=.3)
    heading_command(mapper,0.)
    previous=mapper.sent_facing
    for index in range(25):
        heading_command(mapper,-1.,model_rate=.3)
        trace=mapper.last_command
        assert mapper.facing==pytest.approx((index+1)*.3*.02)
        assert mapper.sent_facing-previous<=.3*.02+1e-12
        assert trace['model_yaw_rate_rps']==.3
        assert trace['heading_rate_limited']
        previous=mapper.sent_facing
    assert mapper.last_command['filtered_heading_correction_rad']>.12
    assert mapper.last_command['heading_correction_rad']==pytest.approx(0.,abs=1e-12)
    assert mapper.modification_counts['heading_rate_limited']==25


def test_tracking_exact_model_turns_preserves_the_integrated_target_across_pi():
    mapper=supported_mapper(heading_tracking_gain=2.,max_yaw_rate=.3)
    initial=np.pi-.03
    heading_command(mapper,initial)
    emitted=[]
    for index in range(20):
        expected=initial+(index+1)*.2*.02
        heading_command(mapper,expected,model_rate=.2)
        emitted.append(mapper.sent_facing)
        assert mapper.facing==pytest.approx(expected)
        assert mapper.sent_facing==pytest.approx(expected)
        assert mapper.last_command['heading_correction_rad']==pytest.approx(0.,abs=1e-12)
    np.testing.assert_allclose(np.diff(emitted),.004,atol=1e-12)


def test_zero_heading_gain_preserves_uncorrected_model_reference():
    mapper=supported_mapper()
    heading_command(mapper,.7,anchor=.5)
    sent=heading_command(mapper,-.8,model_rate=.2,anchor=.5)
    assert mapper.facing==pytest.approx(.204)
    assert mapper.sent_facing==pytest.approx(mapper.facing)
    assert mapper.last_command['heading_correction_rad']==0
    assert not mapper.modification_counts['heading_correction_packets']
    np.testing.assert_allclose(sent['facing'][:2],[np.cos(.204),np.sin(.204)],atol=1e-7)


@pytest.mark.parametrize('options',[{'heading_tracking_gain':-1},
    {'heading_tracking_gain':float('nan')},{'max_heading_correction_rad':-.1},
    {'max_heading_correction_rad':4.},{'heading_correction_time_constant':0.},
    {'heading_correction_time_constant':float('inf')}])
def test_invalid_heading_tracking_options_rejected(options):
    with pytest.raises(ValueError,match='heading tracking'):PlannerMapper(**options)
