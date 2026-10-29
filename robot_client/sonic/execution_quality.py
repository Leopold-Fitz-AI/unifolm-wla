"""Offline execution metrics, independent of grasp outcomes and controller feedback.

Velocity integrals use the recorded 10 Hz control samples and simulation clock;
they are estimates, not high-rate odometry ground truth. Object data is unused.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def measure_window(control, physics, seconds=None):
    if len(control)<2 or len(physics)<2:
        return {'available':False,'reason':'insufficient_robot_samples'}
    ct=np.asarray([r['simulation_time'] for r in control],float)
    pt=np.asarray([r['simulation_time'] for r in physics],float)
    if np.any(np.diff(ct)<=0) or np.any(np.diff(pt)<=0):
        return {'available':False,'reason':'non_monotonic_robot_clock'}
    end=ct[-1] if seconds is None else min(ct[-1],ct[0]+seconds)
    indices=np.flatnonzero(ct<=end+1e-9)
    times=ct[indices]
    if len(times)<2 or times[0]<pt[0] or times[-1]>pt[-1]:
        return {'available':False,'reason':'robot_clock_coverage_missing'}
    xyz=np.asarray([r['base_position'] for r in physics],float)
    yaw=np.unwrap(Rotation.from_quat(np.asarray([r['base_quat_wxyz'] for r in physics])[:,[1,2,3,0]])
                  .as_euler('xyz')[:,2])
    positions=np.stack([np.interp(times,pt,xyz[:,i]) for i in range(3)],axis=1)
    heading=np.interp(times,pt,yaw)
    actual=positions-positions[0]
    raw=np.asarray([control[i]['command_base'] for i in indices],float)
    sent=np.asarray([control[i]['planner_velocity'] for i in indices],float)
    dt=np.diff(times)

    def integrate(velocity):
        vx,vy=velocity[:,0],velocity[:,1]
        world=np.c_[np.cos(heading)*vx-np.sin(heading)*vy,
                    np.sin(heading)*vx+np.cos(heading)*vy]
        return np.vstack([np.zeros(2),np.cumsum(world[:-1]*dt[:,None],axis=0)])

    expected,executed=integrate(raw[:,:2]),integrate(sent)
    model_error=actual[:,:2]-expected
    actuator_error=actual[:,:2]-executed
    result={'available':True,'requested_window_seconds':seconds,
        'full_requested_window':bool(seconds is None or ct[-1]-ct[0]>=seconds-.02),
        'start_simulation_time':float(times[0]),'end_simulation_time':float(times[-1]),
        'duration_seconds':float(times[-1]-times[0]),'control_samples':len(times),
        'maximum_control_gap_seconds':float(dt.max()),
        'actual_base_displacement_m':actual[-1].tolist(),
        'integrated_model_xy_m':expected[-1].tolist(),
        'integrated_planner_xy_m':executed[-1].tolist(),
        'final_error_to_model_xy_m':model_error[-1].tolist(),
        'maximum_error_to_model_xy_m':float(np.linalg.norm(model_error,axis=1).max()),
        'maximum_error_to_planner_xy_m':float(np.linalg.norm(actuator_error,axis=1).max()),
        'actual_yaw_change_rad':float(heading[-1]-heading[0]),
        'integrated_model_yaw_rad':float(np.sum(raw[:-1,2]*dt)),
        'maximum_model_speed_mps':float(np.linalg.norm(raw[:,:2],axis=1).max()),
        'maximum_planner_speed_mps':float(np.linalg.norm(sent,axis=1).max()),
        'maximum_base_tilt_deg':float(max(r['base_tilt_deg'] for r in physics
            if times[0]<=r['simulation_time']<=times[-1]))}
    # New calibrated runs log the actual model goal separately from compensated
    # actuator joint FK. Never label the latter as a model tracking target.
    if all(control[i].get('model_target_ee') is not None for i in indices):
        goal=np.asarray([control[i]['model_target_ee'] for i in indices])[:,:,:3,3]
        measured=np.asarray([control[i]['measured_ee'] for i in indices])[:,:,:3,3]
        errors=np.linalg.norm(goal-measured,axis=2)
        result['model_tcp_position_error_m']={side:{'rms':float(np.sqrt(np.mean(errors[:,i]**2))),
            'maximum':float(errors[:,i].max())} for i,side in enumerate(('left','right'))}
    return result


def audit_trial(path):
    path=Path(path)
    def read(name):
        with (path/name).open() as stream:return [json.loads(line) for line in stream if line.strip()]
    control=[r for r in read('control.jsonl') if r.get('phase')=='vla' and r.get('simulation_time') is not None]
    physics=read('physics.jsonl')
    return {'trial':str(path),'uses_object_feedback':False,
        'integration_method':'zero_order_hold_of_recorded_commands_in_measured_world_yaw',
        'full_model_execution':measure_window(control,physics),
        'first_three_seconds':measure_window(control,physics,3.)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit_trial(args.trial)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
