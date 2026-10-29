from types import SimpleNamespace

import pytest

from robot_client.sonic import realtime


def test_heavy_render_does_not_accumulate_time_drift(monkeypatch):
    clock=SimpleNamespace(now=0.)
    monkeypatch.setattr(realtime.time,'monotonic',lambda:clock.now)
    def sleep(seconds):clock.now+=seconds
    monkeypatch.setattr(realtime.time,'sleep',sleep)
    counts={'physics':0,'render':0,'reward':0,'viewer':0}
    data=SimpleNamespace(time=0.)
    simulator=SimpleNamespace(_running=True,sim_dt=.005,viewer_dt=.02,reward_dt=.02,image_dt=.03)
    def step():
        data.time+=.005
        clock.now+=.0002
        counts['physics']+=1
        if counts['physics']==60:simulator._running=False
    def render():
        clock.now+=.012  # >2 physics timesteps; old pacing loses time every frame.
        counts['render']+=1
    def reward():counts['reward']+=1
    def viewer():counts['viewer']+=1
    closed=[]
    simulator.close=lambda:closed.append(True)
    simulator.sim_env=SimpleNamespace(viewer=None,mj_data=data,sim_step=step,
        update_viewer=viewer,update_reward=reward,update_render_caches=render)
    realtime.run_realtime(simulator)
    assert counts=={'physics':60,'render':10,'reward':15,'viewer':15}
    assert closed
    assert data.time==pytest.approx(.3)
    assert clock.now==pytest.approx(.3,abs=.001)


def test_shutdown_reset_does_not_make_completed_physics_time_negative(monkeypatch):
    clock=SimpleNamespace(now=0.)
    monkeypatch.setattr(realtime.time,'monotonic',lambda:clock.now)
    monkeypatch.setattr(realtime.time,'sleep',lambda seconds:setattr(clock,'now',clock.now+seconds))
    data=SimpleNamespace(time=20.)
    calls=[]
    def is_running():
        calls.append(True)
        if len(calls)>1:
            data.time=0.  # Concurrent DDS cleanup after the final physics step.
            return False
        return True
    sim=SimpleNamespace(_running=True,sim_dt=.005,viewer_dt=.02,reward_dt=.02,image_dt=.1,close=lambda:None)
    sim.sim_env=SimpleNamespace(viewer=SimpleNamespace(is_running=is_running),mj_data=data,
        sim_step=lambda:setattr(data,'time',data.time+.005),update_viewer=lambda:None,
        update_reward=lambda:None,update_render_caches=lambda:None)
    realtime.run_realtime(sim)
    assert sim.realtime_report['physics_seconds']==pytest.approx(.005)
    assert sim.realtime_report['shutdown_time_changed']
    assert not sim.realtime_report['clock_reset_during_step']
