"""Deadline pacing for upstream's single-thread physics/render loop.

Per-step sleeping accumulates render overruns and slows physics relative to
the external 50 Hz controller. Catch up short overruns against one deadline;
never change the physics timestep or advance objects outside MuJoCo.
"""
import logging
import time

LOG=logging.getLogger(__name__)


def run_realtime(simulator):
    count=0
    started=deadline=time.monotonic()
    sim_started=float(simulator.sim_env.mj_data.time)
    last_step_time=sim_started
    clock_reset=False
    max_lag=0.
    durations={'physics':0.,'render':0.,'reward':0.}
    band=getattr(simulator.sim_env,'elastic_band',None)
    suspended=bool(band and band.enable)
    try:
        while simulator._running and (simulator.sim_env.viewer is None or simulator.sim_env.viewer.is_running()):
            compute_start=time.monotonic()
            simulator.sim_env.sim_step()
            stepped=float(simulator.sim_env.mj_data.time)
            if stepped<last_step_time-1e-9:clock_reset=True
            else:last_step_time=stepped
            durations['physics']+=time.monotonic()-compute_start
            enabled=bool(band and band.enable)
            if suspended and not enabled:
                # GPU/controller initialization may block while suspended.
                # Start active-control pacing with a fresh deadline on release.
                started=deadline=time.monotonic()
                sim_started=float(simulator.sim_env.mj_data.time)
                last_step_time=sim_started
                clock_reset=False
                max_lag=0.
                durations={k:0. for k in durations}
            suspended=enabled
            if count%max(1,int(simulator.viewer_dt/simulator.sim_dt))==0:
                simulator.sim_env.update_viewer()
            if count%max(1,int(simulator.reward_dt/simulator.sim_dt))==0:
                compute_start=time.monotonic()
                simulator.sim_env.update_reward()
                durations['reward']+=time.monotonic()-compute_start
            if count%max(1,int(simulator.image_dt/simulator.sim_dt))==0:
                compute_start=time.monotonic()
                simulator.sim_env.update_render_caches()
                durations['render']+=time.monotonic()-compute_start
            deadline+=simulator.sim_dt
            lag=time.monotonic()-deadline
            max_lag=max(max_lag,lag)
            if lag<0:time.sleep(-lag)
            elif suspended and lag>.25:
                deadline=time.monotonic()
            elif lag>.5 and float(simulator.sim_env.mj_data.time)-sim_started>2:
                raise RuntimeError('simulator more than 0.5 seconds behind real time; reduce rendering workload')
            count+=1
    finally:
        elapsed=time.monotonic()-started
        # DDS shutdown may reset mjData before this finally block runs. Keep
        # the last completed physics timestamp, and expose any observed reset.
        simulated=last_step_time-sim_started
        simulator.realtime_report={'mode':'deadline','physics_seconds':simulated,'wall_seconds':elapsed,
            'factor':simulated/max(elapsed,1e-9),'max_lag_seconds':max_lag,'compute_seconds':durations,
            'clock_reset_during_step':clock_reset,
            'shutdown_time_changed':float(simulator.sim_env.mj_data.time)<last_step_time-1e-9}
        LOG.warning('Simulation clock: physics=%.3fs wall=%.3fs factor=%.4f max_lag=%.4fs',
                    simulated,elapsed,simulated/max(elapsed,1e-9),max_lag)
        simulator.close()
