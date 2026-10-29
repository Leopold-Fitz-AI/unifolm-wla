import json
from pathlib import Path

import pytest

from robot_client.sonic.grasp_trial import (CollisionReportWatchdog,
    request_collision_stop, validate_trial_policy)


def test_collision_supervisor_only_stops_and_ignores_grasp_success(tmp_path):
    report=tmp_path/'task-report.json'
    report.write_text(json.dumps({'success':False,'table_collision_audit':{'passed':True}}))
    assert not request_collision_stop(tmp_path)
    assert not (tmp_path/'safety-stop.json').exists()
    report.write_text(json.dumps({'success':True,'table_collision_audit':{
        'passed':False,'maximum_penetration_m':.004}}))
    assert request_collision_stop(tmp_path)
    marker=json.loads((tmp_path/'safety-stop.json').read_text())
    assert set(marker)=={'reason','audit'}
    assert marker['reason']=='independent_table_collision_audit_failed'


def test_no_report_or_in_progress_write_does_not_invent_collision(tmp_path):
    assert not request_collision_stop(tmp_path)
    (tmp_path/'task-report.json').write_text('{')
    assert not request_collision_stop(tmp_path)


def test_complete_report_without_safety_evidence_fails_closed(tmp_path):
    (tmp_path/'task-report.json').write_text('{}')
    assert request_collision_stop(tmp_path)


class Clock:
    value=0.

    def __call__(self):
        return self.value


def passing_report(path, steps=1):
    (path/'task-report.json').write_text(json.dumps({'table_collision_audit':{
        'passed':True,'checked_physics_steps':steps}}))


def test_initial_report_wait_is_bounded_and_cannot_be_extended_by_partial_writes(tmp_path):
    clock=Clock();watchdog=CollisionReportWatchdog(clock=clock)
    assert not request_collision_stop(tmp_path,watchdog)
    assert not watchdog.has_valid_report
    clock.value=59.
    (tmp_path/'task-report.json').write_text('{')
    assert not request_collision_stop(tmp_path,watchdog)
    clock.value=60.
    assert request_collision_stop(tmp_path,watchdog)
    marker=json.loads((tmp_path/'safety-stop.json').read_text())
    assert marker['reason']=='collision_report_initial_timeout'
    passing_report(tmp_path)
    assert request_collision_stop(tmp_path,watchdog)  # A stop is never cancelled.


def test_same_report_cannot_keep_watchdog_alive(tmp_path):
    clock=Clock();watchdog=CollisionReportWatchdog(clock=clock)
    passing_report(tmp_path)
    assert not request_collision_stop(tmp_path,watchdog)
    assert watchdog.has_valid_report
    clock.value=4.9
    assert not request_collision_stop(tmp_path,watchdog)
    clock.value=5.
    assert request_collision_stop(tmp_path,watchdog)
    marker=json.loads((tmp_path/'safety-stop.json').read_text())
    assert marker['reason']=='collision_report_stale'


def test_partial_write_gets_remaining_grace_without_refreshing_deadline(tmp_path):
    clock=Clock();watchdog=CollisionReportWatchdog(clock=clock)
    passing_report(tmp_path)
    assert not request_collision_stop(tmp_path,watchdog)
    clock.value=4.
    (tmp_path/'task-report.json').write_text('{')
    assert not request_collision_stop(tmp_path,watchdog)
    clock.value=5.
    assert request_collision_stop(tmp_path,watchdog)


def test_new_valid_physics_report_refreshes_watchdog_after_transient_write(tmp_path):
    clock=Clock();watchdog=CollisionReportWatchdog(clock=clock)
    passing_report(tmp_path)
    assert not request_collision_stop(tmp_path,watchdog)
    clock.value=3.
    (tmp_path/'task-report.json').write_text('{')
    assert not request_collision_stop(tmp_path,watchdog)
    clock.value=4.
    passing_report(tmp_path,steps=2)
    assert not request_collision_stop(tmp_path,watchdog)
    clock.value=8.9
    assert not request_collision_stop(tmp_path,watchdog)
    clock.value=9.
    assert request_collision_stop(tmp_path,watchdog)


def test_increasing_physics_counter_refreshes_with_identical_file_metadata(tmp_path,monkeypatch):
    clock=Clock();watchdog=CollisionReportWatchdog(clock=clock)
    passing_report(tmp_path)
    report=tmp_path/'task-report.json'
    frozen=report.stat()
    real_stat=Path.stat

    def fixed_report_stat(path,*args,**kwargs):
        return frozen if path==report else real_stat(path,*args,**kwargs)

    monkeypatch.setattr(Path,'stat',fixed_report_stat)
    assert not request_collision_stop(tmp_path,watchdog)
    clock.value=4.
    passing_report(tmp_path,steps=2)
    assert not request_collision_stop(tmp_path,watchdog)
    assert watchdog.last_valid_at==4.
    clock.value=8.9
    passing_report(tmp_path,steps=2)  # Rewriting the same count still cannot refresh.
    assert not request_collision_stop(tmp_path,watchdog)
    clock.value=9.
    assert request_collision_stop(tmp_path,watchdog)


def test_rewriting_a_frozen_physics_counter_is_not_fresh_evidence(tmp_path):
    clock=Clock();watchdog=CollisionReportWatchdog(clock=clock)
    passing_report(tmp_path)
    assert not request_collision_stop(tmp_path,watchdog)
    clock.value=4.
    passing_report(tmp_path)
    assert not request_collision_stop(tmp_path,watchdog)
    clock.value=5.
    assert request_collision_stop(tmp_path,watchdog)


def test_model_only_trial_rejects_default_visual_executor():
    with pytest.raises(ValueError,match='require --policy vla'):
        validate_trial_policy({'execution_mode':'model_only'},'visual')
    validate_trial_policy({'execution_mode':'model_only'},'vla')
    validate_trial_policy({},'visual')
