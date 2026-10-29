import copy
import json
import yaml

import numpy as np
import pytest

from robot_client.sonic.grasp_quality import audit_samples, audit_trial


def samples():
    return [{'simulation_time':i*.02,'suspension_enabled':False,
             'hands':{'left':{'tcp_position':[0,0,1],'wrist_rotation':np.eye(3).ravel().tolist()}},
             'objects':{'cube':{'position':[0,0,1],'bottom_above_table':.12,
                                'jaw_contacts':{'left':[0,1]},'table_contact':False}}}
            for i in range(301)]


def audit(rows):return audit_samples(rows,0,6,'cube','left')


def test_common_robot_object_motion_is_not_slip():
    rows=samples()
    for i,r in enumerate(rows):
        r['hands']['left']['tcp_position'][0]=i*.001
        r['objects']['cube']['position'][0]=i*.001
    result=audit(rows)
    assert result['quality_passed'] and result['maximum_local_slip_m']==0
    assert result['tcp_range_m'][0]==pytest.approx(.3)


def test_continuous_contact_does_not_hide_slow_slip():
    rows=samples()
    for i,r in enumerate(rows):r['objects']['cube']['position'][2]-=.012*i/300
    result=audit(rows)
    assert not result['quality_passed']
    assert result['maximum_local_slip_m']==pytest.approx(.012)
    assert result['reasons']==['excessive_slip']


@pytest.mark.parametrize('rows',[samples()[:101],samples()[100:],samples()[::5]])
def test_incomplete_or_sparse_hold_cannot_pass(rows):
    assert 'incomplete_hold_coverage' in audit(rows)['reasons']


def test_final_support_loss_and_clock_reset_rejected():
    rows=samples();rows[-1]['objects']['cube']['jaw_contacts']['left']=[0]
    assert 'bilateral_support_lost' in audit(rows)['reasons']
    rows[-1]['simulation_time']=0
    with pytest.raises(ValueError,match='clock'):audit(rows)


def test_wrong_object_and_nonfinite_evidence_rejected():
    rows=samples();rows[-1]['objects']['other']=copy.deepcopy(rows[-1]['objects']['cube'])
    assert 'wrong_object_lifted' in audit(rows)['reasons']
    rows[-1]['objects']['cube']['position'][0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):audit(rows)


def test_missing_trial_is_a_failure(tmp_path):
    assert not audit_trial(tmp_path)['quality_passed']


@pytest.mark.parametrize('collision',[{}, {'passed':False}, {'passed':True}])
def test_safe_trial_requires_full_episode_collision_evidence(tmp_path,collision):
    (tmp_path/'task.yaml').write_text(yaml.safe_dump({'target':'cube','required_hand':'left',
        'lift_height':.08,'robot_contacts_from_start':True}))
    (tmp_path/'client-config.yaml').write_text(yaml.safe_dump({'grasp_execution':{'hold_seconds':6}}))
    for name,value in {'grasp-execution.json':{'completed':True,'events':[{'phase':'hold','simulation_time':0}]},
                       'client.json':{'status':'passed'},
                       'task-report.json':{'success':True,'table_collision_audit':collision}}.items():
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'physics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples()))
    report=audit_trial(tmp_path)
    assert report['quality_passed']==collision.get('passed',False)
