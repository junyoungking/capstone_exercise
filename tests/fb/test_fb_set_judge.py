from copy import deepcopy

import numpy as np
import pytest

from feedback import config
from feedback.compare import compare_reps
from feedback.select import select_goals
from feedback.set_judge import compare_plate_reps
from core.compare.summary import from_values


def metrics(values, *, exercise='squat', knee='knee_angle', indices=None):
    reps=[]
    for number,value in zip(indices or range(1,len(values)+1),values):
        positions={p:{'frame':number*10,'values':{m:(value if m==knee else 0.0) for m in config.DISPLAY_METRICS[exercise]}}
                   for p in config.POSITIONS}
        reps.append({'rep':number,'excluded':False,'exclude_reason':None,'detect':dict.fromkeys(config.PHASES,1.0),
                     'positions':positions,'tempo':{'down_s':2.0,'bottom_s':.2,'up_s':1.0,'ratio':2.0}})
    return {'exercise':exercise,'fps':30,'reps':reps}


def bottom(findings, metric='knee_angle'):
    return next(c for c in findings['posture_judgments']['cells'] if c['where']=='BOTTOM' and c['metric']==metric)


def test_no_judgment_for_high_user_scatter():
    f=compare_plate_reps(metrics([0,180,0,180,0,180]),metrics([100]*6))
    cell=bottom(f)
    assert cell['result']=='skip' and cell['reason']=='unstable_user'
    assert cell['s_a']>36
    assert 'rep마다 값이 크게 달라 비교하지 않음 (산포 ' in cell['reason_text']
    assert select_goals(f)['reps'][0]['goal']=={'type':'none'}


def test_short_reference_needs_four_user_values_and_keeps_gaps():
    f=compare_plate_reps(metrics([80]*3),metrics([100]))
    assert (bottom(f)['result'],bottom(f)['reason'],bottom(f)['need'])==('skip','set_too_short',4)
    f=compare_plate_reps(metrics([80]*3,indices=[1,3,5]),metrics([100]*3))
    assert bottom(f)['reason']=='no_scatter'


def test_small_change_cannot_pass_even_with_large_critical_ratio():
    f=compare_plate_reps(metrics([84]*10),metrics([100]*10))
    cell=bottom(f)
    assert cell['critical_ratio']>1 and cell['result']=='same'
    assert not any(c['goal_eligible'] for r in f['reps'] for c in r['cells'])


def test_goal_requires_set_evidence_direction_and_inclusive_minimum():
    # The median is lower, but the last rep differs in the opposite direction.
    f=select_goals(compare_plate_reps(metrics([80,80,80,80,120]),metrics([100]*5)))
    assert bottom(f)['result']=='different'
    assert f['reps'][0]['goal']['type']=='single'
    assert f['reps'][-1]['goal']=={'type':'none'}
    f=select_goals(compare_plate_reps(metrics([82]*10),metrics([100]*10)))
    assert bottom(f)['result']=='different'
    assert f['reps'][0]['goal']['type']=='single'  # |delta| == minimum 18
    assert f['reps'][0]['cells'][0]['k']==1


def tempo_summaries():
    mine=from_values('squat',30,down=[2.5]*5,far_pause=[.2]*5)
    ref=from_values('squat',30,down=[1.0]*5,far_pause=[.2]*5)
    return ([(mine,list(range(1,6)))],[(ref,list(range(1,6)))])


def test_plate_tempo_uses_core_sentences_and_disables_old_rep_rule():
    user,ref=metrics([80]*5),metrics([100]*5)
    for r in ref['reps']:r['tempo'].update(down_s=.5,ratio=.5)
    legacy=select_goals(compare_reps(user,ref))
    assert legacy['reps'][0]['tempo_goal'] is not None
    f=select_goals(compare_plate_reps(user,ref,summaries=tempo_summaries()))
    assert f['tempo_thresholds']=={}
    assert all(r['tempo_goal'] is None for r in f['reps'])
    assert all(t['k'] is None and t['threshold'] is None for r in f['reps'] for t in r['tempo'])
    comparisons=f['tempo_comparison'][0]['comparisons']
    assert {c['value'] for c in comparisons}=={'down','far_pause'}
    assert next(c for c in comparisons if c['value']=='down')['result']=='different'
    sentences=f['tempo_comparison'][0]['sentences']
    assert any('내리는 시간이 기준 영상보다 1.50초 길었습니다' in s['text'] for s in sentences)


def test_side_front_blocks_posture_only_diagonal_is_a_warning():
    user,ref=metrics([80]*5),metrics([100]*5)
    f=compare_plate_reps(user,ref,angles={'user':{'view':'side'},'ref':{'view':'front'}},summaries=tempo_summaries())
    assert f['posture_judgments']['judged_count']==0
    assert f['posture_judgments']['reason_counts']=={'angle_mismatch':25}
    assert bottom(f)['reason_text']=='촬영 각도가 달라 자세는 비교하지 않음'
    assert any(c['result']=='different' for c in f['tempo_comparison'][0]['comparisons'])
    assert bottom(compare_plate_reps(user,ref,angles={'user':{'view':'diagonal'},'ref':{'view':'front'}}))['result']=='different'


def test_no_goal_from_rejected_set_even_when_legacy_rep_threshold_exceeded():
    user,ref=metrics([82,82,82]),metrics([100])
    f=select_goals(compare_plate_reps(user,ref))
    assert all(r['goal']=={'type':'none'} for r in f['reps'])
    assert all(r['unjudged_cell_count']==25 for r in f['reps'])


def test_finding_sentences_cap_and_banned_words():
    f=compare_plate_reps(metrics([80]*5),metrics([100]*5),summaries=tempo_summaries())
    assert len(f['set_findings'])==3
    assert f['set_findings'][0]['sentence']=='시작에서 무릎을 레퍼런스보다 20° 더 굽혔습니다.'
    for text in [c['sentence'] for c in f['set_findings']]+[s['text'] for t in f['tempo_comparison'] for s in t['sentences']]:
        assert not any(word in text for word in config.BANNED)


def test_actual_set_boundaries_never_supply_a_scatter_triple_across_sets():
    user,ref=metrics([80]*6),metrics([100]*5)
    a=from_values('squat',30,down=[1]*2)
    b=from_values('squat',30,down=[1]*5)
    summaries=([(a,[1,2]),(a,[3,4]),(a,[5,6])],[(b,[1,2,3,4,5])])
    f=compare_plate_reps(user,ref,summaries=summaries)
    assert f['posture_judgments']['judged_count']==0
    assert f['posture_judgments']['reason_counts']=={'set_too_short':75}


def test_selected_reference_uses_its_actual_set_and_all_valid_reference_reps():
    user=metrics([80]*5)
    ref=metrics([100]*5+[150]*5)
    a=from_values('squat',30,down=[1]*5)
    f=compare_plate_reps(user,ref,6,summaries=([(a,[1,2,3,4,5])],[(a,[1,2,3,4,5]),(a,[6,7,8,9,10])]))
    assert bottom(f)['b']==150
    assert bottom(f)['ref_reps']==[6,7,8,9,10]
    assert bottom(f)['ref_set']==2
    # The selected ref supplies rep comparisons; set comparison uses its set median.
    ref=metrics([100,100,100,100,150])
    f=compare_plate_reps(user,ref,5)
    assert bottom(f)['b']==100 and bottom(f)['delta']==-20
    assert f['reps'][0]['cells'][0]['diff']==-70


def test_ratio_floor_and_missing_reference_values():
    user=metrics([.20]*5,knee='foot_flatness')
    ref=metrics([.05]*5,knee='foot_flatness')
    cell=bottom(compare_plate_reps(user,ref),'foot_flatness')
    assert cell['s_a']==.03 and cell['s_b']==.03
    assert cell['min_change']==config.DELTA_THRESHOLDS['squat']['foot_flatness']
    for rep in ref['reps']:
        rep['positions']['BOTTOM']['values']['foot_flatness']=None
    cell=bottom(compare_plate_reps(user,ref),'foot_flatness')
    assert cell['reason']=='ref_too_short' and cell['n_b']==0
