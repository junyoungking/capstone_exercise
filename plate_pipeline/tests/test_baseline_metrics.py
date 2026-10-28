import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest
from eval.baseline_metrics import (POSITIONS, ANGLES, rep_variation, rep_mask,
                                   observation_stats, odd_even)


def test_variation_unordered_pairs_sample_sd_and_excluded_reps():
    metrics={'reps':[dict(rep=i,excluded=i==3,positions={p:{'values':{m:v for m in ANGLES}}
                                                        for p in POSITIONS})
                     for i,v in enumerate((0.,2.,6.),1)]}
    cells=rep_variation(metrics)
    assert len(cells)==15
    for cell in cells:
        assert cell['reps']==3 and cell['pairs']==3
        assert cell['pair_median']==4
        assert cell['sd']==pytest.approx(np.std([0,2,6],ddof=1))
    metrics['reps']=metrics['reps'][:1]
    assert all(c['sd'] is None and c['pair_median'] is None for c in rep_variation(metrics))


def test_rep_mask_is_union_and_uses_half_open_bounds():
    seg={'total':8,'reps':[{'phases':{'down':[1,2],'up':[3,5]}},
                          {'phases':{'down':[4,5],'up':[6,7]}}]}
    assert rep_mask(seg).tolist()==[False,True,True,True,True,True,True,False]


def test_coverage_joint_denominator_visibility_threshold_and_jumps():
    raw=np.zeros((5,33,4));raw[:,:,3]=.5
    raw[1,:,:2]=.2;raw[2,:,:2]=.6;raw[3,:,:2]=.9
    raw[2,11,3]=.49
    pose=dict(raw_lms=raw,valid=np.array([True,True,True,False,True]),w=100,h=100)
    rows=[{'plate':{'used':True,'box':[0,0,10,30]}} for _ in range(5)]
    stats=observation_stats(pose,np.array([False,True,True,True,False]),rows)
    assert stats['coverage']==2/3
    assert stats['joints']['shoulder']==dict(visible=3,total=6,rate=.5)
    assert stats['joints']['foot']['total']==12
    assert stats['jumps']==1 and stats['jump_frames'][0]['frame']==2
    assert stats['jump_frames'][0]['distance_px']==pytest.approx(math.sqrt(3200))
    empty=observation_stats(pose,np.zeros(5,bool),rows)
    assert empty['coverage'] is None and empty['joints']['hip']['rate'] is None
    with pytest.raises(ValueError):observation_stats(pose,np.ones(4,bool),rows)


def test_identity_jumps_do_not_bridge_missing_or_nonrep_frames():
    raw=np.zeros((3,33,4));raw[2,:,:2]=1
    pose=dict(raw_lms=raw,valid=np.array([True,False,True]),w=100,h=100)
    rows=[{'plate':{'used':True,'box':[0,0,10,1]}} for _ in range(3)]
    assert observation_stats(pose,np.ones(3,bool),rows)['jumps']==0


def test_odd_even_requires_both_arms_and_minimum_reps():
    result=odd_even({'reps':[{'rep':1},{'rep':2},{'rep':3}]},[])
    assert result['judged']==0 and result['rate'] is None


def _synthetic_metrics(values):
    from feedback import config
    reps=[]
    for number,value in enumerate(values,1):
        positions={p:{'frame':number*10,'values':{m:(value if m=='knee_angle' else 0.)
                     for m in config.DISPLAY_METRICS['squat']}} for p in POSITIONS}
        reps.append(dict(rep=number,excluded=False,exclude_reason=None,positions=positions,
                         detect=dict.fromkeys(config.PHASES,1.),
                         tempo=dict(down_s=2.,bottom_s=.2,up_s=1.,ratio=2.)))
    return dict(exercise='squat',fps=30,reps=reps)


def test_odd_even_constant_and_shifted_arms_and_missing_values():
    from core.compare.summary import from_values
    from copy import deepcopy
    st=from_values('squat',30,down=[2.]*12,far_pause=[.2]*12)
    summaries=[(st,list(range(1,13)))]
    m=_synthetic_metrics([80.]*12);original=deepcopy(m)
    same=odd_even(m,summaries)
    assert same['judged']>0 and same['different']==0 and same['rate']==0
    assert m==original
    changed=odd_even(_synthetic_metrics([80.,110.]*6),summaries)
    assert changed['different']>=5 and changed['rate']>0
    missing=_synthetic_metrics([80.]*12)
    for rep in missing['reps']:
        if rep['rep']%2:
            for p in POSITIONS:rep['positions'][p]['values']['knee_angle']=None
    result=odd_even(missing,summaries)
    knees=[c for c in result['cells'] if c['metric']=='knee_angle']
    assert knees and all(c['result']=='skip' for c in knees)


def test_measure_no_plate_reps_reports_unavailable_without_inventing_intervals():
    from eval.baseline_metrics import measure
    pose=dict(total=5,fps=30.,w=100,h=100,lms=np.full((5,33,4),np.nan),
              raw_lms=np.full((5,33,4),np.nan),valid=np.zeros(5,bool))
    rows=[dict(seq=i,t=i/30) for i in range(5)]
    stats,segments=measure(pose,rows,dict(sets=[],exercise='squat',fps=30.),'unused.mp4','squat')
    assert segments['reps']==[]
    assert stats['observation']['rep_frames']==0 and stats['observation']['coverage'] is None
    assert stats['false_difference']['rate'] is None
    assert all(c['sd'] is None for c in stats['variation'])


def test_completion_events_keep_native_pts_origin_and_drop_incomplete(monkeypatch):
    from types import SimpleNamespace
    import av
    from eval.baseline import count_result
    class Container:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def decode(self,**kwargs):return [SimpleNamespace(time=t) for t in (7.5,7.6,7.7)]
    monkeypatch.setattr(av,'open',lambda _:Container())
    def match(gt,pred):
        assert gt==pytest.approx([.2]) and pred==pytest.approx([.2])
        return 1,0,0,[0.]
    result={'sets':[{'rep_values':[{'back':7.7},{'back':None}]}]}
    record=dict(key='squat/test.mp4',type='squat',reps=[[0,1,2]])
    counts=count_result(record,'fake.mp4',result,SimpleNamespace(match_events=match))
    assert counts['pred_count']==1 and counts['gt_count']==1 and counts['exact']
    assert counts['tp']==1 and counts['fp']==0 and counts['fn']==0


def test_summary_uses_video_medians_and_pooled_denominators():
    from eval.baseline import summarize
    from eval.baseline_metrics import JOINTS
    def video(n,frames,selected,sd,judged,different,seconds):
        return dict(exercise='squat',counts=None,frames=n,
                    variation=[dict(position=p,metric=m,sd=sd,pair_median=sd,reps=100 if sd==2 else 2)
                               for p in POSITIONS for m in ANGLES],
                    false_difference=dict(judged=judged,different=different),
                    observation=dict(rep_frames=frames,pose_frames=selected,jumps=0,
                                     joints={k:dict(visible=selected,total=frames) for k in JOINTS}),
                    timing=dict(decode=0,plate_detection=0,klt=0,pose=0,rest=seconds,total=seconds,fps=n/seconds))
    data=[video(100,4,2,2,2,1,5),video(100,2,1,100,10,0,5)]
    summary=summarize(data,None)
    assert summary['all']['variation'][0]['sd']==51
    assert summary['all']['coverage']==.5
    assert summary['all']['false_difference']['rate']==1/12
    assert summary['all']['timing']['fps']==20
    assert summary['benchpress']['coverage'] is None
