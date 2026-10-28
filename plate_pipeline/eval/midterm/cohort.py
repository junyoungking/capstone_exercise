"""Explicit no-plate exclusions, complete baseline clock gate, matched 61-video arms."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import av
from reproduce import aggregate

HERE=Path(__file__).parent


def included(test):
    excluded=pd.read_csv(HERE/'excluded_no_plate.csv')
    assert len(excluded)==12 and not excluded.key.duplicated().any()
    assert set(excluded.key)<=set(test.key)
    result=test[~test.key.isin(excluded.key)].copy()
    assert len(result)==61
    return result


def clock_gate(v40,dataset,out):
    """Recreate v40 0-based source clock and require B delays before inference."""
    from run import module
    dec=module(v40/'decode.py');records=json.loads((v40/'records.json').read_text());rows=[];delays={};checks=[]
    for r in records:
        if r['split']!='test':continue
        with av.open(str(dataset/r['key'])) as c:ts=[float(f.time) for f in c.decode(video=0) if f.time is not None]
        origin=ts[0];gt=[ts[int(rep[2])]-origin for rep in r['reps']]
        rows.append({'key':r['key'],'clock_origin_s':origin,'gt_events':json.dumps(gt)})
        timeline=pd.read_csv(v40/f"arms/B_generic/timelines/all_test/{r['key']}.csv")
        tp,fp,fn,ds=dec.match_events(gt,timeline.loc[timeline.count_event,'time_s'].to_numpy());delays[r['key']]=ds
        checks.append({'key':r['key'],'tp':tp,'fp':fp,'fn':fn})
    pd.DataFrame(rows).to_csv(out/'gt_clock.csv',index=False)
    d=pd.read_csv(v40/'arms/B_generic/all_test_per_video.csv').set_index('key');got=pd.DataFrame(checks).set_index('key')
    assert np.array_equal(d.loc[got.index,['tp','fp','fn']].to_numpy(),got[['tp','fp','fn']].to_numpy())
    pd.DataFrame(checks).to_csv(out/'v40_clock_baseline_check.csv',index=False)
    results=[]
    for subset in ('reference_test','custom_test','all_test'):
        frame=pd.read_csv(v40/f'arms/B_generic/{subset}_per_video.csv')
        summary=aggregate(v40,frame,[v for key in frame.key for v in delays[key]])
        old=json.loads((v40/f'arms/B_generic/{subset}_summary.json').read_text())
        for metric,value in summary.items():
            same=(value is old[metric] if value is None else np.isclose(value,old[metric],rtol=0,atol=1e-9))
            results.append(dict(subset=subset,metric=metric,recalculated=value,saved=old[metric],same=bool(same)))
    pd.DataFrame(results).to_csv(out/'v40_clock_baseline_summary.csv',index=False)
    assert all(r['same'] for r in results),'B_generic baseline including delay differs; stop before plate evaluation'


def summarize_cohort(v40,out,raw):
    from run import module
    dec=module(v40/'decode.py')
    test=pd.read_csv(out/'test_membership.csv');keys=included(test).key.tolist()
    pd.read_csv(HERE/'excluded_no_plate.csv').to_csv(out/'excluded_no_plate.csv',index=False)
    test.assign(plate_eligible=~test.key.isin(pd.read_csv(HERE/'excluded_no_plate.csv').key)).to_csv(out/'test_membership.csv',index=False)
    truth={r['key']:json.loads(r['gt_events']) for r in pd.read_csv(out/'gt_clock.csv').to_dict('records')}
    dev=set(pd.read_csv(HERE.parent/'lists/team9.csv').path)
    held=set(pd.read_csv(v40/'heldout9.csv').key)
    fields=['key','type','gt_count','pred_count','signed_error','abs_error','exact','obo','tp','fp','fn','tracked_fraction','plate_ratio','view','counted_trips','incomplete_counted_trips','cause','delays']
    records=[]
    for key in keys:
        e=json.loads((raw/key/'result.json').read_text())['midterm_evaluation']
        r={k:e[k] for k in fields};r['development']=Path(key).name in dev;r['subset']='custom_test' if key in held else 'reference_test';records.append(r)
    plate=pd.DataFrame(records);plate.drop(columns=['delays']).to_csv(out/'per_video.csv',index=False)
    plate[['key','view','plate_ratio','tracked_fraction','development']].to_csv(out/'view_tags.csv',index=False)
    pd.DataFrame([dict(view=view,**aggregate(v40,df,[x for ds in df.delays for x in ds])) for view,df in plate.groupby('view')]).to_csv(out/'by_view.csv',index=False)
    summaries=[];comparison=[];by_exercise=[];without=[]
    for arm in ('A_existing','B_generic','C_athlete','plate'):
        frame=plate.copy() if arm=='plate' else pd.read_csv(v40/f'arms/{arm}/all_test_per_video.csv')
        frame=frame[frame.key.isin(keys)].copy();assert set(frame.key)==set(keys)
        if arm!='plate':
            frame['development']=frame.key.map(lambda k:Path(k).name in dev)
            delay_map={}
            for key in keys:
                timeline=pd.read_csv(v40/f'arms/{arm}/timelines/all_test/{key}.csv')
                tp,fp,fn,ds=dec.match_events(truth[key],timeline.loc[timeline.count_event,'time_s'].to_numpy())
                row=frame[frame.key==key].iloc[0];assert (tp,fp,fn)==(row.tp,row.fp,row.fn)
                delay_map[key]=ds
            frame['delays']=frame.key.map(delay_map)
        for subset in ('reference_test','custom_test','all_test'):
            base=frame if subset=='all_test' else frame[frame.key.isin(held) if subset=='custom_test' else ~frame.key.isin(held)]
            for scope,df in [('all',base),('without_development',base[~base.development])]+[(ex,base[base.type==ex]) for ex in ('squat','benchpress','deadlift')]:
                if df.empty:continue
                s=dict(subset=subset,scope=scope,arm=arm,**aggregate(v40,df,[x for ds in df.delays for x in ds]))
                if arm=='plate':summaries.append({k:v for k,v in s.items() if k!='arm'})
                if scope=='all':comparison.append(s)
                elif scope=='without_development':without.append(s)
                else:by_exercise.append(s)
    pd.DataFrame(summaries).to_csv(out/'plate_summary.csv',index=False)
    pd.DataFrame(comparison).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(by_exercise).to_csv(out/'comparison_by_exercise.csv',index=False)
    pd.DataFrame(without).to_csv(out/'comparison_without_development.csv',index=False)
    pd.read_csv(v40/'comparison.csv').to_csv(out/'baseline73_abc.csv',index=False)
    pd.read_csv(v40/'per_exercise_comparison.csv').to_csv(out/'baseline73_abc_by_exercise.csv',index=False)
