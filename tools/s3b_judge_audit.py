#!/usr/bin/env python3
"""S3b judge evidence from saved S3a metrics; no video/model inference."""
from pathlib import Path
import sys,json,itertools
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from feedback import config


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def stage0(source,out):
    videos={}
    for path in sorted(source.glob('*/metrics_user.json')):
        metrics=read(path);cells=[]
        for position in config.POSITIONS:
            for metric in config.DISPLAY_METRICS[metrics['exercise']]:
                # Include every available S3a cell; report eligibility explicitly.
                samples=[(r['rep'],r['positions'][position]['values'][metric]) for r in metrics['reps']
                         if r['positions'][position]['values'][metric] is not None]
                pairs=[{'reps':[a[0],b[0]],'abs_diff':abs(a[1]-b[1])} for a,b in itertools.combinations(samples,2)]
                diffs=[p['abs_diff'] for p in pairs]
                cells.append({'where':position,'metric':metric,'unit':'ratio' if metric in config.RATIO_METRICS else 'deg',
                    'value_count':len(samples),'pair_count':len(pairs),'median_abs_diff':float(np.median(diffs)) if diffs else None,
                    'max_abs_diff':max(diffs) if diffs else None,'pairs':pairs})
        videos[path.parent.name]={'exercise':metrics['exercise'],'rep_count':len(metrics['reps']),
                                 'excluded_rep_numbers':[r['rep'] for r in metrics['reps'] if r['excluded']],'cells':cells}
    result={'source':'S3a development_live metrics_user.json','population':'all available measured cells; excluded reps listed separately',
            'statistic':'absolute pairwise difference, unique unordered pairs, unrounded values','videos':videos}
    write(out/'stage0_rep_differences.json',result)
    print('stage0 videos',len(videos),flush=True)
    for name,where,metric in [('stu10_70','BOTTOM','knee_angle'),('stu10_70','MID_UP','knee_angle'),
                             ('stu3_64','BOTTOM','hip_angle'),('stu2_10_edited','MID_DOWN','elbow_angle')]:
        cell=next(c for c in videos[name]['cells'] if c['where']==where and c['metric']==metric)
        print(name,where,metric,cell['median_abs_diff'],cell['max_abs_diff'],flush=True)
    return result



def legacy_stats(findings):
    from feedback.describe import describe_reps
    from collections import Counter
    reps=findings['reps']; cells={}
    for rep in reps:
        if rep['excluded']:continue
        for c in rep['cells']:
            if c['status']!='ok' or c['k'] is None or c['metric'] in config.ADVISORY_METRICS.get(findings['exercise'],set()):continue
            if config.DELTA_THRESHOLDS[findings['exercise']][c['metric']]<=0:continue
            key=(c['where'],c['metric']);cells[key]=cells.get(key,False) or c['k']>1
    tempo=[t for r in reps if not r['excluded'] for t in r['tempo'] if t['k'] is not None]
    reasons=Counter()
    for where in config.POSITIONS:
        for metric in config.DISPLAY_METRICS[findings['exercise']]:
            if (where,metric) in cells:continue
            if metric in config.ADVISORY_METRICS.get(findings['exercise'],set()):reason='advisory'
            elif config.DELTA_THRESHOLDS[findings['exercise']][metric]<=0:reason='display_only'
            elif all(r['excluded'] for r in reps):reason='all_reps_excluded'
            else:reason='unavailable'
            reasons[reason]+=1
    return {'reason_counts':dict(reasons),'unjudged':sum(reasons.values()),'descriptions':describe_reps(findings),'judged':len(cells),'different':sum(cells.values()),'different_rate':sum(cells.values())/len(cells) if cells else None,
        'goal_reps':sum(r.get('goal',{}).get('type') not in ('none',None) for r in reps if not r['excluded']),
        'rep_count':len(reps),'excluded':[{ 'rep':r['rep'],'reason':r['exclude_reason']} for r in reps if r['excluded']],
        'goals':[{ 'rep':r['rep'],'goal':r.get('goal')} for r in reps],
        'tempo_judged':len(tempo),'tempo_different':sum(t['k']>1 for t in tempo),
        'tempo_rate':sum(t['k']>1 for t in tempo)/len(tempo) if tempo else None}


def current_stats(f):
    s=legacy_stats(f);p=f['posture_judgments'];t=[c for st in f['tempo_comparison'] for c in st['comparisons'] if c['result']!='skip']
    s.update(judged=p['judged_count'],unjudged=p['unjudged_count'],different=p['different_count'],
             different_rate=p['different_count']/p['judged_count'] if p['judged_count'] else None,reason_counts=p['reason_counts'],
             set_sentences=[x['sentence'] for x in f['set_findings']],tempo_judged=len(t),
             tempo_different=sum(c['result']=='different' for c in t),
             tempo_rate=sum(c['result']=='different' for c in t)/len(t) if t else None,
             tempo_sentences=[s['text'] for st in f['tempo_comparison'] for s in st['sentences']])
    return s


def strip_sprite(value):
    if isinstance(value,dict):return {k:strip_sprite(v) for k,v in value.items() if k!='sprite'}
    if isinstance(value,list):return [strip_sprite(v) for v in value]
    return value


def experiments(source,out):
    from copy import deepcopy
    from feedback.set_judge import compare_plate_reps,plate_summaries
    from feedback.compare import compare_reps
    from feedback.select import select_goals
    from feedback.export import export_report
    from core.io.rows import read_rows
    from core.compare.summary import SetSummary,Series
    base=source.parent;cache={}
    names={'public':('public','public',3),'squat':('stu10_70','stu8_67',1),
           'bench':('stu2_10_edited','stu2_10_edited',1),'dead':('stu7_1_1','deadlift_08',1)}
    def context(name):
        if name not in cache:
            p=base/'observations'/name/'plate'
            cache[name]=(read_rows(p/'rows.jsonl.gz'),read(p/'result.json'))
        return cache[name]
    pairs={}
    for name,(u,r,rep) in names.items():
        p=base/'after'/name;dest=out/'pairs'/name
        measured={role:read(p/f'metrics_{role}.json') for role in ('user','ref')}
        previous=read(p/'report.json');angles=previous['meta']['angles']
        f=select_goals(compare_plate_reps(measured['user'],measured['ref'],rep,angles=angles,
                            plate_data={'user':context(u),'ref':context(r)}))
        write(dest/'findings.json',f)
        results={role:{'segments':read(p/f'segments_{role}.json'),
                       'pose':dict(np.load(p/f'pose_{role}.npz',allow_pickle=False))} for role in ('user','ref')}
        report=export_report(results,measured,f,dest,warnings=previous['meta']['warnings'],angles=angles)
        write(dest/'report_data.json',strip_sprite(report))
        before=legacy_stats(read(p/'findings.json'));before['unjudged']=len(f['posture_judgments']['cells'])-before['judged']
        pairs[name]={'videos':[u,r],'ref_rep':rep,'angles':angles,'before':before,'after':current_stats(f)}
        print('pair',name,pairs[name]['after']['judged'],flush=True)
    write(out/'pair_results.json',pairs)
    false={}
    for path in sorted(source.glob('*/metrics_user.json')):
        metrics=read(path);segments=read(path.parent/'segments_user.json')
        if segments['plate_rep_count']<4:continue
        original=plate_summaries(*context(path.parent.name))
        arms=[];summaries=[]
        for parity in (1,0):
            arm=deepcopy(metrics);arm['reps']=[r for r in arm['reps'] if r['rep']%2==parity]
            for rep in arm['reps']:rep['rep']=(rep['rep']+1)//2
            arms.append(arm);sets=[]
            for st,ids in original:
                chosen=[n for n in ids if n%2==parity]
                if not chosen:continue
                start=min(chosen);ss=SetSummary(st.exercise,st.fps,len(chosen))
                for key,ser in st.series.items():
                    selected=[(ids[i-1],v) for i,v in zip(ser.reps,ser.vals) if ids[i-1]%2==parity]
                    # Compact sampling arm while preserving holes left by unavailable values.
                    ss.series[key]=Series(key,[(n-start)//2+1 for n,v in selected],np.asarray([v for n,v in selected]))
                sets.append((ss,[(n+1)//2 for n in chosen]))
            summaries.append(sets)
        if not arms[0]['reps'] or not arms[1]['reps']:continue
        ref_rep=next((r['rep'] for r in arms[1]['reps'] if not r['excluded']),arms[1]['reps'][0]['rep'])
        old=select_goals(compare_reps(*arms,ref_rep,allow_excluded_reference=True))
        new=select_goals(compare_plate_reps(*arms,ref_rep,summaries=tuple(summaries)))
        b,a=legacy_stats(old),current_stats(new)
        for stats in (b,a):stats['goal_rep_rate']=stats['goal_reps']/stats['rep_count'] if stats['rep_count'] else None
        false[path.parent.name]={'exercise':metrics['exercise'],'counted_reps':segments['plate_rep_count'],
            'odd_original':[r['rep'] for r in metrics['reps'] if r['rep']%2],
            'even_original':[r['rep'] for r in metrics['reps'] if not r['rep']%2], 'before':b,'after':a}
        print('false',path.parent.name,a['judged'],a['different_rate'],flush=True)
    result={'method':'Odd/even arms reindexed in sampling order, preserving missing values. Before set cell = any eligible user rep over v4 threshold vs selected even ref. After = native set judgment. Before tempo = rep x down/bottom/up; after tempo = set x core down/far_pause. Undefined rates are null.', 'videos':false}
    write(out/'false_findings.json',result)
    return pairs,result

def simulation(out):
    """Run the fixed existing simulation functions for before/after JSON comparison."""
    from multiprocessing import get_context
    from feedback import plate
    from sim import run_compare as sim
    out.mkdir(parents=True,exist_ok=True)
    sim.RES=str(out)
    with get_context('fork').Pool(6) as pool:
        sim.part_table()
        sim.part_normal(5000,pool)
        sim.part_gen(100,pool)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--experiments',action='store_true')
    p.add_argument('--sim',action='store_true')
    a=p.parse_args()
    if a.sim:simulation(a.out)
    else:
        if a.source is None:p.error('--source is required unless --sim')
        stage0(a.source,a.out)
        if a.experiments:experiments(a.source,a.out)
