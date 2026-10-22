"""Plate-mode set judgments; labels/model keep the original rep comparison."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict

import numpy as np

from feedback import config
from feedback.compare import compare_reps, _reference_available_ratio
from feedback import plate  # plate project's core package path
from core.compare.summary import Series, SetSummary, summarize
from core.compare.judge import compare_series_ref, run_compare, Compare
from core.config import DEFAULT
from core.text.templates import render


def series_from_reps(metrics, where, metric, allowed):
    selected=[r for r in metrics['reps'] if r['rep'] in allowed
              and r['positions'][where]['values'].get(metric) is not None]
    selected.sort(key=lambda r:r['rep'])
    return Series(metric,[r['rep'] for r in selected],
                  np.asarray([r['positions'][where]['values'][metric] for r in selected],float))


def plate_summaries(rows, result):
    """Rebuild the unchanged plate summaries, including far_gap confidence.

    result.json alone omits far_gap. Replaying core signal/session/value functions
    on its rows preserves that exclusion and uses continuous endpoint times,
    rather than the quantized phase lengths displayed in the feedback panel.
    """
    from core.signal.height import height_signal
    from core.signal.final import clean_final
    from core.judge.session import run_session, REST
    from core.measure.times import all_rep_values

    exercise=result['exercise'];fps=result['fps']
    t,y,x=height_signal(rows,result.get('plate_mm',450))
    if not len(t):return []
    final=clean_final(t,y,x,fps,DEFAULT.final)
    sig=final.sig
    sets=run_session(sig.t,sig.y,sig.x,sig.gap_before,exercise,'final',cfg=DEFAULT)
    output=[];offset=0
    for st in sets:
        vals=all_rep_values(sig,final.raw_t,st.reps,REST[exercise],DEFAULT.speed)
        summary=summarize(exercise,fps,st.reps,vals)
        output.append((summary,list(range(offset+1,offset+1+len(st.reps)))))
        offset+=len(st.reps)
    return output


def angle_mismatch(angles):
    return bool(angles) and {angles.get(role,{}).get('view') for role in ('user','ref')}=={'side','front'}


def reason_text(c, unit):
    if c.reason=='unstable_user':
        amount=c.s_a*100 if unit=='ratio' else c.s_a
        return f"rep마다 값이 크게 달라 비교하지 않음 (산포 {amount:.1f}{'%' if unit=='ratio' else '°'})"
    if c.reason=='angle_mismatch':return '촬영 각도가 달라 자세는 비교하지 않음'
    if c.reason=='advisory':return '보조 지표는 자세 판정에서 제외'
    if c.reason=='display_only':return '최소 변화량 0인 표시용 지표'
    if c.reason=='ref_excluded':return '선택한 기준 rep이 제외되어 자세는 비교하지 않음'
    if c.reason=='ref_too_short':return f'기준 값 부족 ({c.n_b}개, 최소 {c.need}개)'
    if c.reason=='set_too_short':return f'사용자 값 부족 ({c.n_a}개, 최소 {c.need}개)'
    if c.reason=='no_scatter':return ('사용자' if c.side=='mine' else '기준')+' 값이 3개 연속되지 않아 산포 계산 불가'
    return None


def set_sentence(cell, exercise):
    n=round(abs(cell['delta'])*(100 if cell['unit']=='ratio' else 1))
    direction='+' if cell['delta']>0 else '-'
    phrase=config.METRIC_TEMPLATES[cell['metric']][direction].format(n=n)
    return config.SINGLE_TEMPLATE.format(where=config.position_labels(exercise)[cell['where']],phrase=phrase)


def _eligible(metrics):
    return {r['rep'] for r in metrics['reps'] if not r['excluded']
            and _reference_available_ratio(r,metrics['exercise'])>=config.MIN_REP_METRIC_RATIO}


def _tempo_comparison(mine, reference, user_set, ref_set):
    comparisons=[c for c in run_compare(mine,reference) if c.kind=='vs_ref']
    sentences=render(comparisons,mine.exercise,reference.exercise)
    return {'user_set':user_set,'ref_set':ref_set,'comparisons':[asdict(c) for c in comparisons],
            'sentences':[{'key':s.key,'text':s.text,'kind':s.kind,'shown':s.shown} for s in sentences]}


def compare_plate_reps(user, ref, ref_rep=1, *, angles=None, plate_data=None, summaries=None):
    """Preserve measured rep cells, while admitting goals only after set evidence.

    summaries is an optional pair of [(SetSummary, global rep ids)] for offline
    evidence tests. Live callers supply rows/result pairs through plate_data.
    """
    findings=compare_reps(user,ref,ref_rep,allow_excluded_reference=True)
    findings.update(source='plate',tempo_thresholds={},set_findings=[],tempo_comparison=[])
    exercise=user['exercise']
    if summaries is None and plate_data is not None:
        summaries=tuple(plate_summaries(*plate_data[role]) for role in ('user','ref'))
    if summaries is None:
        # Posture metrics can be compared independently of plate tempo metadata.
        # Missing raw plate summaries never silently enable the old tempo rule.
        summaries=([ (SetSummary(exercise,user.get('fps',0),len(user['reps'])),[r['rep'] for r in user['reps']]) ],
                   [ (SetSummary(exercise,ref.get('fps',0),len(ref['reps'])),[r['rep'] for r in ref['reps']]) ])
    user_sets,ref_sets=summaries
    ref_index=next((i for i,(_,ids) in enumerate(ref_sets) if ref_rep in ids),None)
    if ref_index is None:
        raise ValueError(f'기준 rep {ref_rep}이 속한 원판 세트 없음')
    ref_summary,ref_ids=ref_sets[ref_index]
    reference_excluded=ref_rep not in _eligible(ref)
    allowed_user={r['rep'] for r in findings['reps'] if not r['excluded']}
    allowed_ref=_eligible(ref).intersection(ref_ids)
    mismatch=angle_mismatch(angles)
    set_cells=[];lookup={};rep_sets={}
    for set_index,(summary,ids) in enumerate(user_sets,1):
        allowed=allowed_user.intersection(ids)
        for rep_id in ids:rep_sets[rep_id]=set_index
        for where in config.POSITIONS:
            for metric in config.DISPLAY_METRICS[exercise]:
                unit='ratio' if metric in config.RATIO_METRICS else 'deg'
                minimum=config.DELTA_THRESHOLDS[exercise][metric]
                a=series_from_reps(user,where,metric,allowed)
                b=series_from_reps(ref,where,metric,allowed_ref)
                # Local rep slots preserve genuine missing observations within
                # an actual set; a different set never supplies a scatter triple.
                offset=min(ids)-1 if ids else 0
                ref_offset=min(ref_ids)-1 if ref_ids else 0
                a.reps=[n-offset for n in a.reps];b.reps=[n-ref_offset for n in b.reps]
                c=Compare('vs_ref',metric,'skip',n_a=a.n,n_b=b.n)
                if mismatch:c.reason='angle_mismatch'
                elif minimum<=0:c.reason='display_only'
                elif metric in config.ADVISORY_METRICS.get(exercise,set()):c.reason='advisory'
                elif reference_excluded:c.reason='ref_excluded'
                else:
                    floor=.03 if unit=='ratio' else 3.0
                    c=compare_series_ref(a,b,metric,floor_a=floor,floor_b=floor,min_change=minimum,
                                         max_user_scatter=2*minimum)
                cell={'set':set_index,'ref_set':ref_index+1,'where':where,'metric':metric,'unit':unit,
                      **asdict(c),'reason_text':reason_text(c,unit),
                      'user_reps':[r['rep'] for r in user['reps'] if r['rep'] in allowed and r['positions'][where]['values'].get(metric) is not None],
                      'ref_reps':[r['rep'] for r in ref['reps'] if r['rep'] in allowed_ref and r['positions'][where]['values'].get(metric) is not None],
                      'critical_ratio':c.ratio}
                set_cells.append(cell);lookup[set_index,where,metric]=cell
        findings['tempo_comparison'].append(_tempo_comparison(summary,ref_summary,set_index,ref_index+1))
    for rep in findings['reps']:
        skipped=0
        for cell in rep['cells']:
            stat=lookup.get((rep_sets.get(rep['rep']),cell['where'],cell['metric']))
            cell['set_judgment']=stat
            eligible=False
            if stat is not None and stat['result']=='different' and cell['diff'] is not None:
                eligible=(cell['diff']*stat['delta']>0 and abs(cell['diff'])>=stat['min_change'])
            cell['goal_eligible']=eligible and not rep['excluded'] and cell['status']=='ok'
            if not cell['goal_eligible']:cell['color']='none'
            skipped+=stat is None or stat['result']=='skip'
        rep['unjudged_cell_count']=skipped
        rep['tempo_goal']=None
        # Keep panel numbers/differences, with no legacy rep tempo decision.
        for item in rep['tempo']:item.update(threshold=None,k=None,color='none')
    found=[cell for cell in set_cells if cell['result']=='different']
    found.sort(key=lambda c: (-(c['critical_ratio'] or 0),c['set'],config.POSITIONS.index(c['where']),
                             config.DISPLAY_METRICS[exercise].index(c['metric'])))
    findings['set_findings']=[{**cell,'sentence':set_sentence(cell,exercise)} for cell in found[:3]]
    reasons=Counter(c['reason'] for c in set_cells if c['result']=='skip')
    findings['posture_judgments']={'cells':set_cells,'judged_count':sum(c['result']!='skip' for c in set_cells),
        'unjudged_count':sum(c['result']=='skip' for c in set_cells),'different_count':len(found),'reason_counts':dict(reasons),
        'angle_mismatch':mismatch}
    return findings
