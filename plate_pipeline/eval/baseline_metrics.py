"""Model-independent M0 statistics. Pose contract: normalized MP33 Nx33x4 NPZ.

lms is the existing smoothed sequence; raw_lms and valid preserve observations.
No inference or model imports are needed for the array statistics below.
"""
from copy import deepcopy
from itertools import combinations
import numpy as np

POSITIONS = ('START', 'MID_DOWN', 'BOTTOM', 'MID_UP', 'LOCKOUT')
ANGLES = ('knee_angle', 'hip_angle', 'trunk_lean')
JOINTS = dict(shoulder=[11,12], elbow=[13,14], wrist=[15,16], hip=[23,24],
              knee=[25,26], ankle=[27,28], foot=[29,30,31,32])


def ratio(n, d):
    return n / d if d else None


def median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def rep_variation(metrics):
    """S3a: every available position value, including excluded reps, unordered pairs.

    Sample SD (ddof=1); fewer than two reps has no dispersion estimate.
    """
    cells = []
    for position in POSITIONS:
        for metric in ANGLES:
            values = [r['positions'][position]['values'].get(metric) for r in metrics['reps']]
            values = [v for v in values if v is not None and np.isfinite(v)]
            differences = [abs(a-b) for a,b in combinations(values, 2)]
            cells.append(dict(position=position, metric=metric, reps=len(values), pairs=len(differences),
                              sd=float(np.std(values, ddof=1)) if len(values)>1 else None,
                              pair_median=median(differences)))
    return cells


def rep_mask(segments):
    mask = np.zeros(segments['total'], bool)
    for rep in segments['reps']:
        a = max(0, rep['phases']['down'][0])
        b = min(len(mask), rep['phases']['up'][1])
        mask[a:b] = True
    return mask


def observation_stats(pose, mask, rows, visibility=.5):
    raw = np.asarray(pose['raw_lms'])
    valid = np.asarray(pose['valid'], bool)
    if raw.shape != (len(mask),33,4) or valid.shape != mask.shape or len(rows)!=len(mask):
        raise ValueError('Pose, rep mask and plate rows must share the same frame clock')
    frames = int(mask.sum())
    selected = int((mask & valid).sum())
    groups = {}
    for name, ids in JOINTS.items():
        good = np.isfinite(raw[:,ids,3]) & (raw[:,ids,3]>=visibility) & valid[:,None]
        numerator = int(good[mask].sum())
        denominator = frames * len(ids)
        groups[name] = dict(visible=numerator, total=denominator, rate=ratio(numerator,denominator))
    # Torso center uses both shoulders and hips; missing torso support cannot
    # establish an identity jump. Only consecutive decoded frames are compared.
    torso = raw[:,[11,12,23,24],:2] * [pose['w'],pose['h']]
    centers = torso.mean(axis=1)
    supported = valid & np.isfinite(torso).all(axis=(1,2))
    jumps = []
    for i in range(1,len(mask)):
        if not (mask[i-1] and mask[i] and supported[i-1] and supported[i]):
            continue
        plate = rows[i].get('plate',{})
        box = plate.get('box')
        if not plate.get('used') or box is None or box[3]<=box[1]:
            continue
        distance = float(np.linalg.norm(centers[i]-centers[i-1]))
        if distance > box[3]-box[1]:
            jumps.append(dict(frame=i, distance_px=distance, plate_height_px=box[3]-box[1]))
    return dict(rep_frames=frames, pose_frames=selected, coverage=ratio(selected,frames),
                jumps=len(jumps), jump_frames=jumps, joints=groups)


def odd_even(metrics, original_summaries):
    """Same split/reindex and native set judgment as tools/s3b_judge_audit.py."""
    from feedback.set_judge import compare_plate_reps
    from core.compare.summary import SetSummary, Series
    arms, summaries = [], []
    for parity in (1,0):
        arm = deepcopy(metrics)
        arm['reps'] = [r for r in arm['reps'] if r['rep']%2==parity]
        for rep in arm['reps']:
            rep['rep'] = (rep['rep']+1)//2
        arms.append(arm)
        sets = []
        for st, ids in original_summaries:
            chosen = [n for n in ids if n%2==parity]
            if not chosen:
                continue
            ss = SetSummary(st.exercise,st.fps,len(chosen))
            for key, ser in st.series.items():
                selected = [(ids[i-1],v) for i,v in zip(ser.reps,ser.vals) if ids[i-1]%2==parity]
                ss.series[key] = Series(key,[(n-min(chosen))//2+1 for n,v in selected],
                                        np.asarray([v for n,v in selected]))
            sets.append((ss,[(n+1)//2 for n in chosen]))
        summaries.append(sets)
    if len(metrics['reps'])<4 or not all(a['reps'] for a in arms):
        return dict(judged=0,different=0,rate=None,reason='fewer than four measurable completed reps')
    ref = next((r['rep'] for r in arms[1]['reps'] if not r['excluded']),arms[1]['reps'][0]['rep'])
    findings = compare_plate_reps(*arms,ref,summaries=tuple(summaries))
    p = findings['posture_judgments']
    return dict(judged=p['judged_count'],different=p['different_count'],
                rate=ratio(p['different_count'],p['judged_count']),
                unjudged=p['unjudged_count'],reasons=p['reason_counts'],cells=p['cells'])


def measure(pose, rows, result, video, exercise):
    """Use unchanged feedback segmentation/measurement and extend angle audit only."""
    from feedback.plate import build_plate_segments
    from feedback.metrics import build_metrics, _frame_sides, _frame_metrics, _position_values
    from feedback.quality import observation_reason, VISIBILITY_MIN
    from feedback.set_judge import plate_summaries
    if not any(st['rep_values'] for st in result['sets']):
        segments = dict(video=str(video),fps=pose['fps'],size=[pose['w'],pose['h']],
                        total=pose['total'],exercise=exercise,view='side',facing=None,
                        source='plate',reps=[],plate_rep_count=0,excluded_reps=[])
    else:
        segments = build_plate_segments(pose,dict(video=str(video),view='side'),exercise,rows,result)
    native = build_metrics(pose,segments)
    audit = deepcopy(native)
    sides = _frame_sides(pose['lms'],pose['valid'],segments)
    frames = _frame_metrics(pose['lms'],pose['valid'],exercise,ANGLES,sides)
    for i in range(pose['total']):
        for metric in ANGLES:
            if not pose['valid'][i] or observation_reason(pose['raw_lms'][i],metric,sides[i]) is not None:
                frames[metric][i] = None
    for rep in audit['reps']:
        for position in POSITIONS:
            p = rep['positions'][position]
            p['values'].update(_position_values(frames,p['frame']))
    return dict(variation=rep_variation(audit),
                false_difference=odd_even(native,plate_summaries(rows,result)),
                observation=observation_stats(pose,rep_mask(segments),rows,VISIBILITY_MIN)), segments
