"""v3 set judgments and bookmarks, ported from the frozen template3.html.

Angles are display comparisons, not validated coaching measurements. Missing
frames stay unavailable and never become fabricated zero-angle observations.
"""
from bisect import bisect_left
import math
import re
import numpy as np

POS = ('START', 'MID_DOWN', 'FAR', 'MID_UP', 'LOCKOUT')
METRICS = (('knee', '무릎각', 8), ('hip', '엉덩이각', 8), ('trunk', '몸통 기울기', 6))
POS_P = dict(zip(POS, (0, .245, .5, .755, 1)))


def finite(a):
    return [float(x) for x in a if x is not None and math.isfinite(x)]


def avg(a):
    return sum(a) / len(a) if len(a) else float('nan')


def median(a):
    return float(np.median(a)) if len(a) else float('nan')


def jsround(x):
    return math.floor(x + .5)


def idx_at(v, t):
    i = min(bisect_left(v['t'], t), len(v['t']) - 1)
    if i > 0 and abs(v['t'][i - 1] - t) < abs(v['t'][i] - t):
        i -= 1
    return i


def prep_reps(v):
    for r in v['reps']:
        i0, i1, fi, fo = [idx_at(v, r[k]) for k in ('start', 'back', 'far_in', 'far_out')]
        dmax = max(v['depth'][i0:i1 + 1], default=0)
        pairs, run = [], 0
        for i in range(i0, min(fi, fo) + 1):
            run = max(run, v['depth'][i] / dmax if dmax else 0)
            pairs.append((v['t'][i], .49 * min(1, max(0, run))))
        pairs.extend([(r['far_in'], .49), (max(r['far_out'], r['far_in'] + .001), .51)])
        run = 0
        for i in range(max(fi, fo) + 1, i1 + 1):
            run = max(run, 1 - v['depth'][i] / dmax if dmax else 0)
            pairs.append((v['t'][i], .51 + .49 * min(1, max(0, run))))
        pairs.append((r['back'], 1))
        pairs.sort(key=lambda x: x[0])
        r['_t'], r['_p'], pm = [], [], 0
        for t, p in pairs:
            pm = max(pm, p)
            r['_t'].append(t)
            r['_p'].append(pm)
        r['dmax'] = dmax


def pos_time(r, pos):
    if pos in ('START', 'FAR', 'LOCKOUT'):
        return r[dict(START='start', FAR='far', LOCKOUT='back')[pos]]
    p, ts, ps = POS_P[pos], r['_t'], r['_p']
    lo, hi = 0, len(ps) - 1
    while hi - lo > 1:
        m = (lo + hi) // 2
        if ps[m] < p:
            lo = m
        else:
            hi = m
    return ts[hi] if ps[hi] - ps[lo] < 1e-9 else ts[lo] + (p - ps[lo]) / (ps[hi] - ps[lo]) * (ts[hi] - ts[lo])


def metrics_of(q, names):
    if q is None:
        return None
    pts = dict(zip(names, q))
    if any(pts.get(j) is None or any(x is None for x in pts[j]) for j in (11,12,13,14,23,24,25,26,27,28)):
        return None
    def angle(a, b, c):
        u, v = np.array(pts[a]) - pts[b], np.array(pts[c]) - pts[b]
        n = math.hypot(*u) * math.hypot(*v)
        return math.degrees(math.acos(max(-1, min(1, float(u @ v) / n)))) if n else float('nan')
    v = (np.array(pts[11]) + pts[12] - np.array(pts[23]) - pts[24]) / 2
    return dict(knee=(angle(23,25,27)+angle(24,26,28))/2,
                hip=(angle(11,23,25)+angle(12,24,26))/2,
                trunk=math.degrees(math.atan2(math.hypot(v[0],v[2]),v[1])))


def per_rep(v, pos, key):
    out = []
    for r in v['reps']:
        i = idx_at(v, pos_time(r, pos))
        ms = [metrics_of(v['j3'][max(0,min(len(v['j3'])-1,k))], v['j3names']) for k in range(i-2,i+3)]
        out.append(avg([m[key] for m in ms if m is not None]))
    return out


def judge(a, b, minimum, floor=2):
    a, b = finite(a), finite(b)
    mu, mr = avg(a), avg(b)
    def variance_mean(x):
        return sum((v-avg(x))**2 for v in x)/(len(x)-1)/len(x) if len(x)>1 else 0
    se = max(floor, math.sqrt(variance_mean(a)+variance_mean(b)))
    lim, d = max(3*se, minimum), mu-mr
    return dict(d=d, se=se, lim=lim, k=abs(d)/lim, mu=mu, mr=mr, hit=abs(d)>lim)


def reps_showing(reps, a, b, sign, lim):
    bf = finite(b)
    if not bf:
        return []
    return [r['n'] for r,x in zip(reps,a) if math.isfinite(x) and (x>max(bf)+lim if sign>0 else x<min(bf)-lim)]


def largest_rep(a, b, sign):
    mean, bi, bv = avg(finite(b)), 0, -math.inf
    for i,x in enumerate(a):
        if math.isfinite(x) and sign*(x-mean)>bv:
            bi,bv = i,sign*(x-mean)
    return bi+1


def calculate(bundle, posture_cards=False):
    u = bundle['user']
    r = bundle.get('ref')
    has_ref = r is not None
    # Empty reference observations are not manufactured for single-set reports.
    videos = [('user',u)] + ([('ref',r)] if has_ref else [])
    for _,v in videos:
        prep_reps(v)
    judgments = {p: {k: judge(per_rep(u,p,k),per_rep(r,p,k) if has_ref else [],mn) for k,_,mn in METRICS} for p in POS}
    angles = {f'{p}.{k}': {role: [round(x,2) if math.isfinite(x) else None for x in per_rep(v,p,k)] for role,v in videos} for p in ('MID_DOWN','FAR','MID_UP') for k,_,_ in METRICS}
    times = {role: [dict(n=rep['n'], **{p:pos_time(rep,p) for p in ('MID_DOWN','FAR','MID_UP')}) for rep in v['reps']] for role,v in videos}
    deadlift = bundle.get('exercise') == 'deadlift'
    all_findings = []
    def add(a,b,sign,j,**fields):
        all_findings.append(dict(no=None,key=None,all=None,**fields,reps=reps_showing(u['reps'],a,b,sign,j['lim']),best=largest_rep(a,b,sign),score=j['k']))
    for s in bundle['sentences']:
        if not has_ref or s['kind']!='finding' or not s['key'].startswith('vs_ref'):
            continue
        k = s['key'].split('.')[1]
        f, pos = ('pause','FAR') if k=='far_pause' else (k,'MID_DOWN' if k=='down' else 'MID_UP')
        if deadlift and pos != 'FAR':
            pos = 'MID_UP' if pos == 'MID_DOWN' else 'MID_DOWN'
        a,b = [x[f] for x in u['reps']],[x[f] for x in r['reps']]
        match = re.search(r'\((.*)\)',s['text'])
        add(a,b,1 if median(a)>median(b) else -1,judge(a,b,.25,.05),kind='ref',text=re.sub(r' \(.*\)\.$','',s['text']),val=match[1] if match else '',pos=pos)
    for pos,where in (('FAR','정점에서' if deadlift else '바닥에서'),('MID_UP','내려오는 중간에' if deadlift else '올라오는 중간에')):
        for k,_,mn in METRICS:
            if not posture_cards or not has_ref: continue
            a,b = per_rep(u,pos,k),per_rep(r,pos,k)
            j = judgments[pos][k]
            if not j['hit']:
                continue
            sign,dd = (1 if j['d']>0 else -1),jsround(abs(j['d']))
            if k=='knee': text=f'{where} 무릎을 기준 영상보다 {dd}° {"덜" if sign>0 else "더"} 굽혔습니다'
            elif k=='hip': text=f'{where} 엉덩이를 기준 영상보다 {dd}° {"덜" if sign>0 else "더"} 접었습니다'
            else: text=f'{where} 상체를 기준 영상보다 {dd}° {"더" if sign>0 else "덜"} 숙였습니다'
            add(a,b,sign,j,kind='ref',text=text,val=f'내 세트 {jsround(j["mu"])}°, 기준 {jsround(j["mr"])}°',pos=pos)
            all_findings[-1]['key'] = k
    all_findings.sort(key=lambda x:-x['score'])
    shown = [x for x in all_findings if x['reps']][:3]
    extra = [x for x in all_findings if not any(x is y for y in shown)]
    vs = [x['v'] for x in u['reps']]
    measured = finite(vs)
    loss = 100*(max(measured)-vs[-1])/max(measured) if measured and max(measured)>0 and vs[-1] is not None else 0
    if loss>=10:
        shown.append(dict(kind='inset',key=None,text=f'마지막 rep의 올리는 속도가 가장 빠른 rep보다 {jsround(loss)}% 느렸습니다',val=f'rep {vs.index(max(measured))+1} {max(measured):.2f} → rep {len(vs)} {vs[-1]:.2f} m/s',pos='MID_DOWN' if deadlift else 'MID_UP',reps=[len(vs)],best=len(vs),score=0))
    for i,c in enumerate(shown):
        c.update(no=i+1,all=len(u['reps'])>1 and len(c['reps'])==len(u['reps']))
    outliers = []
    roms = [x['rom_cm'] for x in u['reps']]
    mr,mv = median(roms),median(measured)
    madr,madv = median([abs(x-mr) for x in roms]),median([abs(x-mv) for x in measured])
    sc = bundle.get('inputs',{}).get('plate_mm',450)/450
    for rep,d,v in zip(u['reps'],roms,vs):
        notes = []
        if abs(d-mr)>max(3,3*madr): notes.append(f'깊이 {"−" if d<mr else "+"}{abs(d-mr):.1f} cm')
        if v is not None and abs(v-mv)>max(.05*sc,3*madv): notes.append(f'속도 {v:.2f} m/s')
        if notes: outliers.append(dict(rep=rep['n'],t=rep['far'],text=f'rep {rep["n"]}: 세트 중앙값 대비 '+ ' · '.join(notes)))
    return dict(cards=shown,extra=extra,outliers=outliers,per_rep_angles=angles,pos_times=times),judgments
