"""Finite JSON and required v3 data shapes checked before report export."""
import math
VIDEO_KEYS='fps width height duration t plate depth reps plate_used crop geo j2 kp2 j3names j3'.split()
REP_KEYS='start mid_down far far_in far_out mid_up back rom_cm v down pause up n'.split()
COMMON_KEYS='sentences set_median v_loss summary inputs exercise schema_version findings kpi show_3d pose3d_scale'.split()

def validate_result(b):
    def req(ok,name):
        if not ok: raise ValueError('Invalid result_v3: '+name)
    def num(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
    def nums(x,n=None): return isinstance(x,list) and (n is None or len(x)==n) and all(num(v) for v in x)
    req(isinstance(b,dict) and all(k in b for k in COMMON_KEYS),'common keys')
    req(b['schema_version']==3,'schema_version')
    req(b['exercise'] in ('squat','benchpress','deadlift'),'exercise')
    req(isinstance(b['show_3d'],bool) and b['show_3d']==(b['exercise']!='benchpress'),'show_3d')
    req(b['pose3d_scale'] in ('input_height','model_native'),'pose3d_scale')
    req(isinstance(b['summary'],str) and (b['v_loss'] is None or num(b['v_loss'])),'summary / v_loss')
    for k in ('plate_mm','load_kg','height_cm'):
        req(k in b['inputs'] and (b['inputs'][k] is None and k!='plate_mm' or num(b['inputs'][k]) and b['inputs'][k]>0),'inputs.'+k)
    for role in ('user','ref'):
        v=b.get(role)
        req(isinstance(v,dict) and all(k in v for k in VIDEO_KEYS),role+' keys')
        for k in ('fps','width','height','duration'): req(num(v[k]) and v[k]>0,role+'.'+k)
        req(nums(v['t']) and len(v['t'])>0 and all(a<c for a,c in zip(v['t'],v['t'][1:])),role+'.t')
        n=len(v['t'])
        req(nums(v['depth'],n) and min(v['depth'])>=0,role+'.depth')
        req(nums(v['crop'],4) and 0<=v['crop'][0]<v['crop'][2]<=v['width'] and 0<=v['crop'][1]<v['crop'][3]<=v['height'],role+'.crop')
        for k in ('plate','kp2','j3'): req(isinstance(v[k],list) and len(v[k])==n,role+'.'+k)
        req(all(q is None or nums(q,3) for q in v['plate']),role+'.plate points')
        for k,names in (('kp2','j2'),('j3','j3names')):
            req(isinstance(v[names],list) and all(isinstance(j,int) for j in v[names]),role+'.'+names)
            req(all(q is None or isinstance(q,list) and len(q)==len(v[names]) and all(p is None and k=='kp2' or nums(p,3) for p in q) for q in v[k]),role+'.'+k+' points')
        req(isinstance(v['reps'],list) and len(v['reps'])>0,role+'.reps')
        for i,r in enumerate(v['reps'],1):
            req(all(k in r and (r[k] is None and k=='v' or num(r[k])) for k in REP_KEYS),role+'.rep values')
            req(r['n']==i and r['start']<=r['far_in']<=r['far_out']<=r['back'],role+'.rep events')
        req(num(v['plate_used']) and 0<=v['plate_used']<=100,role+'.plate_used')
        req(isinstance(v['geo'],dict) and isinstance(v.get('geo_check'),dict),role+'.geo')
    req(isinstance(b['findings'],dict) and all(isinstance(b['findings'].get(k),list) for k in ('cards','extra','outliers')),'findings')
    req(isinstance(b['kpi'],dict) and all(k in b['kpi'] for k in ('d','se','lim','k','mu','mr','hit')),'kpi')
    req(isinstance(b['sentences'],list) and isinstance(b['set_median'],dict),'sentences / set_median')
    def tree(x):
        if isinstance(x,float): req(math.isfinite(x),'nonfinite JSON number')
        elif isinstance(x,dict):
            for v in x.values(): tree(v)
        elif isinstance(x,list):
            for v in x: tree(v)
    tree(b)
    return b
