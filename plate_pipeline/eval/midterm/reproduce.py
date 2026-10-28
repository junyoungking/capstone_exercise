"""Mandatory v40 split and saved B_generic count/event aggregation gate."""
import ast,csv,hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def aggregate(v40,frame,delays=()):
    """Evaluate the original train.summarize aggregate expressions without training imports."""
    tree=ast.parse((v40/'train.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summarize')
    assignment=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='summary' for t in n.targets))
    wanted={'videos','gt_reps','pred_reps','count_mae','count_exact','obo','event_f1','event_tp','event_fp','event_fn','matched_delay_median_s','matched_abs_delay_p95_s'}
    tp,fp,fn=frame[['tp','fp','fn']].sum()
    scope=dict(np=np,d=frame,tp=tp,fp=fp,fn=fn,delays=list(delays))
    return {key.value:eval(compile(ast.Expression(value),str(v40/'train.py'),'eval'),scope)
            for key,value in zip(assignment.value.keys,assignment.value.values) if key.value in wanted}


def reproduce(v40,out):
    out.mkdir(parents=True,exist_ok=True)
    split=pd.read_csv(v40/'effective_split.csv');test=split[split.split=='test'];info=json.loads((v40/'effective_split_info.json').read_text())
    actual=digest(sorted(zip(split.key,split.split)));test_hash=digest(sorted(zip(test.key,test.split)))
    records=json.loads((v40/'records.json').read_text());record_test={r['key'] for r in records if r['split']=='test'}
    assert len(test)==73 and set(test.key)==record_test and not split.key.duplicated().any()
    # Verify the local labels against the frozen completion triples, using the
    # original parser AST (no source training modules or pose caches imported).
    data_tree=ast.parse((v40/'data.py').read_text())
    parser=next(n for n in data_tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_labels')
    namespace=dict(pd=pd,np=np,re=re)
    exec(compile(ast.Module(body=[parser],type_ignores=[]),str(v40/'data.py'),'exec'),namespace)
    labels=pd.read_csv(v40.parents[1]/'new_labels.csv')
    lookup={str(r['type'])+'/'+str(r['name']):r for r in labels.to_dict('records')}
    for record in records:
        if record['key'] not in record_test:continue
        triples,_,_=namespace['parse_labels'](lookup[record['key']])
        if [list(t) for t in triples]!=record['reps']:raise RuntimeError('local labels differ: '+record['key'])
    checks=[{'check':'effective_split_hash (v40 full partition definition)','actual':actual,'expected':info['effective_split_hash'],'pass':actual==info['effective_split_hash']},
            {'check':'test membership vs records','actual':len(test),'expected':73,'pass':set(test.key)==record_test}]
    for subset in ('reference_test','custom_test','all_test'):
        d=pd.read_csv(v40/f'arms/B_generic/{subset}_per_video.csv')
        old=json.loads((v40/f'arms/B_generic/{subset}_summary.json').read_text());new=aggregate(v40,d)
        for key in ('videos','gt_reps','pred_reps','count_mae','count_exact','obo','event_f1','event_tp','event_fp','event_fn'):
            checks.append({'check':subset+'/'+key,'actual':new[key],'expected':old[key],'pass':bool(np.isclose(new[key],old[key],rtol=0,atol=1e-12))})
        if subset=='all_test':assert set(d.key)==set(test.key)
    pd.DataFrame(checks).to_csv(out/'reproduction_checks.csv',index=False)
    test.to_csv(out/'test_membership.csv',index=False)
    passed=all(c['pass'] for c in checks)
    (out/'reproduction.md').write_text(f'''# 0단계 재현 검증

결과: {'통과' if passed else '불일치: 본 실험 중지'}.

v40 holdout.py의 effective_split_hash는 시험 73개만이 아니라 전체 462개 (key, split)를 정렬해 data.digest로 해시한 값이다. 이 원래 정의를 적용했다. 분할을 새로 만들지 않았다.

- 전체 분할 해시: `{actual}`
- 기준 해시: `{info['effective_split_hash']}`
- 시험 73개만의 별도 해시: `{test_hash}` (위 전체 분할 해시와 정의가 다름)
- test 키는 records.json 및 B_generic all_test_per_video.csv와 정확히 일치. 로컬 new_labels.csv의 원본 parse_labels 결과도 records의 완료 rep 삼중 라벨과 73개 모두 일치.
- B_generic의 reference/custom/all 영상별 CSV를 train.summarize의 원본 집계 AST 식으로 재계산했다. 모델·추론·분할 생성 없이 횟수와 TP/FP/FN/event F1 집계를 검증했다. 원시 예측을 다시 디코딩한 검증은 아니다. saved per-video CSV에는 개별 delay가 없으므로 이 CSV 집계에는 delay를 넣지 않았다. run.py는 원판 추론 전에 cohort.clock_gate를 추가로 실행해 원본 프레임 시계와 저장 count_event로 B의 세 묶음 delay 중앙값·p95까지 재현해야 진행한다.

세부 수치: reproduction_checks.csv.
''')
    if not passed:raise RuntimeError('v40 baseline gate failed; do not run plate evaluation')
    return test

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--v40',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();reproduce(a.v40,a.out)
