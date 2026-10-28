"""Package reviewed M1b runs with M0/M1 comparisons and exact plate checks."""
from pathlib import Path
import argparse
from collections import Counter
import json
import zipfile
from eval.baseline import render
from eval.m1b_audit import read, invariance


def fmt(x):return 'N/A' if x is None else f'{x:.4f}'


def table(reports):
    lines=['## 설정별 전체·종목별 비교','',
           '|범위|설정|오류 영상|coverage|점프|홀짝 다른/판정 칸|fps|',
           '|---|---|---:|---:|---:|---:|---:|']
    for scope in ('all','squat','benchpress','deadlift'):
        for name,r in reports.items():
            s=r['summary'][scope];f=s['false_difference']
            errors=sum(v['lifter_review'].startswith('wrong') for v in r['videos'] if scope=='all' or v['exercise']==scope)
            lines.append(f'|{scope}|{name}|{errors}|{fmt(s["coverage"])}|{s["jumps"]}|{f["different"]}/{f["judged"]}|{fmt(s["timing"]["fps"])}|')
    lines+=['','### 횟수 (원본 61개)','',
            '|범위|설정|MAE|exact|OBO|event F1|',
            '|---|---|---:|---:|---:|---:|']
    for scope in ('all','squat','benchpress','deadlift'):
        for name,r in reports.items():
            c=r['summary'][scope]['counts']
            lines.append('|'+scope+'|'+name+'|'+'|'.join(fmt(c[k]) for k in ('count_mae','count_exact','obo','event_f1'))+'|')
    lines+=['','### 단계별 CPU 시간 (초)','',
            '|범위|설정|디코드|원판 검출|사람 검출|KLT|포즈|나머지|합계|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope in ('all','squat','benchpress','deadlift'):
        for name,r in reports.items():
            t=r['summary'][scope]['timing']
            lines.append('|'+scope+'|'+name+'|'+'|'.join(fmt(t.get(k)) for k in ('decode','plate_detection','person_detection','klt','pose','rest','total'))+'|')
    lines+=['','### 관절 묶음 visibility ≥0.5','',
            '|범위|설정|어깨|팔꿈치|손목|엉덩이|무릎|발목|발|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope in ('all','squat','benchpress','deadlift'):
        for name,r in reports.items():
            lines.append('|'+scope+'|'+name+'|'+'|'.join(fmt(v['rate']) for v in r['summary'][scope]['joints'].values())+'|')
    lines+=['','### Rep 간 흔들림 (도)','',
            'M0와 같은 영상별 SD(ddof=1)와 모든 rep 쌍 절대 차이 중앙값, 그 뒤 범위별 영상 중앙값이다. '
            'n은 표본 2개 이상인 영상 수다. 포즈 가용성에 따라 모집단이 다르므로 SD만으로 정확도를 주장하지 않는다.',
            '', '|범위|포지션|지표|설정|n|SD|쌍 차이|','|---|---|---|---|---:|---:|---:|']
    for scope in ('all','squat','benchpress','deadlift'):
        for name,r in reports.items():
            for c in r['summary'][scope]['variation']:
                lines.append(f'|{scope}|{c["position"]}|{c["metric"]}|{name}|{c["videos"]}|{fmt(c["sd"])}|{fmt(c["pair_median"])}|')
    return '\n'.join(lines)+'\n'


def refresh(path):
    r=read(path/'baseline.json');reviews=read(path/'lifter_review.json')
    assert set(reviews)=={v['key'] for v in r['videos']}, f'Unreviewed sheets: {path}'
    assert all(v.startswith(('correct','wrong')) for v in reviews.values())
    for v in r['videos']:v['lifter_review']=reviews[v['key']]
    (path/'baseline.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    markdown=render(r)
    (path/'baseline.md').write_text(markdown)
    (path/'REPORT.md').write_text(markdown)
    with zipfile.ZipFile(path/'baseline.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ('baseline.json','baseline.md'):z.write(path/name,name)
        for v in r['videos']:z.write(path/v['image'],v['image'])
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m0',type=Path,required=True);p.add_argument('--m1',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--notes',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();reports={'M0':read(a.m0/'baseline.json')}
    reports.update({'M1 '+m:read(a.m1/m/'baseline.json') for m in ('lightweight','balanced')})
    checks={}
    for name in ('n3','n5','n10','tiny_m'):
        reports[name]=refresh(a.runs/name)
        assert len(reports[name]['videos'])==69
        assert {v['key'] for v in reports[name]['videos']}=={v['key'] for v in reports['M0']['videos']}
        assert all(v['image_frames']==next(x['image_frames'] for x in reports['M1 balanced']['videos'] if x['key']==v['key']) for v in reports[name]['videos'])
        checks[name]={label:invariance(base,a.runs/name) for label,base in [('M0',a.m0),('M1 balanced',a.m1/'balanced')]}
    guard=refresh(a.runs/'bench_guard');assert len(guard['videos'])==16
    checks['bench_guard']={label:invariance(base,a.runs/'bench_guard') for label,base in [('M0',a.m0),('M1 balanced',a.m1/'balanced')]}
    assert all(c['exact'] for modes in checks.values() for c in modes.values()), 'Plate results changed'
    (a.runs/'invariance.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    passed=[]
    for name in ('n3','n5','n10','tiny_m'):
        r=reports[name];errors=sum(v['lifter_review'].startswith('wrong') for v in r['videos'])
        if errors<=2 and r['summary']['all']['timing']['fps']>=4:passed.append(name)
    chosen=max(passed,key=lambda name:reports[name]['summary']['all']['timing']['fps']) if passed else 'lightweight (unchanged)'
    lead=['# M1b 포즈 속도·정확도 조합 결과','',
          f'완료 기준(리프터 오류 ≤2개, 1스레드 ≥4 fps, 원판 불변)을 함께 통과한 설정: **{", ".join(passed) or "없음"}**. 기본 모드: **{chosen}**.',
          '', '원판 69개 기록·결과(실행 시간 두 필드 제외), 횟수 61개가 M0와 M1 balanced 양쪽에 일치한다. 준비 영상 SHA256과 확인 이미지 표본 프레임도 모두 같다. 새 확인 이미지 292장(4×69 + 벤치 대조 16장)을 검토했다.',
          '', '|설정|표본 오류 영상|coverage|1스레드 fps|', '|---|---:|---:|---:|']
    for name,r in reports.items():
        errors=sum(v['lifter_review'].startswith('wrong') for v in r['videos'])
        summary=r['summary']['all']
        lead.append(f'|{name}|{errors}|{fmt(summary["coverage"])}|{fmt(summary["timing"]["fps"])}|')
    notes=a.notes.read_text().split('\n',1)[1].lstrip()
    text='\n'.join(lead)+'\n\n## 작업 범위와 재현 조건\n\n'+notes+'\n\n'+table(reports)
    text+='\n## M1 balanced 대비 새 오류 영상\n\n'
    olderrors={v['key'] for v in reports['M1 balanced']['videos'] if v['lifter_review'].startswith('wrong')}
    for name in ('n3','n5','n10','tiny_m'):
        errors={v['key'] for v in reports[name]['videos'] if v['lifter_review'].startswith('wrong')}
        text+=f'- {name}: '+(', '.join(sorted(errors-olderrors)) or '새 오류 영상 없음')+'\n'
    text+='\n기존 오류 영상의 새로운 오류 프레임은 아래 영상별 검토 문구에 따로 기록한다.\n'
    text+='\n## 검출 호출과 추적 누락\n\n|설정|전체 프레임|검출 프레임|검출 비율|누락 판정 프레임|\n|---|---:|---:|---:|---:|\n'
    for name in ('n3','n5','n10','tiny_m'):
        videos=reports[name]['videos'];frames=sum(v['frames'] for v in videos)
        detections=sum(len(v['tracking']['detection_frames']) for v in videos)
        losses=sum(len(v['tracking']['lost_frames']) for v in videos)
        text+=f'|{name}|{frames}|{detections}|{fmt(detections/frames)}|{losses}|\n'
    text+='\n누락 프레임은 몸통/손목 지원 부족 등으로 다음 프레임 재검출을 예약한 횟수다. 현재 포즈 valid=False 수와는 다르다. tiny_m은 추적을 하지 않아 이 수가 0이다.\n'
    text+='\n## 리프터 오류 영상과 이미지\n\n'
    for name,r in [*reports.items(),('bench_guard',guard)]:
        text+='\n### '+name+'\n\n|영상|검토 결과|이미지|\n|---|---|---|\n'
        for v in r['videos']:
            if v['lifter_review'].startswith('wrong'):text+=f'|{v["key"]}|{v["lifter_review"]}|{name.replace(" ", "_")}/{v["image"]}|\n'
    text+='\n## 벤치 규칙만 바꾼 16개 비교\n\n|설정|오류 영상|coverage|몸통 점프|\n|---|---:|---:|---:|\n'
    for name,r in [('M1 balanced (전)',reports['M1 balanced']),('balanced guard (후)',guard)]:
        s=r['summary']['benchpress'];errors=sum(v['lifter_review'].startswith('wrong') for v in r['videos'] if v['exercise']=='benchpress')
        text+=f'|{name}|{errors}|{fmt(s["coverage"])}|{s["jumps"]}|\n'
    text+='\n### 벤치 16개 영상별 coverage와 오류\n\n|영상|coverage 전|coverage 후|오류 전|오류 후|후 확인 이미지|\n|---|---:|---:|---|---|---|\n'
    oldbench={v['key']:v for v in reports['M1 balanced']['videos'] if v['exercise']=='benchpress'}
    for v in guard['videos']:
        old=oldbench[v['key']]
        text+=f'|{v["key"]}|{fmt(old["observation"]["coverage"])}|{fmt(v["observation"]["coverage"])}|{old["lifter_review"].startswith("wrong")}|{v["lifter_review"].startswith("wrong")}|bench_guard/{v["image"]}|\n'
    text+='\n## 손가락 의존성 수정 전후 기준 rep 자격\n\n'
    text+='같은 raw/smoothed 포즈와 segmentation에 대해 필요 관절만 바꿔 native `_eligible` 결과를 센다. 모든 rep 중 실제 세트 비교에서 기준으로 선택할 수 있는 rep 수이며, 홀짝 진단의 판정 칸 수와 다르다.\n\n'
    text+='|설정|종목|수정 전|수정 후|\n|---|---|---:|---:|\n'
    eligibility=read(a.runs/'m1_reference_eligibility.json')
    sources={**{'M1 '+m:entries for m,entries in eligibility.items()},
             **{name:[dict(exercise=v['exercise'],**v['reference_eligibility']) for v in r['videos']] for name,r in reports.items() if name in ('n3','n5','n10','tiny_m')},
             'bench_guard':[dict(exercise=v['exercise'],**v['reference_eligibility']) for v in guard['videos']]}
    for name,entries in sources.items():
        for ex in ('benchpress','deadlift'):
            if not any(v['exercise']==ex for v in entries):continue
            before,after=[sum(len(v[k]) for v in entries if v['exercise']==ex) for k in ('before','after')]
            text+=f'|{name}|{ex}|{before}|{after}|\n'
    text+='\n## ORT 6스레드: 60초·30 fps 영상\n\n|설정|원판+포즈 처리 초|지표 포함 전체 초|처리 fps|\n|---|---:|---:|---:|\n'
    benchmark=read(a.runs/'threads6/benchmark.json')
    assert len(benchmark)==6 and {b['name'] for b in benchmark}=={'balanced','n3','n5','n10','tiny_m','lightweight'}, 'Finish all ORT6 runs first'
    assert all(b['threads']==6 and b['frames']==1800 and b['fps']==30 for b in benchmark)
    assert len({b['video_sha256'] for b in benchmark})==1
    for b in benchmark:text+=f'|{b["name"]}|{fmt(b["processing_s"])}|{fmt(b["end_to_end_s"])}|{fmt(b["processing_fps"])}|\n'
    text+='\n입력 SHA256: `'+benchmark[0]['video_sha256']+'`. 단계별 시간과 영상 경로는 `threads6/benchmark.json`에 있다.\n'
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(text)
    (a.runs/'REPORT.md').write_text(text)
    with zipfile.ZipFile(a.runs/'m1b-pose-speed.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.write(a.report,'REPORT.md')
        for name in ('invariance.json','m1_reference_eligibility.json','evaluator_sources.json'):z.write(a.runs/name,name)
        z.write(a.runs/'threads6/benchmark.json','threads6/benchmark.json')
        for name,path in [('M0',a.m0),('M1_lightweight',a.m1/'lightweight'),('M1_balanced',a.m1/'balanced'),*[(n,a.runs/n) for n in ('n3','n5','n10','tiny_m','bench_guard')]]:
            for file in ('baseline.json','baseline.md','lifter_review.json'):z.write(path/file,name+'/'+file)
            for img in sorted((path/'images').glob('*.jpg')):z.write(img,name+'/images/'+img.name)
            for pattern in ('*.jpg','*.json'):
                for extra in sorted((path/'details').glob(pattern)):z.write(extra,name+'/details/'+extra.name)
    print(json.dumps(dict(passed=passed,chosen=chosen,plate_invariance=True),ensure_ascii=False))


if __name__=='__main__':main()
