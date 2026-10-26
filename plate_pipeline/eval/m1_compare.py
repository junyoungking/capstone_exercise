"""Compare frozen M0 and two M1 runs, assert exact plate invariants, package evidence.

python -m eval.m1_compare --mp /path/baseline_mp --rtm eval_out/baseline_rtm
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import zipfile


def read(path):
    return json.loads(path.read_text())


def fmt(value):
    return 'N/A' if value is None else f'{value:.4f}'


def invariants(base, candidate):
    old, new = read(base/'baseline.json'), read(candidate/'baseline.json')
    a, b = {v['key']:v for v in old['videos']}, {v['key']:v for v in new['videos']}
    assert set(a) == set(b), '69-video cohort changed'
    assert len(a) == 69
    failures=[]
    evidence=[]
    for key in a:
        checks={'counts':a[key]['counts']==b[key]['counts'],
                'contact_frames':a[key]['image_frames']==b[key]['image_frames']}
        for field in ('video_sha256','plate_model_sha256','metrics_sha256'):
            checks[field]=a[key]['fingerprint'][field]==b[key]['fingerprint'][field]
        hashes={}
        for stage in ('plate','pose_plate'):
            x,y=base/'raw'/key/stage,candidate/'raw'/key/stage
            if not x.exists() and not y.exists():continue
            if not x.exists() or not y.exists():
                checks[stage]=False
                continue
            for name in ('rows.jsonl.gz','result.json'):
                if name.endswith('.gz'):
                    left,right=gzip.decompress((x/name).read_bytes()),gzip.decompress((y/name).read_bytes())
                else:
                    left,right=read(x/name),read(y/name)
                    # Runtime is not a measured plate outcome; all other fields,
                    # including every serialized rep/depth/speed, must match.
                    for obj in (left,right):
                        for field in ('recog_s','recog_fps'):obj.pop(field,None)
                    left=json.dumps(left,ensure_ascii=False).encode()
                    right=json.dumps(right,ensure_ascii=False).encode()
                label=stage+'/'+name
                checks[label]=left==right
                hashes[label]={'mp':hashlib.sha256(left).hexdigest(),
                               'rtm':hashlib.sha256(right).hexdigest()}
        if not all(checks.values()):failures.append(dict(key=key,checks=checks))
        evidence.append(dict(key=key,checks=checks,hashes=hashes))
    return dict(videos=len(a),count_videos=sum(v['counts'] is not None for v in a.values()),
                exact=not failures,failures=failures,evidence=evidence)


def comparison(reports):
    lines=['## M0 / M1 비교','',
           'M0와 같은 69개 영상, 61개 횟수 평가, 동일한 집계식과 관절·rep 구간 분모를 쓴다. '
           '홀짝 판정 칸 수와 측정 가능한 rep 수는 포즈 가용성에 따라 달라진다. '
           '분모가 줄어든 0건을 정확도 개선으로 해석하지 않는다.',
           '', '|범위|모드|MAE|exact|OBO|event F1|coverage|몸통 점프|홀짝 다른 칸/판정 칸|fps|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scope in ('all','squat','benchpress','deadlift'):
        for mode,report in reports.items():
            s=report['summary'][scope];c=s['counts'];f=s['false_difference']
            lines.append('|'+scope+'|'+mode+'|'+'|'.join(fmt(c[k]) for k in ('count_mae','count_exact','obo','event_f1'))+
                         f'|{fmt(s["coverage"])}|{s["jumps"]}|{f["different"]}/{f["judged"]}|{fmt(s["timing"]["fps"])}|')
    lines+=['','### 관절 묶음 visibility ≥0.5','',
            '|범위|모드|어깨|팔꿈치|손목|엉덩이|무릎|발목|발|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope in ('all','squat','benchpress','deadlift'):
        for mode,report in reports.items():
            s=report['summary'][scope]
            lines.append('|'+scope+'|'+mode+'|'+'|'.join(fmt(v['rate']) for v in s['joints'].values())+'|')
    lines+=['','### CPU 단계별 시간 (초)','',
            '한 영상씩 CPUExecutionProvider, ORT/OpenCV 1 thread. 포즈 시간은 모든 검출 사람의 top-down 추론·전후처리 합계다. '
            'MP33 어댑터·점수 분포 계산·사람 선택·EMA는 나머지 시간에 들어간다. MediaPipe의 사람 검출은 포즈 내부에 포함돼 분리할 수 없다. 전체 fps는 M0처럼 원판·포즈 두 패스를 포함하고 CFR 준비·지표·이미지 생성은 제외한다.',
            '', '|범위|모드|디코드|원판 검출|사람 검출|KLT|포즈|나머지|합계|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope in ('all','squat','benchpress','deadlift'):
        for mode,report in reports.items():
            t=report['summary'][scope]['timing']
            lines.append('|'+scope+'|'+mode+'|'+'|'.join(fmt(t.get(k)) for k in ('decode','plate_detection','person_detection','klt','pose','rest','total'))+'|')
    lines+=['','### Rep 간 흔들림 (도)','',
            '영상별 SD(ddof=1)와 모든 중복 없는 rep 쌍 절대 차이 중앙값을 구하고, 범위별로 영상 중앙값을 낸다. '
            '제외 rep도 잴 수 있는 칸이면 포함한다. 표본 2개 미만은 N/A다. n은 SD 표본이 2개 이상인 영상 수(M0/LW/BAL)다. 측정 가능한 영상 집합이 달라질 수 있으므로 SD 감소만으로 정확도 개선을 증명하지 않는다. 각 모드 baseline.json에 모든 영상 원값이 있다.',
            '', '|범위|포지션|지표|n M0/LW/BAL|M0 SD|LW SD|BAL SD|M0 쌍 차이|LW 쌍 차이|BAL 쌍 차이|',
            '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    modes=list(reports)
    for scope in ('all','squat','benchpress','deadlift'):
        for i,cell in enumerate(reports['mediapipe']['summary'][scope]['variation']):
            cells=[reports[m]['summary'][scope]['variation'][i] for m in modes]
            lines.append('|'+scope+'|'+cell['position']+'|'+cell['metric']+'|'+
                         '/'.join(str(c['videos']) for c in cells)+'|'+
                         '|'.join(fmt(c[k]) for k in ('sd','pair_median') for c in cells)+'|')
    return '\n'.join(lines)+'\n'


def score_report(report):
    lines=['|영상|점수 수|최소|p01|p05|p25|p50|p75|p95|p99|최대|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in report['videos']:
        d=v['score_distribution']
        lines.append('|'+v['key']+'|'+str(d['n'])+'|'+'|'.join(fmt(d['quantiles'].get(k)) for k in ('min','p01','p05','p25','p50','p75','p95','p99','max'))+'|')
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mp',type=Path,required=True);p.add_argument('--rtm',type=Path,required=True)
    p.add_argument('--report',type=Path,help='Reviewed narrative prepended to comparison tables')
    a=p.parse_args();reports={'mediapipe':read(a.mp/'baseline.json')}
    checks={}
    for mode in ('lightweight','balanced'):
        path=a.rtm/mode
        report=read(path/'baseline.json')
        reviews=read(path/'lifter_review.json')
        assert set(reviews)=={v['key'] for v in report['videos']}, 'Review every contact sheet first'
        assert all(r.startswith(('correct','wrong')) for r in reviews.values()), 'Unfinished image review'
        for video in report['videos']:
            video['lifter_review']=reviews[video['key']]
        # Refresh review text without rerunning inference or replacing the
        # measured invocation wall time with a short --resume rendering pass.
        from eval.baseline import render
        (path/'baseline.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        markdown=render(report)
        (path/'baseline.md').write_text(markdown)
        (path/'REPORT.md').write_text(markdown)
        with zipfile.ZipFile(path/'baseline.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for name in ('baseline.json','baseline.md'):archive.write(path/name,name)
            for video in report['videos']:archive.write(path/video['image'],video['image'])
        reports[mode]=report
        checks[mode]=invariants(a.mp,path)
    a.rtm.mkdir(parents=True,exist_ok=True)
    (a.rtm/'invariance.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    text=(a.report.read_text()+'\n\n' if a.report else '# M1 RTMPose 결과\n\n')+comparison(reports)
    for mode in ('lightweight','balanced'):
        text+='\n## '+mode+' 점수 분포\n\n'+score_report(reports[mode])
    (a.rtm/'REPORT.md').write_text(text)
    with zipfile.ZipFile(a.rtm/'m1-rtmpose.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ('REPORT.md','invariance.json'):z.write(a.rtm/name,name)
        sim_files=[a.rtm/name for name in ('sim_m0.json','sim_m1.json')]
        if all(path.exists() for path in sim_files):
            assert sim_files[0].read_bytes()==sim_files[1].read_bytes(), 'Synthetic core regression'
            for path in sim_files:z.write(path,path.name)
        for mode in ('lightweight','balanced'):
            path=a.rtm/mode
            for name in ('baseline.json','baseline.md','lifter_review.json'):
                z.write(path/name,mode+'/'+name)
            for image in sorted((path/'images').glob('*.jpg')):
                z.write(image,mode+'/images/'+image.name)
            for image in sorted((path/'details').glob('*.jpg')):
                z.write(image,mode+'/details/'+image.name)
    print(json.dumps({m:{'exact':c['exact'],'videos':c['videos']} for m,c in checks.items()}))
    if not all(c['exact'] for c in checks.values()):raise SystemExit('Plate invariants failed; inspect invariance.json')


if __name__=='__main__':main()
