"""Assemble reviewed M2 diagnostics, exact plate invariants, report and ZIP."""
from pathlib import Path
from itertools import combinations
import argparse,json,shutil,zipfile
import numpy as np
from eval.baseline import write,sha,render
from eval.m1b_audit import invariance
from eval.m2_reviews import annotate
from core.geometry import plate_geometry,standing_frames,stature
from core.io.rows import read_rows


def read(p):return json.loads(Path(p).read_text())
def fmt(x):return 'N/A' if x is None else f'{x:.4f}'
def med(v):
    v=[x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None



def diameter_diagnostic(d):
    root=Path(d['source'])/'raw'/d['key']
    with np.load(root/'pose.npz',allow_pickle=False) as f:
        raw=f['raw_lms'].astype(float);w=f['w'].item();h=f['h'].item()
    raw[:,:,:2]*=[w,h]
    stage=root/('pose_plate' if (root/'pose_plate').exists() else 'plate')
    rows=read_rows(stage/'rows.jsonl.gz')
    boxes=np.asarray([r['plate']['box'] if r.get('plate',{}).get('used') else [np.nan]*4 for r in rows])
    stand=standing_frames(raw)
    g=plate_geometry(boxes,(w,h),450)
    corrected=stature(raw,g,stand)[0]
    uncorrected=stature(raw,dict(g,lifter_m_per_px=g['plate_m_per_px']),stand)[0]
    changed=stature(raw,plate_geometry(boxes,(w,h),400),stand)[0]
    expected=d['geometry']['stature_cm']
    assert (corrected is None and expected is None) or np.isclose(corrected*100,expected)
    return dict(stature_450_cm=None if corrected is None else corrected*100,
                stature_400_cm=None if changed is None else changed*100,
                plate_plane_stature_450_cm=None if uncorrected is None else uncorrected*100,
                kd_median=float(np.median(g['kd'])),ratio_clipped=g['ratio']>1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m0',type=Path,required=True);p.add_argument('--m1',type=Path,required=True)
    p.add_argument('--m1b',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--notes',type=Path,required=True)
    a=p.parse_args();bench=a.runs/'bench_n10';source=a.m1b/'n10';selected=a.runs/'selected2d'
    base=read(source/'baseline.json');new=read(bench/'baseline.json');bv={v['key']:v for v in new['videos']}
    assert len(bv)==16
    oldreviews=read(source/'lifter_review.json');breviews=read(bench/'lifter_review.json')
    assert set(breviews)==set(bv) and all(v.startswith(('correct','wrong')) for v in breviews.values())
    for v in new['videos']:v['lifter_review']=breviews[v['key']]
    write(bench/'baseline.json',new)
    (bench/'baseline.md').write_text(render(new).replace('M1b based on plate-pipeline `69102a0`.','M2 based on plate-pipeline `d5e19be`.'))
    selected.mkdir(exist_ok=True);videos=[];reviews={}
    for old in base['videos']:
        key=old['key'];root=bench if key in bv else source;v=dict(bv.get(key,old))
        v['lifter_review']=breviews[key] if key in bv else oldreviews[key]
        reviews[key]=v['lifter_review'];videos.append(v)
        for stage in ('plate','pose_plate'):
            src=root/'raw'/key/stage
            if src.exists():shutil.copytree(src,selected/'raw'/key/stage,dirs_exist_ok=True)
    merged=dict(base,videos=videos,reuse='53 nonbench M1b n10 + 16 fresh M2 bench guard-off; no new combined 2D speed estimate')
    write(selected/'baseline.json',merged);write(selected/'lifter_review.json',reviews)
    references={'M0':a.m0,'M1 lightweight':a.m1/'lightweight','M1 balanced':a.m1/'balanced',
                **{'M1b '+n:a.m1b/n for n in ('n3','n5','n10','tiny_m')}}
    checks={name:invariance(root,selected) for name,root in references.items()}
    assert all(v['exact'] and v['videos']==69 and v['count_videos']==61 for v in checks.values())
    write(a.runs/'invariance.json',checks)
    annotated=read(a.runs/'lifter_review_intervals.json')
    wrong=read(bench/'wrong_frames.json')
    annotated['M2 bench guard off']=annotate('M2 bench guard off',bench,wrong)
    # Keep historical nonbench wrong samples; replace bench samples with the fresh review.
    wrong69={'squat/val1235.mp4':[0],'squat/stu6_11_2.mp4':[95],**wrong}
    annotated['M2 all selected2d']=annotate('M2 all selected2d',selected,wrong69)
    write(a.runs/'lifter_review_intervals.json',annotated)
    pose_root=a.runs/'pose3d';review3=read(pose_root/'image_review.json')
    data=[read(pose_root/'raw'/v['key']/'result3d.json') for v in videos]
    assert {d['key'] for d in data}==set(review3) and len(data)==69
    for d in data:
        r=review3[d['key']]
        assert r['status'] in ('reviewed','anomaly') and r['image_sha256']==sha(pose_root/d['image']) and r['frame']==d['image_frame']
        d['review']=r
        write(pose_root/'raw'/d['key']/'result3d.json',d)
    diagnostics={d['key']:diameter_diagnostic(d) for d in data}
    write(a.runs/'geometry_diagnostics.json',diagnostics)
    write(pose_root/'summary.json',dict(videos=[{k:v for k,v in d.items() if k not in ('frames','j3','angles3d')} for d in data]))
    lines=['# M2 3D와 원판 기하 결과','',
           '기본값: **tracked N=10, bench_rep_guard=False**. 3D는 진단 출력이며 판정에는 미사용.',
           '원판 69개 기록·결과와 횟수 61개는 M0·M1·M1b 모든 전체 설정과 동일(실행 시간 필드 제외). 원판 rows는 gzip 해제 후 바이트 일치. 3D 확인 이미지 69장을 모두 검토했다.','',
           f'키 추정 불가 {sum(d["geometry"]["stature_cm"] is None for d in data)}개, 150~200cm 밖 {sum(d["geometry"]["stature_cm"] is not None and not 150<=d["geometry"]["stature_cm"]<=200 for d in data)}개다. 3D 이미지 이상 관찰은 {sum(r["status"]=="anomaly" for r in review3.values())}개이며, 특히 벤치 복원은 판정용 정확도로 검증되지 않았다.', '',
           '## 벤치 16개: guard 전후','', '|설정|coverage|리프터 오류 안|리프터 오류 밖|오류 합집합|','|---|---:|---:|---:|---:|']
    for name,report,ann in [('M1b n10 guard on',base,annotated['M1b n10']),('M2 n10 guard off',new,annotated['M2 bench guard off'])]:
        entries=[v for v in ann['videos'] if v['key'] in bv]
        inside=sum(bool(v['wrong_inside']) for v in entries);outside=sum(bool(v['wrong_outside']) for v in entries)
        union=sum(bool(v['wrong_inside'] or v['wrong_outside']) for v in entries)
        lines.append(f'|{name}|{fmt(report["summary"]["benchpress"]["coverage"])}|{inside}|{outside}|{union}|')
    lines+=['','|영상|coverage 전|coverage 후|안 오류 프레임|밖 오류 프레임|확인 이미지|','|---|---:|---:|---|---|---|']
    annby={v['key']:v for v in annotated['M2 bench guard off']['videos']}
    for v in new['videos']:
        old=next(x for x in base['videos'] if x['key']==v['key']);ann=annby[v['key']]
        lines.append(f'|{v["key"]}|{fmt(old["observation"]["coverage"])}|{fmt(v["observation"]["coverage"])}|{ann["wrong_inside"]}|{ann["wrong_outside"]}|bench_n10/{v["image"]}|')
    lines+=['','## 과거 lifter_review의 완료 rep 안/밖 분류','', '|설정|안 오류 영상|밖 오류 영상|합집합|','|---|---:|---:|---:|']
    for name,ann in annotated.items():lines.append(f'|{name}|{ann["wrong_inside_videos"]}|{ann["wrong_outside_videos"]}|{ann["wrong_union_videos"]}|')
    lines+=['','### 오류 표본과 이미지','', '|설정|영상|안 프레임|밖 프레임|이미지|','|---|---|---|---|---|']
    for name,ann in annotated.items():
        for v in ann['videos']:
            if v['wrong_inside'] or v['wrong_outside']:
                prefix='bench_n10' if name=='M2 bench guard off' or (name=='M2 all selected2d' and v['key'] in bv) else 'M1b_n10' if name=='M2 all selected2d' else name.replace(' ','_')
                lines.append(f'|{name}|{v["key"]}|{v["wrong_inside"]}|{v["wrong_outside"]}|{prefix}/{v["image"]}|')
    lines+=['','## 3D 요약','', '|종목|영상|뼈 CV 전|뼈 CV 후|가속도 전 m/s²|가속도 후 m/s²|3D 처리 초|','|---|---:|---:|---:|---:|---:|---:|']
    for scope in ('all','squat','benchpress','deadlift'):
        chosen=[d for d in data if scope=='all' or d['exercise']==scope]
        cvs={phase:med([b['cv'] for d in chosen for st in d['per_set_bone_cv'].values() for b in st[phase]]) for phase in ('before','after')}
        lines.append(f'|{scope}|{len(chosen)}|{fmt(cvs["before"])}|{fmt(cvs["after"])}|{fmt(med([d["acceleration_before"] for d in chosen]))}|{fmt(med([d["acceleration_after"] for d in chosen]))}|{fmt(sum(d["timing"]["processing_s"] for d in chosen))}|')
    lines+=['','CV는 세트별 모든 tree 뼈의 중앙값, 가속도는 영상별 중앙값들의 중앙값이다. 세트별 목표 길이가 달라도 고정 후 세트 내 CV는 수치 오차 수준이다. 상세 분모와 뼈별 값은 영상별 JSON에 있다.',
            '', '## 영상별 기하·분절','', '|영상|각도 °|거리 m|키 cm|넙다리 cm|정강이 cm|몸통 cm|위팔 cm|아래팔 cm|서 있는 프레임|키 보정|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for d in data:
        g=d['geometry'];s=g['segments_cm'];values=[g['angle_deg'],g['distance_m'],g['stature_cm'],*[s[k] for k in ('thigh','shin','torso','upper_arm','forearm')]]
        lines.append('|'+d['key']+'|'+'|'.join(fmt(v) for v in values)+f'|{g["standing_frames"]}|{g["scaled_to_stature"]}|')
    lines+=['','키 보정 False인 분절은 모델 자체 미터 스케일이며 원판 키로 교정되지 않았다. 키 null은 추정 불가다.',
            '', '### 키 이상치와 추정 불가','', '|영상|키 cm|바 간격 보정 전 cm|kd 중앙값|원인/가정 한계|','|---|---:|---:|---:|---|']
    for d in data:
        g=d['geometry'];h=g['stature_cm']
        if h is None or h<150 or h>200:
            reason='; '.join(d['warnings']) if h is None else ('발/신체 잘림과 원근·고정 초점/바 깊이 가정 편향; 실측 키 정답 없음' if d['key'].startswith('team9/') else '서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음')
            if d['key']=='benchpress/stu10_11.mp4':reason='완료 rep 없음; pre-lift 오선택과 편집/가림으로 standing 관측 오염 가능. '+reason
            if d['key']=='squat/val1235.mp4':reason='rep 안 오선택과 가려진 몸통 관측. '+reason
            diag=diagnostics[d['key']]
            lines.append(f'|{d["key"]}|{fmt(h)}|{fmt(diag["plate_plane_stature_450_cm"])}|{fmt(diag["kd_median"])}|{reason}|')
    lines+=['','### 같은 사람으로 본 클립 비교','', '영상 확인에서 같은 얼굴·체형·장비/체육관을 가진 `squat_01/02/03`과 `deadlift_01/02/03`을 각각 한 그룹으로 봤다. 원본 데이터에 신원 정답은 없으므로 시각적 그룹 가정이다. 존재하지 않는 04/05를 추가하지 않았다.',
            '', '|그룹|지표 cm|클립 수|최대−최소|모든 쌍 차이 중앙값|','|---|---|---:|---:|---:|']
    for exercise in ('squat','deadlift'):
        group=[d for d in data if d['key'] in [f'{exercise}/{exercise}_{i:02}.mp4' for i in (1,2,3)]]
        for metric in ('stature','thigh','shin','torso','upper_arm','forearm'):
            vals=[d['geometry']['stature_cm'] if metric=='stature' else d['geometry']['segments_cm'][metric] for d in group]
            vals=[v for v in vals if v is not None]
            lines.append(f'|{exercise}_01/02/03|{metric}|{len(vals)}|{fmt(max(vals)-min(vals) if len(vals)>1 else None)}|{fmt(med([abs(x-y) for x,y in combinations(vals,2)]))}|')
    # An actual clip diameter change on identical observed frames.
    key='squat/squat_01.mp4';v=next(v for v in videos if v['key']==key);root=source/'raw'/key
    with np.load(root/'pose.npz',allow_pickle=False) as f:raw=f['raw_lms'].copy();w=f['w'].item();h=f['h'].item()
    raw[:,:,:2]*=[w,h];stage=root/('pose_plate' if (root/'pose_plate').exists() else 'plate')
    rows=read_rows(stage/'rows.jsonl.gz');boxes=np.asarray([r['plate']['box'] if r.get('plate',{}).get('used') else [np.nan]*4 for r in rows]);stand=standing_frames(raw)
    heights=[diagnostics[key][field] for field in ('stature_450_cm','stature_400_cm')]
    lines+=['','### 450 → 400 mm 예시','',f'`{key}`의 같은 프레임에서 키는 {heights[0]:.4f} → {heights[1]:.4f} cm ({heights[1]-heights[0]:+.4f} cm, 비율 {heights[1]/heights[0]:.6f})다. 원판 면 거리/깊이/속도 비율은 정확히 400/450={400/450:.6f}이지만 키에는 지름과 무관한 0.70m 항이 있어 그 비율과 다르다. 분절의 추가 스케일 비율은 같은 키 비율이다.']
    lines+=['','## 2D·3D rep 간 흔들림','', '영상별 SD(ddof=1)와 모든 쌍 절대 차이 중앙값을 구한 뒤 종목별 영상 중앙값을 낸다. 2D는 M1b n10 원래 값이다. 벤치에는 guard 변경도 섞여 있으므로 차원을 바꾼 효과만으로 해석하지 않는다.',
            '', '|범위|포지션|지표|2D n|2D SD|2D 쌍 차이|3D n|3D SD|3D 쌍 차이|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for scope in ('all','squat','benchpress','deadlift'):
        chosen=[d for d in data if scope=='all' or d['exercise']==scope]
        for cell in base['summary'][scope]['variation']:
            c=[v for d in chosen for v in d['variation3d'] if v['position']==cell['position'] and v['metric']==cell['metric'] and v['sd'] is not None]
            lines.append(f'|{scope}|{cell["position"]}|{cell["metric"]}|{cell["videos"]}|{fmt(cell["sd"])}|{fmt(cell["pair_median"])}|{len(c)}|{fmt(med([v["sd"] for v in c]))}|{fmt(med([v["pair_median"] for v in c]))}|')
    lines+=['','## 3D 이미지 이상 목록','', '|영상|프레임|관찰|이미지|','|---|---:|---|---|']
    for key,r in review3.items():
        if r['status']=='anomaly':lines.append(f'|{key}|{r["frame"]}|{r["note"]}|pose3d/{r["image"]}|')
    lines+=['','전체 69개 이미지의 검토·입력 해시는 `pose3d/image_review.json`에 있다. 뒤집힘/좌우 바뀜을 확정할 3D 정답은 없으며, 명백한 투영 이상과 불확실성을 구분했다.',
            '', '## 속도','', '|범위|프레임|추론 초|기하·후처리 초|합계 초|','|---|---:|---:|---:|---:|']
    for scope in ('all','squat','benchpress','deadlift'):
        chosen=[d for d in data if scope=='all' or d['exercise']==scope]
        lines.append(f'|{scope}|{sum(d["total"] for d in chosen)}|'+ '|'.join(fmt(sum(d['timing'][k] for d in chosen)) for k in ('inference_s','geometry_postprocess_s','processing_s'))+'|')
    bm=read(a.runs/'threads6/benchmark.json');oldbm=read(a.m1b/'threads6/benchmark.json');n10=next(x for x in oldbm if x['name']=='n10')
    assert bm['frames']==1800 and bm['fps']==30 and bm['video_sha256']==n10['video_sha256']
    lines+=['','|설정, ORT 6|원판+2D 초|3D 추론 초|3D 기하/후처리 초|모델 준비·저장 포함 처리 초|지표 포함 전체 초|','|---|---:|---:|---:|---:|---:|---:|',
            f'|M1b n10 (당시 guard on)|{fmt(n10["processing_s"])}|N/A|N/A|{fmt(n10["processing_s"])}|{fmt(n10["end_to_end_s"])}|',
            f'|M2 n10 guard off + 3D|{fmt(bm["two_d_s"])}|{fmt(bm["three_d"]["inference_s"])}|{fmt(bm["three_d"]["geometry_postprocess_s"])}|{fmt(bm["processing_s"])}|{fmt(bm["end_to_end_s"])}|',
            '', '동일 입력 SHA256: `'+bm['video_sha256']+'`. M2의 torch도 6스레드다. 준비된 60초/30fps 영상을 단독 처리했다.',
            '', '## 구현·가정·재현','',a.notes.read_text().split('\n',1)[1]]
    text='\n'.join(lines)
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(text);(a.runs/'REPORT.md').write_text(text)
    with zipfile.ZipFile(a.runs/'m2-3d.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ('REPORT.md','invariance.json','lifter_review_intervals.json','geometry_diagnostics.json'):z.write(a.runs/name,name)
        for name in ('baseline.json','baseline.md','lifter_review.json','wrong_frames.json','shards.json'):z.write(bench/name,'bench_n10/'+name)
        for path in sorted((bench/'images').glob('*.jpg')):z.write(path,'bench_n10/images/'+path.name)
        for path in sorted(pose_root.glob('raw/**/result3d.json')):z.write(path,'pose3d/'+str(path.relative_to(pose_root)))
        for path in sorted((pose_root/'images').glob('*.jpg')):z.write(path,'pose3d/images/'+path.name)
        for name in ('summary.json','image_review.json'):z.write(pose_root/name,'pose3d/'+name)
        z.write(a.runs/'threads6/benchmark.json','threads6/benchmark.json')
        for name,root in {**references,'M1b bench_guard':a.m1b/'bench_guard'}.items():
            ann=annotated[name]
            z.writestr(name.replace(' ','_')+'/lifter_review_intervals.json',json.dumps(ann,ensure_ascii=False,indent=2))
            for v in ann['videos']:
                if v['wrong_inside'] or v['wrong_outside']:z.write(root/v['image'],name.replace(' ','_')+'/'+v['image'])
    print(json.dumps(dict(videos=69,invariance=True,reviewed=69,zip=str(a.runs/'m2-3d.zip'))))


if __name__=='__main__':main()
