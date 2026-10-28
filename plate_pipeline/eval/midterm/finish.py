"""Evidence report, completion-event timelines and share archive (no media/model)."""
import argparse,json,hashlib,zipfile,sys,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'plate_pipeline'),str(Path(__file__).parent)]
from run import module
from reproduce import aggregate
from core.io.rows import read_rows
from core.signal.height import height_signal


def finish(v40,raw):
    font_manager.fontManager.addfont(str(raw/'NotoSansCJK-KR.otf'))
    plt.rcParams.update({'font.family':'Noto Sans CJK KR','axes.unicode_minus':False})
    out=ROOT/'plate_pipeline/results/midterm';d=pd.read_csv(out/'per_video.csv')
    from cohort import included
    assert set(d.key)==set(included(pd.read_csv(out/'test_membership.csv')).key)
    assert pd.read_csv(out/'v40_clock_baseline_summary.csv').same.all()
    failures=[]
    for _,r in d.iterrows():
        dest=raw/r.key;res=json.loads((dest/'result.json').read_text());e=res['midterm_evaluation'];gt=e['gt_events'];pred=e['pred_events']
        rows=read_rows(dest/'rows.jsonl.gz');t,y,_=height_signal(rows)
        fig,axes=plt.subplots(3,1,figsize=(16,9),sharex=True)
        origin=e.get('clock_origin_s',0)
        times=[row['t']-origin for row in rows]
        t=t-origin
        center=[row.get('plate',{}).get('center',[np.nan,np.nan])[1] if row.get('plate',{}).get('used') else np.nan for row in rows]
        axes[0].plot(times,center,color='#17A08F',lw=1);axes[0].invert_yaxis();axes[0].set_ylabel('tracked center y (px)')
        axes[1].plot(t,y*100,color='#17A08F',lw=1);axes[1].set_ylabel('height (cm)')
        stop=times[-1] if times else 1
        for vals,label,color in [(gt,'GT completion','#1F2A5C'),(pred,'Plate back','#17A08F')]:
            axes[2].step([0,*vals,stop],[0,*range(1,len(vals)+1),len(vals)],where='post',color=color,label=label)
            for x in vals:axes[1].axvline(x,color=color,alpha=.35,lw=.8)
        axes[2].set_ylabel('completed reps');axes[2].set_xlabel('time (s)');axes[2].legend(loc='upper left')
        for ax in axes:ax.grid(alpha=.15)
        fig.suptitle(f'{r.key} | GT={r.gt_count}, plate completed={r.pred_count}, internal trips={r.counted_trips}\ntracked={r.tracked_fraction:.1%}, view={r.view}, tp/fp/fn={r.tp}/{r.fp}/{r.fn}')
        fig.tight_layout(rect=(0,0,1,.93));fig.savefig(dest/'timeline.png',dpi=100,facecolor='white');plt.close(fig)
        if r.abs_error:
            span=float(np.ptp(y))*100 if len(y) else 0.
            if r.incomplete_counted_trips:
                cause=f'내부 trip {r.counted_trips}개 중 종료 back 공란 {r.incomplete_counted_trips}개를 완료 이벤트에서 제외'
            elif r.tracked_fraction<.6:
                cause='추적 프레임 부족으로 유효 왕복 rep을 생성하지 못함'
            elif span<13:
                cause=f'관측 원판 높이 변화가 {span:.1f} cm로 최소 rep 진폭 13 cm보다 작아 rep 미생성'
            elif r.pred_count>r.gt_count:
                cause='정답보다 많은 왕복을 rep으로 포함(과계수); 추가 왕복의 실제 동작/검출 identity는 미확인'
            else:cause=r.cause
            reason=f'{cause}; 추적 {r.tracked_fraction:.1%}, 높이 범위 {span:.1f} cm, 완료 {r.pred_count}/{r.gt_count}. 자동 진단 단서이며 물체 정체/분할 오류의 수작업 확정은 안 함'
            failures.append(dict(key=r.key,reason=reason,timeline=f'videos/{r.key}/timeline.png'))
    pd.DataFrame(failures).to_csv(out/'errors.csv',index=False)
    plate=pd.read_csv(out/'plate_summary.csv');views=pd.read_csv(out/'by_view.csv')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    lines=['# 중간 발표 원판 횟수 평가 보고서','','브랜치: `'+branch+'`; 기준: `ae31956`. 기존 S4 작업 트리와 루트의 준비 파일은 보존했다. PLAN/core/feedback/server/model은 수정하지 않았다.','',
        '## 0. 재현 확인','','reproduction.md와 reproduction_checks.csv: 원래 v40 전체 분할 해시 c91ed670… 및 B 세 묶음 횟수·이벤트 집계 일치. test 73개 키와 로컬 라벨의 완료 삼중 프레임도 records와 일치. train.summarize의 원본 AST 집계식을 직접 불러 쓰며, 매칭은 원본 decode.match_events다.',
        'v40_clock_baseline_check.csv: 로컬 영상의 원본 PTS에서 영상 첫 PTS를 뺀 v40 기준 시계로 GT를 복원해 B의 저장 count_event와 다시 매칭했다. 73개 모두 TP/FP/FN이 원본과 일치. delay 재집계와 원본은 v40_clock_baseline_summary.csv에서 대조하며, count/event 및 delay 중앙값·p95를 1e-9 오차 안에서 대조해 통과해야 첨부를 만든다. 처음에는 PyAV 절대 PTS를 그대로 써서 B delay 중앙값이 −0.015132초(원본 −0.004671초)로 달랐고, 첫 PTS를 뺀 공통 변환으로 원본을 재현했다.','',
        '## 1. 이벤트·지표 정의','','v40 data.make_pack은 L1/L2/L3 완료 삼중 라벨의 L3 원본 프레임 timestamp를 gt_events로 쓴다. 같은 로컬 영상의 PyAV 디코드 프레임 PTS(반올림 전)로 대응하고, GT와 예측 모두 첫 PTS를 뺐다. 이는 원본 v40 tracking_compare의 0시작 OpenCV 시계와 맞추는 영상 공통 변환이며 GT별 정렬이 아니다. 원판 원본 rows/result의 back은 원래 PTS 그대로 보존했다. 원판 rep_values.back은 쉬는 자리로 복귀한 완료 시각으로 선택했다. 내부 횟수는 증가했으나 back이 공란인 trip은 완료 이벤트에서 제외하고 별도 열로 남겼다. 숫자를 맞추기 위해 정답에 이벤트를 정렬하거나 옮기지 않았다.',
        'count_mae/exact/obo는 영상별 오차의 평균; OBO는 |오차|≤1. event F1은 TP/FP/FN을 합친 micro 집계이며 tolerance=0.5초, 최대 매칭 수·최소 절대 시각 오차의 순서 매칭이다. delay는 pred−GT의 중앙값과 |delay|의 p95다. 오프라인 최종 back 시각의 오차이며 온라인 처리·통지 지연을 뜻하지 않는다.','',
        '## 2. 실행 조건','','v40 원본 test 73개는 고정하고 명시적 원판 없음 12개만 제외한 61개를 평가한다. reference_test 52, custom_test 9, all_test 61이다. 다운로드·새 분할 없음. plate_tiny.onnx, device=auto, 실제 provider=CPUExecutionProvider. 라벨 종목을 그대로 사용(benchpress→bench). 포즈는 끄고 core 임계·선택·추적·세트 규칙은 기준 코드 그대로 사용했다. 평가 도구에서 ORT 실행 스레드만 세션당 1개, OpenCV 1개로 제한하고 영상 4개를 병렬 처리했다.','',
        '각도는 used인 det/det_low/rim 측정 상자의 가로/세로 비 중앙값이다. KLT의 추정 크기는 넣지 않았다. ≥0.75 옆, ≥0.4 비스듬, 그 아래 정면, 측정 없음은 없음. 물체 오검출이면 이 태그도 틀릴 수 있어 촬영 방향 정답으로 해석하지 않는다. 추적률은 used/전체 디코드 프레임이다.',
        f'개발용 team9와 파일명 일치: {int(d.development.sum())}/61개. 제외 집계는 plate_summary.csv의 without_development이며, 0개라면 원래 값과 같다.','',
        '## 3. 결과','','주 비교는 동일 61개 A/B/C/원판의 조건부 성능이다. 원판이 없는 영상까지 포함한 전체 시스템 성능이라고 주장하지 않는다. summary.md와 comparison.csv가 주 비교이며, 기존 73개 A/B/C는 baseline73_abc.csv 및 baseline73_abc_by_exercise.csv에 그대로 보존했다. event F1은 영상별 평균이 아니라 합계 TP/FP/FN으로 계산했다.',
        '종목별 동일 집합 비교는 comparison_by_exercise.csv, 개발 영상 stu8_7을 추가로 제외한 60개 비교는 comparison_without_development.csv다. 종목 인식 A 0.863, B·C 0.973은 원본 73개 v40 수치이며 61개 집계나 원판 종목 인식 성능이 아니다.',
        '제외 규칙이 추가되기 전 이미 실행한 12개 원판 산출물은 로컬 중간 파일로만 보존했다. 최종 원판 지표·실패 목록·공유 ZIP에서는 제외하며 0회 예측으로 넣지 않았다. 아래 제외 목록 외에 검출이 안 된 영상을 추가로 빼지 않았다.','',
        '### 원판 없음 명시적 제외 12개','','| 영상 | 제외 사유 |','|---|---|']
    for row in pd.read_csv(out/'excluded_no_plate.csv').to_dict('records'):lines.append(f"| {row['key']} | {row['reason']} |")
    lines+=['','### 동일 61개 중 추가 검출 실패','','']
    missing=d[d.view=='없음']
    if missing.empty:lines.append('원판 측정 프레임이 완전히 0인 추가 영상은 없다. 몇 프레임이라도 검출된 것이 올바른 원판 identity/지속 추적/횟수를 보장하지는 않는다.')
    else:
        for row in missing.to_dict('records'):lines.append(f"- {row['key']}: 원판 측정 없음, 예측 {row['pred_count']}, 정답 {row['gt_count']}. 61개 분모와 실패 집계에 유지.")
    lines+=['','추적률 60% 미만인 영상(진단 표시만 하며 제외하지 않음):']
    for row in d[d.tracked_fraction<.6].to_dict('records'):
        lines.append(f"- {row['key']}: 추적 {row['tracked_fraction']:.1%}, 완료 {row['pred_count']}/{row['gt_count']}, 태그 {row['view']}.")
    lines+=['','### 원판 조건부 결과','']
    lines+=['','### 원본 73개 A/B/C (보존, 주 비교 아님)','','| 방법 | n | MAE | exact | OBO | event F1 |','|---|---:|---:|---:|---:|---:|']
    for row in pd.read_csv(out/'baseline73_abc.csv').query("subset == 'all_test'").to_dict('records'):
        lines.append(f"| {row['arm']} | {row['videos']} | {row['count_mae']:.3f} | {row['count_exact']:.3f} | {row['obo']:.3f} | {row['event_f1']:.3f} |")
    lines+=['','### 동일 61개 주 비교','','| 묶음 | 방법 | n | MAE | exact | OBO | event F1 | TP/FP/FN | delay median/p95(s) |','|---|---|---:|---:|---:|---:|---:|---|---|']
    for row in pd.read_csv(out/'comparison.csv').to_dict('records'):
        lines.append(f"| {row['subset']} | {row['arm']} | {row['videos']} | {row['count_mae']:.3f} | {row['count_exact']:.3f} | {row['obo']:.3f} | {row['event_f1']:.3f} | {row['event_tp']}/{row['event_fp']}/{row['event_fn']} | {row['matched_delay_median_s']:.4f}/{row['matched_abs_delay_p95_s']:.4f} |")
    lines+=['','### 원판 세부 집계','']
    lines+=['| 묶음 | scope | n | MAE | exact | OBO | event F1 | TP/FP/FN | delay median/p95(s) |','|---|---|---:|---:|---:|---:|---:|---|---|']
    for _,s in plate.iterrows():lines.append(f'| {s.subset} | {s.scope} | {int(s.videos)} | {s.count_mae:.3f} | {s.count_exact:.3f} | {s.obo:.3f} | {s.event_f1:.3f} | {int(s.event_tp)}/{int(s.event_fp)}/{int(s.event_fn)} | {s.matched_delay_median_s:.4f}/{s.matched_abs_delay_p95_s:.4f} |')
    lines+=['','## 4. 횟수가 틀린 영상별 단서','','아래 원인은 자동 관측으로 분류한 진단 단서이며 확정 원인이 아니다. 모든 영상의 timeline은 ZIP videos/ 아래에 있다.','', '| 영상 | 원인·단서 |','|---|---|']
    lines += [f"| {f['key']} | {f['reason']} |" for f in failures]
    lines+=['','## 5. 검증·제한','','루트 tests/fb: 235 passed; plate_pipeline/tests: 123 passed; eval/midterm/test_metrics.py: 2 passed. 입력 GT 시계·원본 집계·유한 표본을 확인한 시험이며 독립 촬영 데이터의 최종 성능 보장은 아니다.',
        'Noto Sans KR 요청은 설치된 Noto Sans CJK KR 한국어 서브폰트를 사용했다. 세 발표 그림은 밝은 바탕 1600×900, 요청한 색상이다. 기존 v40 모델의 학습 데이터/원판 검출기 학습 데이터와의 추가 중복 감사는 하지 않았다.','',
        '## 재현 명령','','```bash',
        'export V40=/home/kojy1009/capstone_exercise/new_data/experiments_athlete_pipeline_v40/athlete_yolo_balanced_phase_sameconditions_seed42',
        'python plate_pipeline/eval/midterm/run.py --local-root /home/kojy1009/capstone_exercise --model /home/kojy1009/capstone_exercise/plate_pipeline/models/plate_tiny.onnx --workers 4',
        'python plate_pipeline/eval/midterm/present.py --v40 "$V40" --raw out/midterm',
        'python plate_pipeline/eval/midterm/finish.py --v40 "$V40" --raw out/midterm','```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    provenance=[]
    for p in [v40/'effective_split.csv',v40/'records.json',v40/'train.py',v40/'data.py',v40/'decode.py',Path(__file__).parent/'excluded_no_plate.csv',v40.parents[1]/'new_labels.csv',Path('/home/kojy1009/capstone_exercise/plate_pipeline/models/plate_tiny.onnx')]:
        provenance.append({'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    pd.DataFrame(provenance).to_csv(out/'input_hashes.csv',index=False)
    archive=ROOT/'out/midterm_share.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                assert p.suffix.lower() in ('.csv','.md','.png');z.write(p,Path('results/midterm')/p.relative_to(out))
        for key in d.key:
            for name in ('rows.jsonl.gz','result.json','timeline.png'):z.write(raw/key/name,Path('videos')/key/name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('archive',archive,archive.stat().st_size,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--v40',type=Path,required=True);p.add_argument('--raw',type=Path,required=True)
    a=p.parse_args();finish(a.v40,a.raw)
