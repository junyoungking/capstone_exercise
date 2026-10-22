"""Build both midterm figures from current plate code and saved B_generic outputs.

From the repository root (no pose model inference):
  python plate_pipeline/docs/midterm/src/figs_ai.py --check

Inputs remain external and can be relocated with --local-root/--raw-root/--v40.
Only this script and the four PNG/SVG figures belong in git. REPORT.md, raw
recognition output, font cache, check logs and the PNG-only ZIP go to eval_out.
The --check option also requires node on PATH.
Use --reuse-state-rows only to redraw an already generated current-code result.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[4]
PIPE = ROOT / 'plate_pipeline'
sys.path.insert(0, str(PIPE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory
import numpy as np

from core.config import DEFAULT
from core.io import video
from core.io.rows import read_rows
from core.judge.session import REST, run_session
from core.signal.final import clean_final
from core.signal.height import height_signal

PLATE, POSE, NAVY, GREY = '#17A08F', '#E2862E', '#1F2A5C', '#9AA3AE'
LIGHT_PLATE, LIGHT_GREY = '#E8F6F3', '#F1F3F5'
EXERCISES = {'squat': '스쿼트', 'deadlift': '데드리프트', 'benchpress': '벤치프레스'}


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def korean_font(output, explicit=None):
    """Extract the Korean face, rather than matplotlib's first (Japanese) TTC face."""
    if explicit:
        font = explicit
    else:
        source = Path(subprocess.check_output(
            ['fc-match', '-f', '%{file}', 'Noto Sans CJK KR'], text=True).strip())
        if source.suffix == '.ttc':
            from fontTools.ttLib import TTCollection
            collection = TTCollection(source)
            face = next(f for f in collection.fonts
                        if f['name'].getDebugName(1) == 'Noto Sans CJK KR')
            font = output / 'NotoSansCJK-KR.otf'
            face.save(font)
        else:
            font = source
    assert fm.FontProperties(fname=font).get_name() == 'Noto Sans CJK KR'
    fm.fontManager.addfont(str(font))
    plt.rcParams.update({
        'font.family': 'Noto Sans CJK KR', 'font.size': 16,
        'text.color': NAVY, 'axes.labelcolor': NAVY, 'axes.edgecolor': NAVY,
        'xtick.color': NAVY, 'ytick.color': NAVY, 'axes.unicode_minus': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'svg.fonttype': 'path', 'svg.hashsalt': '중간발표그림',
    })
    return font


def save_figure(fig, name, destination):
    """16:9 logical 1600x900 canvas; 200 dpi PNG is exactly 3200x1800."""
    from PIL import Image
    png, svg = destination / f'{name}.png', destination / f'{name}.svg'
    fig.savefig(png, dpi=200, facecolor='white')  # no tight bbox / crop
    fig.savefig(svg, facecolor='white', metadata={'Date': None, 'Creator': '중간 발표 그림'})
    tree = ET.parse(svg)
    tree.getroot().set('width', '1600px')
    tree.getroot().set('height', '900px')
    # Preserve SVG's internal viewBox: only its displayed size changes.
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    tree.write(svg, encoding='utf-8', xml_declaration=True)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    with Image.open(png) as im:
        assert im.size == (3200, 1800)
        assert np.allclose(im.info['dpi'], (200, 200), atol=.01)
    assert not tree.findall('.//{http://www.w3.org/2000/svg}image')
    plt.close(fig)


def signal_and_trips(rows, result):
    t, y, x = height_signal(rows, result['plate_mm'])
    final = clean_final(t, y, x, result['fps'], DEFAULT.final)
    s = final.sig
    sets = run_session(s.t, s.y, s.x, s.gap_before, result['exercise'], cfg=DEFAULT)
    trips = [trip for st in sets for trip in st.reps]
    assert len(trips) == sum(st['reps'] for st in result['sets'])
    return final, trips


def sample_at(signal, t):
    """Look up a state-machine sample, never interpolate a marking time."""
    i = np.flatnonzero(signal.t == t)
    assert len(i) == 1, f'Trip time is not a final-signal sample: {t}'
    return float(signal.y[i[0]])


def state_source(args):
    dest = args.output / 'state_source'
    if args.reuse_state_rows:
        result = json.loads((dest / 'result.json').read_text())
        rows = read_rows(dest / 'rows.jsonl.gz')
    else:
        import cv2
        import onnxruntime as ort
        cv2.setNumThreads(1)
        # Same CPU runtime parallelism as midterm-eval; no inference thresholds changed.
        original = ort.InferenceSession
        def one_thread_session(*a, **kw):
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            return original(*a, sess_options=options, **kw)
        ort.InferenceSession = one_thread_session
        try:
            from server.jobs import process_video
            result, rows, _ = process_video(args.state_video, 'squat', model=args.model,
                                           out_dir=dest, device='cpu', pose=False)
        finally:
            ort.InferenceSession = original
    final, trips = signal_and_trips(rows, result)
    choices = [1] + [i for i in range(len(trips)) if i != 1]  # second rep first
    values = [v for st in result['sets'] for v in st['rep_values']]
    for i in choices:
        if i >= len(trips):
            continue
        tr, value = trips[i], values[i]
        has_endpoints = all(value[k] is not None for k in ('start', 'far_in', 'far_out', 'back'))
        segment = (final.sig.t >= tr.entry_t) & (final.sig.t <= tr.end_t)
        if has_endpoints and not value['blank'] and not tr.entry_gap and not tr.far_gap and not np.any(
                final.sig.gap_before[segment] > DEFAULT.final.interp_s):
            return final, tr, i + 1, result
    raise RuntimeError('No complete rep without gaps in stu10_70')


def state_inset(fig):
    c = DEFAULT.rep
    ax = fig.add_axes([.49, .735, .46, .125])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    for x, label, fill, edge in [(0, '대기', LIGHT_GREY, GREY),
                                 (.40, '진행', LIGHT_PLATE, PLATE),
                                 (.82, '1회', '#FFFFFF', PLATE)]:
        ax.add_patch(FancyBboxPatch((x, .34), .17, .38,
                                   boxstyle='round,pad=0.01,rounding_size=0.04',
                                   facecolor=fill, edgecolor=edge, lw=1.4, clip_on=False))
        ax.text(x+.085, .53, label, ha='center', va='center', fontsize=14, weight='bold')
    for left, right in [(.18, .39), (.58, .81)]:
        ax.annotate('', (right, .53), (left, .53), arrowprops={'arrowstyle': '->', 'color': NAVY, 'lw': 1.3})
    ax.text(.285, .88, f'{c.entry_m*100:g} 센티미터 벗어남', ha='center', fontsize=11)
    ax.text(.69, .88, f'{c.min_amp_m*100:g} 센티미터 이상 + {c.ret*100:g}% 돌아옴', ha='center', fontsize=11)
    ax.plot([.905, .905, .085], [.32, .10, .10], color=GREY, lw=1.2)
    ax.annotate('', (.085, .32), (.085, .10), arrowprops={'arrowstyle': '->', 'color': GREY, 'lw': 1.2})
    ax.text(.49, .06, '다시 대기', fontsize=11, va='top', ha='center', color=NAVY)


def draw_state(final, trip, rep_number, destination):
    c, s = DEFAULT.rep, final.sig
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.text(.065, .94, '원판의 이동과 복귀로 1회를 세는 상태 전환', fontsize=28, weight='bold')
    fig.text(.065, .885, f'스쿼트 {rep_number}번째 반복 · 최종 원판 신호', fontsize=16, color=NAVY)
    state_inset(fig)
    ax = fig.add_axes([.105, .145, .845, .535])
    lo, hi = trip.entry_t - .5, trip.end_t + .5
    mask = (s.t >= lo) & (s.t <= hi)
    distance = -(s.y - trip.ref_y) * 100  # coordinate conversion only; no new peak / crossing search
    amp = trip.amp * 100
    far_distance = -(trip.far_y - trip.ref_y) * 100
    assert np.isclose(far_distance, amp, atol=1e-12)
    assert sample_at(s, trip.far_t) == trip.far_y
    ax.axvspan(lo, trip.entry_t, color=LIGHT_GREY, zorder=0)
    ax.axvspan(trip.entry_t, trip.end_t, color=LIGHT_PLATE, zorder=0)
    ax.axvspan(trip.end_t, hi, color=LIGHT_GREY, zorder=0)
    ax.plot(s.t[mask], distance[mask], color=PLATE, lw=3.3, zorder=3)
    for y, color, dash in [(0, NAVY, '-'), (c.entry_m*100, GREY, '--'),
                           (c.min_amp_m*100, GREY, '--'), ((1-c.ret)*amp, NAVY, '--')]:
        ax.axhline(y, color=color, ls=dash, lw=1.25, alpha=.9, zorder=1)
    # Labels on the left and right avoid the signal and the close threshold lines.
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    box = dict(facecolor='white', edgecolor='none', alpha=.92, pad=2)
    ax.text(.012, -.8, '기준점 0 센티미터', transform=trans, va='bottom', fontsize=14, bbox=box)
    ax.text(.012, c.entry_m*100-.5, f'진입선 {c.entry_m*100:g} 센티미터', transform=trans, va='bottom', fontsize=14, bbox=box)
    ax.text(.012, c.min_amp_m*100-.5, f'최소 이동폭 {c.min_amp_m*100:g} 센티미터', transform=trans, va='bottom', fontsize=14, bbox=box)
    ax.text(.99, (1-c.ret)*amp+1, f'{c.ret*100:g}% 돌아옴', transform=trans, ha='right', va='top', fontsize=14, bbox=box)
    entry_y = -(sample_at(s, trip.entry_t)-trip.ref_y)*100
    entry_point = ax.scatter([trip.entry_t], [entry_y], color=PLATE, s=55, zorder=5)
    ax.annotate(f'진행 시작\n{trip.entry_t:.3f}초', (trip.entry_t, entry_y),
                xytext=(-12, -72), textcoords='offset points', fontsize=15,
                arrowprops={'arrowstyle': '->', 'color': NAVY, 'lw': 1.1}, ha='right', bbox=box)
    far_point = ax.scatter([trip.far_t], [far_distance], color=PLATE, edgecolor='white', s=115, lw=1.5, zorder=5)
    ax.annotate(f'가장 먼 점 · {trip.far_t:.3f}초\n이동폭 {amp:.2f} 센티미터',
                (trip.far_t, far_distance), xytext=(0, 30), textcoords='offset points',
                fontsize=16, ha='center', va='bottom',
                arrowprops={'arrowstyle': '->', 'color': NAVY, 'lw': 1.1})
    end_line = ax.axvline(trip.end_t, color=PLATE, lw=2, zorder=2)
    ax.text(trip.end_t+.035, amp*.68, f'1회\n{trip.end_t:.3f}초', fontsize=18,
            weight='bold', ha='left', va='center', color=PLATE, bbox=box)
    # The actual artists must use Trip fields exactly, not rounded display strings.
    assert float(entry_point.get_offsets()[0, 0]) == trip.entry_t
    assert float(far_point.get_offsets()[0, 0]) == trip.far_t
    assert float(far_point.get_offsets()[0, 1]) == far_distance
    assert list(end_line.get_xdata()) == [trip.end_t, trip.end_t]
    assert trip.amp >= c.min_amp_m and not trip.flags
    ax.set(xlim=(lo, hi), ylim=(amp+8, -6), xlabel='시간(초)',
           ylabel='기준점에서 먼 쪽으로 간 거리(센티미터)')
    ax.tick_params(labelsize=14)
    ax.spines[['top', 'right']].set_visible(False)
    for x, text in [((lo+trip.entry_t)/2, '대기'), ((trip.entry_t+trip.end_t)/2, '진행'),
                    ((trip.end_t+hi)/2, '대기')]:
        ax.text(x, 1.025, text, transform=ax.get_xaxis_transform(), ha='center', fontsize=14)
    fig.text(.105, .045, '내려가는 방향을 아래로 표시 · 선과 점의 시각은 상태기계가 기록한 값을 사용', fontsize=13, color=NAVY)
    save_figure(fig, 'state_machine', destination)
    return {'rep': rep_number, 'trip': asdict(trip), 'far_distance_cm': far_distance,
            'amp_cm': amp, 'entry_m': c.entry_m, 'min_amp_m': c.min_amp_m, 'ret': c.ret,
            'return_line_cm': (1-c.ret)*amp, 'plot_window': [lo, hi], 'artist_assertions': True}


def comparison_inputs(args):
    plate = read_csv(PIPE / 'results/midterm/per_video.csv')
    assert len(plate) == 61
    pose_path = args.v40 / 'arms/B_generic/all_test_per_video.csv'
    pose = {r['key']: r for r in read_csv(pose_path)}
    candidates = []
    for p in plate:
        b = pose[p['key']]
        assert int(b['gt_count']) == int(p['gt_count'])
        gt, bc, pc = int(p['gt_count']), int(b['pred_count']), int(p['pred_count'])
        if pc == gt and bc != gt:
            raw = args.raw_root / p['key'] / 'result.json'
            result = json.loads(raw.read_text())
            candidates.append({'key': p['key'], 'exercise': p['type'], 'truth': gt, 'pose': bc,
                               'plate': pc, 'duration': video.info(args.local_root/'new_data/new_dataset'/p['key']).duration})
    # Prefer squat/deadlift, one clear omission with another counted rep as context,
    # and <=60 s. No saved predictions are changed to make the example clearer.
    eligible = [r for r in candidates if r['exercise'] in ('squat', 'deadlift')
                and r['truth'] >= 2 and r['truth']-r['pose'] == 1 and r['duration'] <= 60]
    selected = min(eligible, key=lambda r: (r['duration'], r['key']))
    key = selected['key']
    timeline_path = args.v40 / f'arms/B_generic/timelines/all_test/{key}.csv'
    timeline = read_csv(timeline_path)
    res = json.loads((args.raw_root / key / 'result.json').read_text())
    rows = read_rows(args.raw_root / key / 'rows.jsonl.gz')
    gt_path = PIPE / 'results/midterm/gt_clock.csv'
    clock = next(r for r in read_csv(gt_path) if r['key'] == key)
    gt = json.loads(clock['gt_events'])
    assert gt == res['midterm_evaluation']['gt_events']
    origin = float(clock['clock_origin_s'])
    assert origin == res['midterm_evaluation']['clock_origin_s']
    pose_events = [float(r['time_s']) for r in timeline if r['count_event'].lower() in ('true', '1')]
    assert len(pose_events) == selected['pose']
    assert len(gt) == selected['truth']
    final, trips = signal_and_trips(rows, res)
    assert len(trips) == selected['plate']
    assert selected['key'] == 'deadlift/deadlift_30.mp4', 'Reinspect the saved example before drawing a different one'
    assert len(gt) == 2 and len(pose_events) == 1 and abs(pose_events[0]-gt[0]) <= .5
    # One omission: the second GT completion has no saved count_event within the
    # same +/-0.5 s tolerance used by event F1. The figure does not re-decode phases.
    missed = [t for t in gt if not any(abs(p-t) <= .5 for p in pose_events)]
    assert missed == [gt[1]]
    return candidates, selected, timeline, final.sig, trips, origin, gt, pose_events, missed, timeline_path


def draw_comparison(data, destination):
    _, selected, timeline, signal, trips, origin, gt, events, missed, _ = data
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.text(.065, .94, f"정답 {selected['truth']}회 · 포즈 구간 분류 {selected['pose']}회 · 원판 {selected['plate']}회", fontsize=28, weight='bold')
    fig.text(.065, .885, '같은 데드리프트 영상의 구간 분류와 원판 신호', fontsize=16)
    band = fig.add_axes([.14, .70, .80, .11])
    curve = fig.add_axes([.14, .32, .80, .30], sharex=band)
    truth = fig.add_axes([.14, .17, .80, .10], sharex=band)
    ts = np.array([float(r['time_s']) for r in timeline])
    duration = max(selected['duration'], ts[-1], signal.t[-1]-origin)
    # Generic labels ready/outbound/return translated for deadlift direction.
    labels = {0: '준비', 1: '상승', 2: '하강', -1: '관측 없음'}
    colors = {0: '#F9E8D7', 1: POSE, 2: '#F0B783', -1: LIGHT_GREY}
    indexed = list(enumerate(timeline))
    segments = []
    for phase, group in itertools.groupby(indexed, key=lambda r: int(r[1]['decoded_phase'])):
        members = list(group); a, b = members[0][0], members[-1][0]
        start, end = ts[a], ts[b+1] if b+1 < len(ts) else ts[b]
        segments.append({'phase': phase, 'name': labels[phase], 'start_s': start, 'end_s': end})
        band.add_patch(Rectangle((start, .12), end-start, .65, facecolor=colors[phase], edgecolor='white', lw=1))
        if end-start >= .35:
            band.text((start+end)/2, .445, labels[phase], ha='center', va='center', fontsize=15)
    band.scatter(events, [.94]*len(events), marker='v', color=POSE, s=105, clip_on=False, zorder=5)
    band.set(ylim=(0, 1.08), ylabel='포즈 구간')
    # Actual clean_final output, shifted by the same source-clock origin as GT.
    curve.plot(signal.t-origin, signal.y*100, color=PLATE, lw=3)
    end_times = [tr.end_t-origin for tr in trips]
    end_heights = [sample_at(signal, tr.end_t)*100 for tr in trips]
    marker = curve.scatter(end_times, end_heights, marker='v', color=PLATE, s=105, zorder=5)
    assert np.array_equal(marker.get_offsets()[:, 0], end_times)
    curve.set_ylabel('원판 바 높이(센티미터)')
    curve.grid(axis='y', color=GREY, lw=.6, alpha=.25)
    curve.margins(y=.18)
    truth.vlines(gt, .20, .80, color=NAVY, lw=3)
    for i,t in enumerate(gt, 1):
        truth.text(t, .04, f'{i}회', color=NAVY, ha='center', va='top', fontsize=14)
    truth.set(ylim=(-.2, 1), ylabel='정답 반복', xlabel='시간(초)')
    for ax in (band, curve, truth):
        ax.set_xlim(0, duration+.08)
        ax.tick_params(labelsize=14)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in (band, truth):
        ax.set_yticks([])
        ax.spines[['left', 'bottom']].set_visible(False)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_horizontalalignment('right')
        ax.yaxis.set_label_coords(-.025, .4)
    band.tick_params(axis='x', bottom=False, labelbottom=False)
    curve.tick_params(axis='x', labelbottom=False)
    # Enclose the second decoded pose cycle through its missing completion,
    # without extending the box into the separate GT axis or inventing an event. Saved second ascent begins at 4.625 seconds.
    second_start = next(s['start_s'] for s in segments if s['phase'] == 1 and s['start_s'] > events[0])
    trans = blended_transform_factory(band.transData, fig.transFigure)
    rect = Rectangle((second_start-.10, .685), duration-second_start+.15, .14,
                     transform=trans, facecolor='none', edgecolor=GREY, lw=1.6, zorder=10)
    fig.add_artist(rect)
    fig.text(.14+(.80*((second_start+duration)/2)/(duration+.08)), .837,
             '빠짐', fontsize=18, weight='bold', ha='center', color=NAVY)
    fig.text(.14, .065, '세모: 각 방식이 1회를 센 시각 · 정답 눈금: 평가에 사용한 완료 시각', fontsize=14)
    save_figure(fig, 'pose_vs_plate', destination)
    return {'segments': segments, 'pose_events': events, 'plate_end_times': end_times,
            'gt_events': gt, 'missing_gt_events': missed,
            'highlight': [second_start-.10, duration+.05], 'clock_origin_s': origin}


def write_report(args, font, state, result, data, comparison, checks):
    candidates, selected, *_, pose_path = data
    gt_path = PIPE / 'results/midterm/gt_clock.csv'
    key = selected['key']
    sources = [args.state_video, args.model, pose_path,
               args.v40/'arms/B_generic/all_test_per_video.csv', gt_path,
               args.raw_root/key/'rows.jsonl.gz', args.raw_root/key/'result.json']
    meta = {'base_branch': 'integ/midterm-eval', 'base_commit': '67e7173',
            'state': state, 'comparison': comparison, 'selected': selected,
            'candidates': candidates, 'inputs': {str(p): sha(p) for p in sources},
            'font': str(font), 'node_for_checks': shutil.which('node'), 'providers': result['detector_providers'],
            'checks': checks}
    (args.output/'figure_sources.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    tr = state['trip']
    report = f'''# 중간 발표 그림 보고서

작업 브랜치: `integ/midterm-figs`; 기준: `integ/midterm-eval` (`67e7173`).
그림 네 개(PNG 두 개·SVG 두 개)와 `docs/midterm/src/figs_ai.py`만 커밋한다.
보고서·ZIP·입력/중간 결과·글꼴 캐시는 eval_out에 두고 커밋하지 않는다.

## 그림 1

- 영상: `{args.state_video}` (team9, stu10_70).
- 현재 코드로 CPU에서 원판을 새로 추론. 포즈는 사용하지 않음. {'이 실행은 재그리기 옵션으로 직전의 현재 코드 결과를 읽음.' if args.reuse_state_rows else '이번 실행에서 새로 처리함.'}
- 쓴 반복: **{state['rep']}번째**. rep 값의 start/far_in/far_out/back과 blank, Trip의 entry_gap/far_gap 및 최종 신호 공백을 확인해 선택.
- entry: **{tr['entry_t']:.12f}초**, far: **{tr['far_t']:.12f}초**, end: **{tr['end_t']:.12f}초**.
- A: **{state['amp_cm']:.12f} cm**; Trip.far_y={tr['far_y']:.12f} m, ref_y={tr['ref_y']:.12f} m.
- RepCfg에서 읽은 entry={state['entry_m']*100:g} cm, min_amp={state['min_amp_m']*100:g} cm, ret={state['ret']:g}.
  복귀선 (1−ret)×A={state['return_line_cm']:.12f} cm.
- 최종 신호는 height_signal → clean_final 출력. 그 신호로 run_session 안의 RepMachine을 실행해 받은 Trip을 사용.
  entry/far/end를 다시 찾거나 스무딩·극값 탐색·시각 보간하지 않았다. y만 −(y−Trip.ref_y)×100으로 좌표 변환.
- 표시 구간: {state['plot_window'][0]:.12f}~{state['plot_window'][1]:.12f}초 (entry/end 앞뒤 각각 0.5초).
- matplotlib 표시 객체의 entry/far/end x좌표가 Trip 원본 float와 정확히 같은지 assert 통과.
  far의 원본 y와 최종 신호 표본도 정확히 같은지 확인. 스쿼트는 y축을 뒤집어 하강이 아래로 보인다.

## 그림 2 후보 전체 목록

동일 61개에서 원판 완료 횟수=정답, 저장 B_generic 횟수≠정답인 후보 전체 {len(candidates)}개.
원판 완료 횟수는 midterm-eval per_video.csv의 pred_count를 그대로 사용한다.

| 영상 | 종목 | 정답 | 포즈 | 원판 |
|---|---|---:|---:|---:|
'''
    for r in candidates:
        report += f"| {r['key']} | {EXERCISES[r['exercise']]} | {r['truth']} | {r['pose']} | {r['plate']} |\n"
    report += f'''
## 그림 2 선택·구간·출처

- 선택: **{key}**. 데드리프트, {selected['duration']:.3f}초(60초 이하), 정답 {selected['truth']} / 포즈 {selected['pose']} / 원판 {selected['plate']}회.
- 스쿼트·데드 후보를 먼저 고려했다. 2회 이상 중 1회 누락인 짧은 사례를 골라, 첫 회는 세고 둘째 회는 빠지는 대비가 한 영상에 보이게 했다.
  이 조건에서는 이 영상 하나다. 1회 영상 전체 누락이나 deadlift_03의 추가 1회 사례보다 누락 위치를 읽기 쉽다.
- 포즈 누락: 두 번째 반복의 GT 완료 **{comparison['missing_gt_events'][0]:.12f}초**.
  저장된 둘째 상승은 4.625~6.750초, 하강은 6.750~{comparison['segments'][-1]['end_s']:.6f}초의 표본 구간으로 보이며,
  마지막 저장 표본 7.750초까지 준비(0)로 돌아오지 않아 두 번째 count_event가 없다.
  평가의 ±0.5초 매칭 범위에도 대응 포즈 이벤트가 없다. 회색 상자와 “빠짐”으로 표시.
  원본 영상은 {selected['duration']:.6f}초이고 저장 포즈 격자의 마지막 시각은 7.750초다. 저장된 격자를 그대로 표시했으며 뒤로 연장하지 않았다.
  영상 종료 부근의 사례이므로 다른 긴 영상 전체의 구간 분류 성능으로 일반화하지 않는다.
- 포즈가 1회를 센 시각(저장 count_event): {comparison['pose_events']}초.
- 원판이 1회를 센 시각(Trip.end_t, 공통 시계): {comparison['plate_end_times']}초.
- 정답 완료 이벤트(event F1과 동일): {comparison['gt_events']}초.
- **포즈 예측 파일:** `{pose_path}`.
- 포즈 횟수 파일: `{args.v40/'arms/B_generic/all_test_per_video.csv'}`.
- 원판 횟수 후보 파일: `{PIPE/'results/midterm/per_video.csv'}`.
- 원판 원시 행·결과: `{args.raw_root/key}`.
- GT: `{gt_path}`의 gt_events를 읽고 저장 result.json.midterm_evaluation.gt_events와 완전히 같은지 assert.
- 구간 띠는 저장 decoded_phase를 연속 표본별로 묶었다. 예측이나 디코더를 다시 실행하지 않는다.
  원본 라벨 0=ready, 1=outbound, 2=return을 데드리프트 방향에 맞춰 준비·상승·하강으로 번역했다.
  구간 끝은 다음 저장 표본 시각이며 마지막 구간은 마지막 저장 표본에서 끝낸다. 새 표본이나 이벤트를 만들지 않았다.
- 원판 최종 신호는 저장 rows로 현재 clean_final을 실행하고 Trip.end_t를 그대로 사용.
  원판 시각은 원본 PTS에서 {comparison['clock_origin_s']:.12f}초를 뺀 동일 영상 공통 시계다. GT별 이동/정렬은 없다.
- midterm event F1은 원판 rep_values.back을 완료 이벤트로 쓰지만, 그림의 원판 세모는 요청대로 Trip.end_t다.
  end_t(복귀율 충족)와 back(완전 복귀 측정 시각)을 같은 값으로 바꾸지 않았다.

## 저장·검증

- 흰 바탕, 논리 크기 1600×900, PNG 3200×1800 / 200 dpi, SVG 표시 크기 1600×900.
- 글꼴: Noto Sans CJK KR (Korean face), SVG는 글자 윤곽을 포함해 외부 글꼴 없이 읽을 수 있다.
- 색: 원판 {PLATE}, 포즈 {POSE}, 정답·축 {NAVY}, 보조선·상자 {GREY}.
- PNG 크기·DPI, SVG 벡터 여부, Trip 표시 시각 assert 통과.
- --check 실행에는 PATH에 Node가 필요하다. 이번 실행의 Node: `{shutil.which('node')}`.
- pytest: {checks if checks else '별도 실행 전 (--check 옵션으로 두 시험을 실행하고 로그를 남길 수 있음)'}.
- 세부 입력 SHA-256, 구간/이벤트 원본 값: `{args.output/'figure_sources.json'}`.

재현:

```bash
python plate_pipeline/docs/midterm/src/figs_ai.py --local-root {args.local_root} \\
  --raw-root {args.raw_root} --v40 {args.v40} --check
```

그림 위치: `{args.figs}`. PNG 두 개만 묶은 첨부: `{args.output/'midterm_figs.zip'}`.
'''
    (args.output/'REPORT.md').write_text(report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    local = Path(os.environ.get('MIDTERM_LOCAL_ROOT', str(Path.home()/'capstone_exercise')))
    sibling = ROOT.parent/'midterm-eval/out/midterm'
    p.add_argument('--local-root', type=Path, default=local)
    p.add_argument('--raw-root', type=Path, default=sibling if sibling.is_dir() else ROOT/'out/midterm')
    p.add_argument('--v40', type=Path)
    p.add_argument('--state-video', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--font', type=Path)
    p.add_argument('--figs', type=Path, default=PIPE/'docs/midterm/figs')
    p.add_argument('--output', type=Path, default=PIPE/'eval_out/midterm_figs')
    p.add_argument('--reuse-state-rows', action='store_true')
    p.add_argument('--check', action='store_true', help='run both required pytest suites; write logs outside git')
    args = p.parse_args()
    args.v40 = args.v40 or args.local_root/'new_data/experiments_athlete_pipeline_v40/athlete_yolo_balanced_phase_sameconditions_seed42'
    args.model = args.model or args.local_root/'plate_pipeline/models/plate_tiny.onnx'
    args.state_video = args.state_video or next((args.local_root/'plate_pipeline/data/team/test9').rglob('stu10_70.mp4'))
    args.output.mkdir(parents=True, exist_ok=True); args.figs.mkdir(parents=True, exist_ok=True)
    font = korean_font(args.output, args.font)
    final, trip, number, result = state_source(args)
    state = draw_state(final, trip, number, args.figs)
    data = comparison_inputs(args)
    comparison = draw_comparison(data, args.figs)
    checks = {}
    if args.check:
        for label, cwd, path in [('원판 시험', PIPE, 'tests'), ('피드백 시험', ROOT, 'tests/fb')]:
            done = subprocess.run([sys.executable, '-m', 'pytest', '-q', path], cwd=cwd, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (args.output/f'{path.replace("/", "_")}_pytest.txt').write_text(done.stdout)
            print(done.stdout, flush=True)
            assert done.returncode == 0, label
            checks[label] = done.stdout.strip().splitlines()[-1]
    write_report(args, font, state, result, data, comparison, checks)
    with zipfile.ZipFile(args.output/'midterm_figs.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ('state_machine', 'pose_vs_plate'):
            z.write(args.figs/f'{name}.png', f'{name}.png')
    print('두 그림과 보고서:', args.figs, args.output, flush=True)


if __name__ == '__main__':
    main()
