"""Plate pipeline adapter: public video/rows/height APIs and result.json only."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import sys

import numpy as np

from feedback import config

# The plate project exposes core/server as top-level packages.
PLATE_ROOT = config.PROJECT_ROOT / 'plate_pipeline'
if str(PLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATE_ROOT))


def prepare_videos(settings: dict, out: Path) -> None:
    """Use one oriented CFR file for recognition, pose and browser playback."""
    from tools.check_video import ffprobe, fix, BROWSER_OK

    prepared = {}
    for role in ('user', 'ref'):
        item = settings[role]
        path = Path(item['video'])
        path = (config.PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        if path not in prepared:
            info = ffprobe(path)
            if '_error' in info:
                raise ValueError(info['_error'])
            needs_copy = (info.get('codec') not in BROWSER_OK
                          or info.get('pix_fmt') not in ('yuv420p', 'yuvj420p')
                          or bool(info.get('rotation'))
                          or info.get('variable_fps', False)
                          or abs(info.get('r_fps', 0) - info.get('avg_fps', 0)) > config.VIDEO_FPS_TOLERANCE)
            prepared[path] = fix(path, out / f'video_{role}') if needs_copy else path
        item['original_video'] = str(path)
        item['video'] = str(prepared[path])


def analyze_video(settings: dict, exercise: str, out: Path, *, model=None, device='auto'):
    from server.jobs import process_video
    from core.io.rows import read_rows

    plate_exercise = 'bench' if exercise in ('bench', 'benchpress') else exercise
    box = settings.get('plate_box')
    if isinstance(box, str):
        box = [float(v) for v in box.split(',')]
    process_video(settings['video'], plate_exercise, init_box=box,
                  model=str(model) if model else None, device=device, pose=False, out_dir=out)
    return read_rows(out / 'rows.jsonl.gz'), json.loads((out / 'result.json').read_text())


def check_frame_count(rows: list, pose: dict) -> None:
    if len(rows) != pose['total'] or [r['seq'] for r in rows] != list(range(pose['total'])):
        raise ValueError(f"원판 rows({len(rows)})와 포즈 프레임({pose['total']})이 다릅니다")


def nearest_seq(rows: list, time: float) -> int:
    return int(min(rows, key=lambda row: abs(row['t'] - time))['seq'])


def _crossing(height, start, end):
    midpoint = (height[start] + height[end]) / 2
    direction = 1 if height[end] >= height[start] else -1
    crossings = np.flatnonzero(direction * (height[start:end + 1] - midpoint) >= 0)
    if not len(crossings):
        raise ValueError('원판 중간 높이 교차 없음')
    return start + int(crossings[0])


def plate_rep_mask(rows, result):
    """Completed plate rep intervals on the prepared frame clock, no pose dependency."""
    mask = np.zeros(len(rows), dtype=bool)
    for group in result['sets']:
        for value in group['rep_values']:
            if any(value.get(k) is None for k in ('start', 'far_in', 'far_out', 'back')):
                continue
            start, back = nearest_seq(rows, value['start']), nearest_seq(rows, value['back'])
            if start <= back:
                mask[start:back+1] = True
    return mask


def build_plate_segments(pose, settings, exercise, rows, result, overrides=None):
    from core.signal.height import height_signal
    from export_overlay import dominant_side
    from feedback.segment import _rep_facing, apply_override
    from unified_feedback_v4 import normalize_exercise_name

    check_frame_count(rows, pose)
    exercise = normalize_exercise_name(exercise)
    t, y, _ = height_signal(rows, result.get('plate_mm', 450))
    if not len(t):
        raise ValueError('원판 높이 신호 없음')
    heights = np.interp([r['t'] for r in rows], t, y)
    sequence = [frame if valid else None for frame, valid in zip(pose['lms'], pose['valid'])]
    values = [v for st in result['sets'] for v in st['rep_values']]
    values.sort(key=lambda v: float('inf') if v['start'] is None else v['start'])
    reps, excluded, votes = [], [], []
    overrides = overrides or {}
    for number, value in enumerate(values, 1):
        if value['back'] is None:
            excluded.append({'rep': number, 'reason': 'rep 끝 시점 없음'})
            continue
        if any(value[k] is None for k in ('start', 'far_in', 'far_out')):
            excluded.append({'rep': number, 'reason': 'rep 원판 시점 없음'})
            continue
        start, far_in, far_out, back = [nearest_seq(rows, value[k]) for k in ('start', 'far_in', 'far_out', 'back')]
        if value['far_in'] > value['far_out']:
            lo, hi = sorted((far_in, far_out))
            far_in = far_out = lo + int(np.argmax(np.abs(heights[lo:hi + 1] - heights[start])))
        if not (0 <= start <= far_in <= far_out <= back < pose['total']):
            raise ValueError(f'rep {number}: 원판 시점 순서 오류')
        phases = {'down': [start, far_in], 'bottom': [far_in, far_out + 1], 'up': [far_out + 1, back + 1]}
        bottom = far_in + int(np.argmax(np.abs(heights[far_in:far_out + 1] - heights[start])))
        positions = dict(zip(config.POSITIONS, (start, _crossing(heights, start, bottom), bottom,
                                                _crossing(heights, bottom, back), back)))
        if str(number) in overrides:
            positions = apply_override(positions, overrides[str(number)], phases, pose['total'])
        side = dominant_side(sequence, (start, back + 1), exercise)
        facing = _rep_facing(sequence, positions['START'], side)
        if facing:
            votes.append(facing)
        reps.append({'rep': number, 'phases': phases, 'positions': positions, 'side': side})
    if set(overrides) - {str(rep['rep']) for rep in reps}:
        raise ValueError('원판 rep에 없는 override')
    return {'video': settings['video'], 'fps': pose['fps'], 'size': [pose['w'], pose['h']],
            'total': pose['total'], 'exercise': exercise, 'view': settings['view'],
            'facing': Counter(votes).most_common(1)[0][0] if votes else None,
            'source': 'plate+override' if overrides else 'plate', 'reps': reps,
            'plate_rep_count': len(values), 'excluded_reps': excluded}
