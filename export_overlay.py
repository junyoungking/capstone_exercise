#!/usr/bin/env python3
"""
영상 플레이어용 데이터 내보내기.

사용자 / 레퍼런스 영상을 나란히 놓고 비교하는 player.html 을 만든다.
레퍼런스는 자기 시계가 아니라 사용자의 phase 진행도에 맞춰 재생된다.

  python export_overlay.py --labels data/labels/squat_01.json --out player_01.html
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

from offline_compare import (  # noqa: E402
    N_SAMPLES,
    SequenceCache,
    build_rep_data,
    collect_findings,
    compare_rep,
    llm_sentence,
    reducer_for,
    resample_indices,
    template_sentence,
)
from unified_feedback_v4 import (  # noqa: E402
    ADVISORY_METRICS,
    DELTA_THRESHOLDS,
    DISPLAY_METRICS,
    METRIC_LABELS_KO,
    POSE_CONNECTIONS,
    SIDE_IDS,
    SideLock,
    compute_metrics,
    normalize_exercise_name,
)

# 지표가 의존하는 관절. 빨갛게 칠할 대상.
METRIC_JOINTS: Dict[str, List] = {
    "knee_angle":         ["hip", "knee", "ankle"],
    "hip_angle":          ["shoulder", "hip", "knee"],
    "elbow_angle":        ["shoulder", "elbow", "wrist"],
    "elbow_angle_avg":    ["shoulder", "elbow", "wrist"],
    "lockout_angle_min":  ["shoulder", "elbow", "wrist"],
    "trunk_lean":         [11, 12, 23, 24],
    "neck_lean":          [0, 11, 12],
    "foot_flatness":      ["heel", "foot", "ankle"],
    "head_hip_line":      [0, 23, 24],
    "wrist_elbow_x_diff": ["elbow", "wrist"],
    "bench_line_diff":    [0, 11, 12, 23, 24],
}


def joints_for(metric: str, side: str) -> List[int]:
    ids = SIDE_IDS[side]
    return sorted({ids[j] if isinstance(j, str) else int(j)
                   for j in METRIC_JOINTS.get(metric, [])})


def bones_within(js: List[int]) -> List[List[int]]:
    s = set(js)
    return [[a, b] for a, b in POSE_CONNECTIONS if a in s and b in s]


def dominant_side(seq, span, exercise: str) -> str:
    lock, side = SideLock(exercise), "right"
    for i in range(int(span[0]), min(int(span[1]), len(seq))):
        if seq[i] is not None:
            side = lock.update(seq[i])
    return side


def peak_frames(seq, span, exercise: str, phase: str, side: str) -> Dict[str, int]:
    """지표별 대표값이 나온 프레임. reducer 가 min/max 면 그 극값의 프레임,
    mean 이면 평균에 가장 가까운 프레임."""
    idxs = resample_indices(int(span[0]), min(int(span[1]), len(seq)), N_SAMPLES)
    series: Dict[str, List[tuple]] = {}
    for i in idxs:
        if i < 0 or i >= len(seq) or seq[i] is None:
            continue
        for k, v in compute_metrics(seq[i], exercise, side).items():
            if math.isfinite(v):
                series.setdefault(k, []).append((i, float(v)))

    out: Dict[str, int] = {}
    for k, pairs in series.items():
        how = reducer_for(phase, k)
        if how == "min":
            out[k] = min(pairs, key=lambda t: t[1])[0]
        elif how == "max":
            out[k] = max(pairs, key=lambda t: t[1])[0]
        else:
            avg = sum(v for _, v in pairs) / len(pairs)
            out[k] = min(pairs, key=lambda t: abs(t[1] - avg))[0]
    return out


def dump_frames(seq, spans, decimals: int = 4) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for a, b in spans:
        for i in range(int(a), min(int(b), len(seq))):
            if str(i) in out or seq[i] is None:
                continue
            out[str(i)] = [[round(float(p[0]), decimals),
                            round(float(p[1]), decimals),
                            round(float(p[3]), 2)] for p in seq[i]]
    return out


def instant_metrics(seq, spans_side, exercise: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for (a, b), side in spans_side:
        for i in range(int(a), min(int(b), len(seq))):
            if seq[i] is None or str(i) in out:
                continue
            out[str(i)] = {k: round(float(v), 2)
                           for k, v in compute_metrics(seq[i], exercise, side).items()
                           if math.isfinite(v)}
    return out


def build_sprite(path: Path, frames: Sequence[int], width: int, quality: int) -> Optional[dict]:
    """레퍼런스 프레임을 한 장의 이미지로 합친다.
    비디오 시크 대신 배경 위치만 바꾸므로 끊김이 없다."""
    if not frames:
        return None
    want = sorted(set(int(f) for f in frames))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    grabbed: Dict[int, np.ndarray] = {}
    idx, target = 0, set(want)
    cell_w = cell_h = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in target:
            h, w = frame.shape[:2]
            cell_w = width
            cell_h = int(round(h * width / w / 2) * 2)
            grabbed[idx] = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        idx += 1
    cap.release()
    if not grabbed:
        return None

    keys = sorted(grabbed)
    n = len(keys)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), np.uint8)
    pos: Dict[str, list] = {}
    for i, k in enumerate(keys):
        c, r = i % cols, i // cols
        sheet[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = grabbed[k]
        pos[str(k)] = [c, r]

    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    print(f"  스프라이트 {cols}x{rows} · 셀 {cell_w}x{cell_h} · {len(b64) / 1.33e6:.1f} MB")
    return {"cols": cols, "rows": rows, "cell": [cell_w, cell_h],
            "pos": pos, "data": "data:image/jpeg;base64," + b64}


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {"fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "n": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    return info


def main():
    ap = argparse.ArgumentParser(description="영상 플레이어용 데이터 내보내기")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="player.html")
    ap.add_argument("--template", default="player_template.html")
    ap.add_argument("--data-out", default=None)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--expert-width", type=int, default=300,
                    help="레퍼런스 스프라이트 셀 가로 픽셀 (작을수록 파일이 가벼움)")
    ap.add_argument("--expert-quality", type=int, default=72, help="스프라이트 JPEG 품질")
    ap.add_argument("--expert-video", action="store_true",
                    help="스프라이트 대신 비디오로 (시크가 끊길 수 있음)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    exercise = normalize_exercise_name(cfg["exercise"])
    thresholds = DELTA_THRESHOLDS.get(exercise, {})
    display = DISPLAY_METRICS.get(exercise, [])
    advisory = ADVISORY_METRICS.get(exercise, set())
    cache = SequenceCache(BASE_DIR, exercise)

    print(f"[1/5] 랜드마크 추출 ({exercise})")
    user_reps = build_rep_data(cache, cfg["user"], exercise)
    expert_reps = build_rep_data(cache, cfg["expert"], exercise)

    user_video = cfg["user"].get("video", user_reps[0]["video"])
    expert_video = cfg["expert"].get("video", expert_reps[0]["video"])
    user_seq, expert_seq = cache.get(user_video), cache.get(expert_video)
    ui, ei = video_info(BASE_DIR / user_video), video_info(BASE_DIR / expert_video)

    print("[2/5] 비교 및 판정")
    reps_out: List[dict] = []
    user_spans_side, expert_spans_side = [], []

    for i, ur in enumerate(user_reps):
        one_to_one = len(expert_reps) == len(user_reps)
        er = expert_reps[i] if one_to_one else expert_reps[0]
        er_label = cfg["expert"]["reps"][i if one_to_one else 0]
        ur_label = cfg["user"]["reps"][i]

        res = compare_rep(ur, er, exercise, thresholds, display, advisory)
        findings = collect_findings(res)
        res["findings"] = findings

        if res["excluded"]:
            res["sentence"] = f"이 rep 은 판정에서 제외됐습니다 ({res['exclude_reason']})."
        elif args.llm:
            res["sentence"] = llm_sentence(exercise, ur["index"], findings, args.top_k)
        else:
            res["sentence"] = template_sentence(findings, args.top_k)

        phases_out = []
        for ph in res["phases"]:
            name = ph["phase"]
            uspan, espan = ur_label["phases"][name], er_label["phases"][name]
            uside = dominant_side(user_seq, uspan, exercise)
            eside = dominant_side(expert_seq, espan, exercise)
            user_spans_side.append((tuple(uspan), uside))
            expert_spans_side.append((tuple(espan), eside))

            peaks = peak_frames(user_seq, uspan, exercise, name, uside)
            epeaks = peak_frames(expert_seq, espan, exercise, name, eside)

            flagged, marks = [], []
            for row in ph["rows"]:
                if row.get("status") != "over":
                    continue
                flagged.extend(joints_for(row["metric"], uside))
                if row["metric"] in peaks:
                    marks.append({
                        "frame": peaks[row["metric"]],
                        "expert_frame": epeaks.get(row["metric"]),
                        "metric": row["metric"],
                        "label": row["label"],
                        "diff": row["diff"],
                        "unit": row["unit"],
                        "k": row["ratio_to_threshold"],
                    })
            flagged = sorted(set(flagged))
            marks.sort(key=lambda m: abs(m["k"] or 0), reverse=True)

            phases_out.append({
                "phase": name, "side": uside,
                "start": int(uspan[0]), "end": int(uspan[1]),
                "expert_start": int(espan[0]), "expert_end": int(espan[1]),
                "rows": ph["rows"], "peaks": peaks,
                "flagged_joints": flagged, "flagged_bones": bones_within(flagged),
                "marks": marks,
            })

        reps_out.append({
            "rep": res["rep"], "excluded": res["excluded"],
            "exclude_reason": res["exclude_reason"], "sentence": res["sentence"],
            "findings": findings, "phases": phases_out,
            "start": min(p["start"] for p in phases_out),
            "end": max(p["end"] for p in phases_out),
        })
        print(f"  rep {res['rep']}: {'제외' if res['excluded'] else f'{len(findings)}건'}")

    print("[3/5] 사용자 스켈레톤 추출")
    uframes = dump_frames(user_seq, [s for s, _ in user_spans_side])
    uinst = instant_metrics(user_seq, user_spans_side, exercise)
    print(f"  {len(uframes)} 프레임")

    print("[4/5] 레퍼런스 스켈레톤 추출")
    eframes = dump_frames(expert_seq, [s for s, _ in expert_spans_side])
    print(f"  {len(eframes)} 프레임")

    sprite = None
    if not args.expert_video:
        sprite = build_sprite(BASE_DIR / expert_video, [int(k) for k in eframes],
                              args.expert_width, args.expert_quality)
        if sprite is None:
            print("  [경고] 스프라이트 생성 실패 — 비디오 방식으로 대체합니다")

    payload = {
        "exercise": exercise,
        "user": {"video": user_video, "fps": round(ui["fps"], 3),
                 "size": [ui["w"], ui["h"]], "total": ui["n"], "frames": uframes},
        "expert": {"video": expert_video, "fps": round(ei["fps"], 3),
                   "size": [ei["w"], ei["h"]], "total": ei["n"], "frames": eframes,
                   "sprite": sprite},
        "connections": [list(c) for c in POSE_CONNECTIONS],
        "labels": METRIC_LABELS_KO,
        "thresholds": thresholds,
        "threshold_note": "임시값. 9월 촬영 데이터로 재산출 예정.",
        "reps": reps_out,
        "instant": uinst,
    }

    if args.data_out:
        Path(args.data_out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"      {args.data_out} 저장")

    print("[5/5] 플레이어 생성")
    tpl = BASE_DIR / args.template
    if not tpl.exists():
        print(f"  [경고] {tpl.name} 없음")
        return
    out = Path(args.out)
    out.write_text(tpl.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    print(f"      {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"      영상 경로는 HTML 위치 기준 상대경로입니다: {user_video}, {expert_video}")


if __name__ == "__main__":
    main()
