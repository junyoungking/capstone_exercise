#!/usr/bin/env python3
"""
오프라인 rep/phase 비교 파이프라인.

rep 과 phase 는 사람이 라벨링한 것을 그대로 받는다 (분할 모델은 이번 범위 밖).
지표 계산은 unified_feedback_v4.compute_metrics 를 그대로 재활용한다.

  python offline_compare.py --labels labels.json --out result.json
  python offline_compare.py --labels labels.json --out result.json --llm

결과는 result.json 과 demo.html 로 나온다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# ── 기존 코드 재활용 ────────────────────────────────────────────
from unified_feedback_v4 import (  # noqa: E402
    DELTA_THRESHOLDS,
    DISPLAY_METRICS,
    METRIC_LABELS_KO,
    ADVISORY_METRICS,
    LandmarkSmoother,
    SideLock,
    compute_metrics,
    create_landmarker,
    extract_landmarks,
    normalize_exercise_name,
    valid_lms,
    MODEL_PATH,
)


# ============================================================
# 설정
# ============================================================

# phase 구간을 몇 개 지점으로 리샘플할지. 길이가 달라도 이 개수로 맞춘다.
N_SAMPLES = 24

# 지표별 대표값 추출 방식. rep 이 아니라 phase 단위로 적용된다.
REDUCERS: Dict[str, str] = {
    "knee_angle": "min",
    "hip_angle": "min",
    "elbow_angle": "min",
    "lockout_angle_min": "min",
    "trunk_lean": "max",
    "neck_lean": "max",
    "foot_flatness": "max",
    "wrist_elbow_x_diff": "max",
    "head_hip_line": "mean",
    "bench_line_diff": "mean",
}

# 특정 phase 에서는 반대쪽 극값이 의미 있다. (예: up 구간의 락아웃)
PHASE_REDUCER_OVERRIDE: Dict[tuple, str] = {
    ("up", "knee_angle"): "max",
    ("up", "hip_angle"): "max",
    ("up", "elbow_angle"): "max",
    ("up", "lockout_angle_min"): "max",
}

# 차이 부호를 사람 말로 옮길 때 쓴다. (양수일 때, 음수일 때)
DIRECTION_KO: Dict[str, tuple] = {
    "knee_angle": ("덜 굽혔습니다", "더 굽혔습니다"),
    "hip_angle": ("덜 접었습니다", "더 접었습니다"),
    "elbow_angle": ("덜 굽혔습니다", "더 굽혔습니다"),
    "lockout_angle_min": ("더 폈습니다", "덜 폈습니다"),
    "trunk_lean": ("더 숙였습니다", "덜 숙였습니다"),
    "neck_lean": ("더 젖혔습니다", "덜 젖혔습니다"),
    "foot_flatness": ("뒤꿈치가 더 떴습니다", "뒤꿈치가 더 붙었습니다"),
    "head_hip_line": ("머리가 더 벗어났습니다", "머리가 더 정렬됐습니다"),
    "wrist_elbow_x_diff": ("손목이 더 어긋났습니다", "손목이 더 정렬됐습니다"),
    "bench_line_diff": ("라인이 더 벗어났습니다", "라인이 더 맞았습니다"),
}

# 각도 지표는 도(°), 비율 지표는 몸통길이 정규화값이라 단위가 다르다.
RATIO_METRICS = {
    "foot_flatness",
    "head_hip_line",
    "wrist_elbow_x_diff",
    "bench_line_diff",
}

# phase 안에서 이 비율 미만으로만 값이 잡히면 그 지표는 측정 불가로 둔다.
MIN_VALID_RATIO = 0.60

# rep 안에서 측정 가능한 지표가 이 비율 미만이면 rep 자체를 제외한다.
MIN_REP_METRIC_RATIO = 0.50


# ============================================================
# 랜드마크 추출
# ============================================================


def extract_sequence(video_path: Path, exercise: str) -> List[Optional[np.ndarray]]:
    """영상 전체를 프레임별 랜드마크 배열로 바꾼다. 없으면 None."""
    if not video_path.exists():
        raise FileNotFoundError(f"영상을 찾을 수 없습니다: {video_path}")

    landmarker = create_landmarker(MODEL_PATH)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    smoother = LandmarkSmoother()
    seq: List[Optional[np.ndarray]] = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(idx / max(fps, 1e-6) * 1000)
        raw = extract_landmarks(landmarker, frame, ts)
        seq.append(smoother.update(raw) if valid_lms(raw) else None)
        idx += 1

    cap.release()
    print(f"  {video_path.name}: {len(seq)} 프레임, 검출 {sum(x is not None for x in seq)}")
    return seq


class SequenceCache:
    """영상 파일당 한 번만 랜드마크를 뽑는다. rep 이 여러 영상에 흩어져 있어도 재사용."""

    def __init__(self, base_dir: Path, exercise: str):
        self.base_dir = base_dir
        self.exercise = exercise
        self._cache: Dict[str, List[Optional[np.ndarray]]] = {}

    def get(self, video: str) -> List[Optional[np.ndarray]]:
        if video not in self._cache:
            self._cache[video] = extract_sequence(self.base_dir / video, self.exercise)
        return self._cache[video]


def resample_indices(start: int, end: int, n: int) -> List[int]:
    """[start, end) 구간을 n 개 지점으로 균등 리샘플한 프레임 인덱스."""
    if end <= start:
        return [start]
    if n == 1:
        return [start]
    step = (end - 1 - start) / (n - 1)
    return [int(round(start + step * i)) for i in range(n)]


# ============================================================
# phase 단위 지표 집계
# ============================================================


def reducer_for(phase: str, metric: str) -> str:
    return PHASE_REDUCER_OVERRIDE.get((phase, metric), REDUCERS.get(metric, "mean"))


def reduce_values(values: List[float], how: str) -> float:
    if how == "min":
        return float(min(values))
    if how == "max":
        return float(max(values))
    return float(statistics.fmean(values))


def phase_metrics(
    seq: List[Optional[np.ndarray]],
    span: Sequence[int],
    exercise: str,
    phase: str,
    side_lock: SideLock,
) -> Dict[str, dict]:
    """한 phase 구간의 지표별 대표값과 유효 비율."""
    start, end = int(span[0]), int(span[1])
    idxs = resample_indices(start, min(end, len(seq)), N_SAMPLES)

    collected: Dict[str, List[float]] = {}
    n_points = 0

    for i in idxs:
        if i < 0 or i >= len(seq):
            continue
        lms = seq[i]
        if lms is None:
            continue
        n_points += 1
        side = side_lock.update(lms)
        m = compute_metrics(lms, exercise, side)
        for k, v in m.items():
            if not math.isfinite(v):
                continue
            collected.setdefault(k, []).append(float(v))

    total = max(len(idxs), 1)
    out: Dict[str, dict] = {}
    for k, vals in collected.items():
        ratio = len(vals) / total
        if ratio < MIN_VALID_RATIO:
            continue
        out[k] = {
            "value": reduce_values(vals, reducer_for(phase, k)),
            "valid_ratio": round(ratio, 3),
            "reducer": reducer_for(phase, k),
        }
    out["_frames_detected"] = {"value": n_points / total, "valid_ratio": 1.0, "reducer": "mean"}
    return out


# ============================================================
# 비교
# ============================================================


def compare_rep(
    user_rep: dict,
    expert_rep: dict,
    exercise: str,
    thresholds: Dict[str, float],
    display: List[str],
    advisory: set,
) -> dict:
    phases_out = []
    n_checked = 0
    n_available = 0

    common_phases = [p for p in user_rep["phases"] if p in expert_rep["phases"]]

    for phase in common_phases:
        um = user_rep["phases"][phase]
        em = expert_rep["phases"][phase]
        rows = []

        for metric in display:
            n_checked += 1
            if metric not in um or metric not in em:
                rows.append({
                    "metric": metric,
                    "label": METRIC_LABELS_KO.get(metric, metric),
                    "status": "unavailable",
                    "reason": "가림 또는 검출 실패",
                })
                continue

            n_available += 1
            uv = um[metric]["value"]
            ev = em[metric]["value"]
            diff = uv - ev
            thr = float(thresholds.get(metric, 0.0))
            is_advisory = metric in advisory
            over = (thr > 0) and (abs(diff) > thr) and not is_advisory

            rows.append({
                "metric": metric,
                "label": METRIC_LABELS_KO.get(metric, metric),
                "status": "advisory" if is_advisory else ("over" if over else "within"),
                "unit": "ratio" if metric in RATIO_METRICS else "deg",
                "point": um[metric]["reducer"],
                "user": round(uv, 2),
                "expert": round(ev, 2),
                "diff": round(diff, 2),
                "threshold": thr,
                "ratio_to_threshold": round(diff / thr, 3) if thr > 0 else None,
                "valid_ratio": min(um[metric]["valid_ratio"], em[metric]["valid_ratio"]),
            })

        phases_out.append({"phase": phase, "rows": rows})

    metric_ratio = n_available / max(n_checked, 1)
    excluded = metric_ratio < MIN_REP_METRIC_RATIO

    return {
        "rep": user_rep["index"],
        "expert_rep": expert_rep["index"],
        "excluded": excluded,
        "exclude_reason": f"측정 가능 지표 {metric_ratio:.0%}" if excluded else None,
        "metric_ratio": round(metric_ratio, 3),
        "phases": phases_out,
    }


def collect_findings(rep_result: dict) -> List[dict]:
    """임계를 넘은 항목만 편차 크기순으로."""
    found = []
    for ph in rep_result["phases"]:
        for r in ph["rows"]:
            if r.get("status") == "over":
                found.append({**r, "phase": ph["phase"]})
    found.sort(key=lambda x: abs(x.get("ratio_to_threshold") or 0), reverse=True)
    return found


# ============================================================
# 문장 생성
# ============================================================


def template_sentence(findings: List[dict], top_k: int = 3) -> str:
    if not findings:
        return "레퍼런스와 유의미한 차이가 없습니다."

    seen, uniq = set(), []
    for f in findings:                      # 이미 편차 크기순
        if f["metric"] in seen:
            continue
        seen.add(f["metric"])
        uniq.append(f)

    parts = []
    for f in uniq[:top_k]:
        pos, neg = DIRECTION_KO.get(f["metric"], ("더 큽니다", "더 작습니다"))
        word = pos if f["diff"] > 0 else neg
        amount = f"{abs(f['diff']):.1f}" + ("도" if f["unit"] == "deg" else "")
        parts.append(f"{f['phase']} 구간에서 {f['label']}을 {amount} {word}")
    return ". ".join(parts) + "."


def llm_sentence(exercise: str, rep: int, findings: List[dict], top_k: int = 3) -> str:
    """ANTHROPIC_API_KEY 가 있으면 LLM 으로, 없으면 템플릿으로."""
    if not findings:
        return "레퍼런스와 유의미한 차이가 없습니다."

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return template_sentence(findings, top_k)

    try:
        import anthropic
    except ImportError:
        print("  [경고] anthropic 패키지가 없어 템플릿으로 대체합니다.")
        return template_sentence(findings, top_k)

    payload = {
        "exercise": exercise,
        "rep": rep,
        "findings": [
            {
                "phase": f["phase"],
                "metric": f["label"],
                "point": f["point"],
                "user": f["user"],
                "reference": f["expert"],
                "diff": f["diff"],
                "unit": f["unit"],
            }
            for f in findings[:top_k]
        ],
    }

    system = (
        "너는 운동 자세 비교 결과를 한국어 문장으로 옮기는 역할이다.\n"
        "규칙:\n"
        "1. 아래 JSON 수치만 근거로 쓴다. 없는 내용을 추가하지 않는다.\n"
        "2. 원인을 추측하지 않는다. 무엇이 얼마나 다른지만 말한다.\n"
        "3. 옳고 그름을 판정하지 않는다. '틀렸다', '잘못됐다' 같은 표현을 쓰지 않는다.\n"
        "4. 2~3문장. 존댓말. 수치를 반드시 포함한다."
    )

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# ============================================================
# 메인
# ============================================================


def build_rep_data(cache: "SequenceCache", side: dict, exercise: str) -> List[dict]:
    """side 는 라벨 JSON 의 user 또는 expert 블록. rep 마다 video 를 덮어쓸 수 있다."""
    default_video = side.get("video")
    out = []
    for rep in side["reps"]:
        video = rep.get("video", default_video)
        if not video:
            raise ValueError(f"rep {rep['index']} 에 video 가 지정되지 않았습니다")
        seq = cache.get(video)
        side_lock = SideLock(exercise)
        phases = {}
        for phase, span in rep["phases"].items():
            phases[phase] = phase_metrics(seq, span, exercise, phase, side_lock)
        out.append({"index": rep["index"], "video": video, "phases": phases})
    return out


def pick_report(reps: List[dict], display: List[str]) -> List[dict]:
    """레퍼런스로 쓸 rep 을 숫자로 고르기 위한 품질 요약."""
    rows = []
    for r in reps:
        ratios, detected, n_avail, n_total = [], [], 0, 0
        for phase, metrics in r["phases"].items():
            detected.append(metrics.get("_frames_detected", {}).get("value", 0.0))
            for m in display:
                n_total += 1
                if m in metrics:
                    n_avail += 1
                    ratios.append(metrics[m]["valid_ratio"])
        rows.append({
            "rep": r["index"],
            "video": r["video"],
            "metric_coverage": round(n_avail / max(n_total, 1), 3),
            "mean_valid_ratio": round(statistics.fmean(ratios), 3) if ratios else 0.0,
            "mean_frames_detected": round(statistics.fmean(detected), 3) if detected else 0.0,
        })
    # 지표 커버리지 우선, 같으면 유효 비율
    rows.sort(key=lambda x: (x["metric_coverage"], x["mean_valid_ratio"]), reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="rep/phase 수동 라벨 기반 오프라인 비교")
    ap.add_argument("--labels", required=True, help="rep/phase 라벨 JSON")
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--html", default="demo.html")
    ap.add_argument("--template", default="demo_template.html")
    ap.add_argument("--llm", action="store_true", help="ANTHROPIC_API_KEY 로 문장 생성")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--pick-expert", action="store_true",
                    help="expert rep 품질만 비교하고 끝낸다 (레퍼런스 선정용)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    exercise = normalize_exercise_name(cfg["exercise"])
    thresholds = DELTA_THRESHOLDS.get(exercise, {})
    display = DISPLAY_METRICS.get(exercise, [])
    advisory = ADVISORY_METRICS.get(exercise, set())
    cache = SequenceCache(BASE_DIR, exercise)

    # ── 레퍼런스 선정 모드 ──
    if args.pick_expert:
        print(f"[1/2] 랜드마크 추출 ({exercise})")
        expert_reps = build_rep_data(cache, cfg["expert"], exercise)
        print("[2/2] rep 품질 비교")
        rows = pick_report(expert_reps, display)
        print(f"\n{'rep':>4} {'영상':<18} {'지표커버리지':>12} {'평균유효비율':>12} {'검출률':>8}")
        for r in rows:
            print(f"{r['rep']:>4} {r['video']:<18} {r['metric_coverage']:>12.1%} "
                  f"{r['mean_valid_ratio']:>12.1%} {r['mean_frames_detected']:>8.1%}")
        best = rows[0]
        print(f"\n권장: rep {best['rep']} ({best['video']})")
        Path(args.out).write_text(
            json.dumps({"exercise": exercise, "ranking": rows, "recommended": best},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"      {args.out} 저장")
        return

    print(f"[1/4] 랜드마크 추출 ({exercise})")
    print("[2/4] phase 단위 지표 집계")
    user_reps = build_rep_data(cache, cfg["user"], exercise)
    expert_reps = build_rep_data(cache, cfg["expert"], exercise)

    print("[3/4] 비교 및 판정")
    results = []
    for i, ur in enumerate(user_reps):
        er = expert_reps[i] if len(expert_reps) == len(user_reps) else expert_reps[0]
        rep_res = compare_rep(ur, er, exercise, thresholds, display, advisory)
        findings = collect_findings(rep_res)
        rep_res["findings"] = findings
        if rep_res["excluded"]:
            rep_res["sentence"] = f"이 rep 은 판정에서 제외됐습니다 ({rep_res['exclude_reason']})."
        elif args.llm:
            rep_res["sentence"] = llm_sentence(exercise, ur["index"], findings, args.top_k)
        else:
            rep_res["sentence"] = template_sentence(findings, args.top_k)
        results.append(rep_res)
        mark = "제외" if rep_res["excluded"] else f"{len(findings)}건"
        print(f"  rep {ur['index']}: {mark}")

    payload = {
        "exercise": exercise,
        "user_video": cfg["user"].get("video", user_reps[0]["video"]),
        "expert_video": expert_reps[0]["video"],
        "thresholds": thresholds,
        "threshold_note": "임시값. 9월 촬영 데이터로 재산출 예정.",
        "metric_labels": METRIC_LABELS_KO,
        "reps": results,
    }

    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4/4] {args.out} 저장")

    tpl_path = BASE_DIR / args.template
    if tpl_path.exists():
        html = tpl_path.read_text(encoding="utf-8").replace(
            "/*__DATA__*/null",
            json.dumps(payload, ensure_ascii=False),
        )
        Path(args.html).write_text(html, encoding="utf-8")
        print(f"      {args.html} 저장 — 브라우저로 바로 열면 됩니다")
    else:
        print(f"      [경고] {tpl_path.name} 이 없어 HTML 은 건너뜁니다")


if __name__ == "__main__":
    main()
