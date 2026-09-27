# -*- coding: utf-8 -*-
"""
unified_feedback_v4_llm.py
==========================

`unified_feedback_v4.py`를 직접 수정하지 않고 재사용하는 LLM 피드백 변형 버전.

핵심 차이
---------
- MediaPipe / phase model / expert alignment / metric 계산 / 렌더링은 v4를 그대로 사용한다.
- v4의 운동별 `choose_issue()`가 계산한 이상 지표/랜드마크/심각도는 그대로 유지하고,
  표시 문장만 지표 기반 LLM 피드백으로 교체한다.
- LLM에는 원본 영상이나 포즈 계산을 맡기지 않는다. 코드가 계산한 user/expert 지표,
  delta, threshold, phase, advisory/reliability metadata만 전달한다.
- 원격 LLM API 설정이 없으면 local 요약기로 대체하지 않고 실행을 중단한다.

실행
----
    python unified_feedback_v4_llm.py

원격 LLM 사용 예시(OpenAI-compatible chat completions)
----------------------------------------------------
    set LLM_FEEDBACK_PROVIDER=openai
    set OPENAI_API_KEY=...
    set LLM_FEEDBACK_MODEL=...
    python unified_feedback_v4_llm.py

주의
----
- `LLM_FEEDBACK_MODEL` 기본값은 의도적으로 비워 둔다. 모델명은 환경변수로 명시한다.
- API 키/모델/provider가 설정되지 않으면 local fallback 없이 설정 오류를 발생시킨다.
- LLM API는 완료된 rep count가 1, 2, 3...으로 증가할 때만 호출하고, 같은 rep 안에서는 마지막 API 응답을 재사용한다.
- 원본 v4 출력 파일을 덮어쓰지 않도록 output 파일명에 `_llm` suffix를 붙인다.
- realtime source가 웹캠이 아니라 영상 파일이면 입력 영상 EOF까지 처리하고 결과 MP4를 자동 저장한다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import unified_feedback_v4 as base

ORIGINAL_CHOOSE_ISSUE = base.choose_issue


# ============================================================
# 1. LLM feedback runtime config
# ============================================================

LLM_FEEDBACK_PROVIDER = os.environ.get("LLM_FEEDBACK_PROVIDER", "openai").strip().lower()
LLM_FEEDBACK_MODEL = os.environ.get("LLM_FEEDBACK_MODEL", "").strip()
LLM_FEEDBACK_ENDPOINT = os.environ.get(
    "LLM_FEEDBACK_ENDPOINT",
    "https://api.openai.com/v1/chat/completions",
).strip()
LLM_FEEDBACK_API_KEY_ENV = os.environ.get("LLM_FEEDBACK_API_KEY_ENV", "OPENAI_API_KEY").strip()
LLM_FEEDBACK_TIMEOUT_SEC = float(os.environ.get("LLM_FEEDBACK_TIMEOUT_SEC", "12"))
# 0 means send every computed metric. Set a positive value only if token/cost must be capped.
LLM_FEEDBACK_MAX_METRICS = int(os.environ.get("LLM_FEEDBACK_MAX_METRICS", "0"))
LLM_FEEDBACK_TRIGGER_NORM = float(os.environ.get("LLM_FEEDBACK_TRIGGER_NORM", "0.75"))
LLM_FEEDBACK_TEMPERATURE = float(os.environ.get("LLM_FEEDBACK_TEMPERATURE", "0.55"))
LLM_FEEDBACK_PRINT_PAYLOAD = os.environ.get("LLM_FEEDBACK_PRINT_PAYLOAD", "0").strip() == "1"
LLM_FEEDBACK_PRINT_RESPONSE = os.environ.get("LLM_FEEDBACK_PRINT_RESPONSE", "0").strip() == "1"
LLM_FEEDBACK_LOG_JSONL = os.environ.get(
    "LLM_FEEDBACK_LOG_JSONL",
    str(base.BASE_DIR / ".omx" / "logs" / "llm_feedback_debug.jsonl"),
).strip()
LLM_FEEDBACK_LOG_ENABLED = (
    os.environ.get("LLM_FEEDBACK_LOG_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}
    and LLM_FEEDBACK_LOG_JSONL.lower() not in {"", "0", "false", "off", "none", "no"}
)
DISABLED_LLM_PROVIDERS = {"", "local", "none", "off"}
GENERIC_FEEDBACK_PHRASES = {
    "운동자세 개선이 필요합니다",
    "운동 자세 개선이 필요합니다",
    "자세 개선이 필요합니다",
    "자세를 개선하세요",
    "자세를 교정하세요",
    "운동 자세를 교정하세요",
}


def _llm_debug_log_path() -> Optional[Path]:
    if not LLM_FEEDBACK_LOG_ENABLED:
        return None
    path = Path(LLM_FEEDBACK_LOG_JSONL).expanduser()
    if not path.is_absolute():
        path = base.BASE_DIR / path
    return path


def _write_llm_debug_log(event: Dict[str, Any]) -> None:
    """Append JSONL debug data for the exact LLM request/response/display text.

    The Authorization header/API key is intentionally never written.
    """
    path = _llm_debug_log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "provider": LLM_FEEDBACK_PROVIDER,
            "model": LLM_FEEDBACK_MODEL,
            "endpoint": LLM_FEEDBACK_ENDPOINT,
            **event,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        # Debug logging must not break realtime inference.
        print(f"[LLM][WARN] debug log write failed: {exc}")


# ============================================================
# 2. Metric metadata: 운동별 피드백 rule이 아니라 지표 설명만 둔다.
# ============================================================

METRIC_METADATA: Dict[str, Dict[str, Any]] = {
    "knee_angle": {
        "label_ko": "무릎각",
        "unit": "deg",
        "body_part": "knee",
        "description": "엉덩이-무릎-발목이 이루는 관절각입니다.",
        "larger_value_generally_means": "무릎이 더 펴져 있거나 굴곡이 덜한 상태",
        "smaller_value_generally_means": "무릎이 더 많이 접힌 상태",
    },
    "hip_angle": {
        "label_ko": "엉덩이각",
        "unit": "deg",
        "body_part": "hip",
        "description": "어깨-엉덩이-무릎이 이루는 관절각입니다.",
        "larger_value_generally_means": "고관절이 더 펴진 상태",
        "smaller_value_generally_means": "고관절이 더 접힌 상태",
    },
    "elbow_angle": {
        "label_ko": "팔꿈치각",
        "unit": "deg",
        "body_part": "elbow",
        "description": "어깨-팔꿈치-손목이 이루는 관절각입니다.",
        "larger_value_generally_means": "팔꿈치가 더 펴진 상태",
        "smaller_value_generally_means": "팔꿈치가 더 접힌 상태",
    },
    "elbow_angle_avg": {
        "label_ko": "팔꿈치각",
        "unit": "deg",
        "body_part": "elbow",
        "description": "팔꿈치 관절각의 대표값입니다.",
        "larger_value_generally_means": "팔꿈치가 더 펴진 상태",
        "smaller_value_generally_means": "팔꿈치가 더 접힌 상태",
    },
    "lockout_angle_min": {
        "label_ko": "락아웃각",
        "unit": "deg",
        "body_part": "elbow",
        "description": "상단/락아웃 근처에서 팔꿈치가 펴진 정도를 나타내는 각도입니다.",
        "larger_value_generally_means": "락아웃이 더 펴진 상태",
        "smaller_value_generally_means": "락아웃이 덜 된 상태",
    },
    "trunk_lean": {
        "label_ko": "몸통기울기",
        "unit": "deg",
        "body_part": "trunk",
        "description": "몸통 축이 수직선에서 벗어난 절대 각도입니다.",
        "larger_value_generally_means": "몸통이 더 많이 기울어진 상태",
        "smaller_value_generally_means": "몸통이 더 수직에 가까운 상태",
    },
    "trunk_lean_signed": {
        "label_ko": "몸통기울기부호",
        "unit": "deg",
        "body_part": "trunk",
        "description": "방향 부호를 포함한 몸통 기울기입니다.",
        "larger_value_generally_means": "정의된 양의 방향으로 더 기울어진 상태",
        "smaller_value_generally_means": "정의된 음의 방향으로 더 기울어진 상태",
    },
    "trunk_theta": {
        "label_ko": "몸통축각",
        "unit": "deg",
        "body_part": "trunk",
        "description": "몸통 축의 signed angle입니다.",
        "larger_value_generally_means": "정의된 양의 방향으로 몸통 축이 더 회전한 상태",
        "smaller_value_generally_means": "정의된 음의 방향으로 몸통 축이 더 회전한 상태",
    },
    "neck_lean": {
        "label_ko": "목-머리각",
        "unit": "deg",
        "body_part": "neck",
        "description": "머리/목 라인이 수직에서 벗어난 정도입니다.",
        "larger_value_generally_means": "머리/목이 더 많이 기울어진 상태",
        "smaller_value_generally_means": "머리/목이 더 수직에 가까운 상태",
    },
    "foot_flatness": {
        "label_ko": "발바닥밀착",
        "unit": "torso_ratio",
        "body_part": "foot",
        "description": "뒤꿈치와 발끝 y좌표 차이를 torso length로 정규화한 값입니다.",
        "larger_value_generally_means": "뒤꿈치와 발끝 높이 차이가 더 큰 상태",
        "smaller_value_generally_means": "발바닥 높이 차이가 더 작은 상태",
    },
    "head_hip_line": {
        "label_ko": "머리-엉덩이",
        "unit": "torso_ratio",
        "body_part": "head_trunk",
        "description": "머리와 엉덩이 중심의 수평 오프셋을 torso length로 정규화한 값입니다.",
        "larger_value_generally_means": "머리와 엉덩이 라인의 수평 차이가 더 큰 상태",
        "smaller_value_generally_means": "머리와 엉덩이가 수직 라인에 더 가까운 상태",
    },
    "wrist_elbow_x_diff": {
        "label_ko": "손목-팔꿈치",
        "unit": "torso_ratio",
        "body_part": "wrist_elbow",
        "description": "손목과 팔꿈치의 수평 오프셋을 torso length로 정규화한 값입니다.",
        "larger_value_generally_means": "손목과 팔꿈치의 수평 정렬 차이가 더 큰 상태",
        "smaller_value_generally_means": "손목과 팔꿈치가 수직 정렬에 더 가까운 상태",
    },
    "bench_line_diff": {
        "label_ko": "벤치라인",
        "unit": "torso_ratio",
        "body_part": "trunk",
        "description": "머리-어깨-엉덩이 라인에서 벗어난 정도를 정규화한 값입니다.",
        "larger_value_generally_means": "상체 기준선에서 더 많이 벗어난 상태",
        "smaller_value_generally_means": "상체 기준선에 더 가까운 상태",
    },
    "bar_proxy_conf": {
        "label_ko": "바벨Proxy",
        "unit": "confidence",
        "body_part": "bar_proxy",
        "description": "손/손목 기반 바벨 proxy 추정 신뢰도입니다. 자세 오류 판단값이 아니라 참고값입니다.",
        "larger_value_generally_means": "바벨 proxy 추정 신뢰도가 높은 상태",
        "smaller_value_generally_means": "바벨 proxy 추정 신뢰도가 낮은 상태",
    },
}

METRIC_LANDMARKS: Dict[str, List[int]] = {
    "knee_angle": [23, 24, 25, 26, 27, 28],
    "hip_angle": [11, 12, 23, 24, 25, 26],
    "elbow_angle": [11, 12, 13, 14, 15, 16],
    "elbow_angle_avg": [11, 12, 13, 14, 15, 16],
    "lockout_angle_min": [11, 12, 13, 14, 15, 16],
    "trunk_lean": [11, 12, 23, 24],
    "trunk_lean_signed": [11, 12, 23, 24],
    "trunk_theta": [11, 12, 23, 24],
    "neck_lean": [0, 11, 12],
    "foot_flatness": [27, 28, 29, 30, 31, 32],
    "head_hip_line": [0, 23, 24],
    "wrist_elbow_x_diff": [13, 14, 15, 16],
    "bench_line_diff": [0, 11, 12, 23, 24],
    "bar_proxy_conf": [],
}


COACHING_CUES: Dict[str, Dict[str, Dict[str, str]]] = {
    "squat": {
        "knee_angle": {
            "user_higher": "무릎각이 더 크면 깊이가 얕은 쪽이므로 무릎을 더 접고 엉덩이를 아래로 내려라.",
            "user_lower": "무릎각이 더 작으면 너무 깊게 접힌 쪽이므로 무릎을 과하게 밀지 말고 깊이를 조금 줄여라.",
        },
        "hip_angle": {
            "user_higher": "엉덩이각이 더 크면 고관절이 덜 접힌 쪽이므로 엉덩이를 더 뒤로 빼고 내려라.",
            "user_lower": "엉덩이각이 더 작으면 고관절이 과하게 접힌 쪽이므로 가슴을 세우고 엉덩이를 조금 펴라.",
        },
        "trunk_lean": {
            "user_higher": "몸통이 더 숙여졌으니 가슴을 들고 상체를 더 세워라.",
            "user_lower": "몸통이 너무 세워진 쪽이면 엉덩이를 살짝 뒤로 빼서 균형을 맞춰라.",
        },
        "foot_flatness": {
            "user_higher": "발 앞뒤 높이 차이가 크므로 발바닥 전체를 바닥에 더 눌러라.",
            "user_lower": "발바닥 접지는 유지하되 체중이 한쪽으로 쏠리지 않게 해라.",
        },
        "head_hip_line": {
            "user_higher": "머리와 엉덩이 라인이 벌어졌으니 시선과 골반을 같은 축에 더 맞춰라.",
            "user_lower": "머리-엉덩이 축은 크게 벗어나지 않았으니 현재 정렬을 유지해라.",
        },
    },
    "deadlift": {
        "knee_angle": {
            "user_higher": "무릎이 더 펴진 쪽이면 바닥 구간에서 무릎을 조금 더 접어 바에 몸을 가까이 둬라.",
            "user_lower": "무릎이 더 접힌 쪽이면 스쿼트처럼 앉지 말고 무릎을 조금 더 펴며 힙힌지를 만들어라.",
        },
        "hip_angle": {
            "user_higher": "고관절이 덜 접힌 쪽이면 엉덩이를 뒤로 보내 힙힌지를 더 만들어라.",
            "user_lower": "고관절이 더 접힌 쪽이면 가슴을 열고 엉덩이를 조금 펴서 락아웃 방향으로 가져가라.",
        },
        "trunk_lean": {
            "user_higher": "상체가 더 숙여졌으니 가슴을 열고 등을 단단히 세워라.",
            "user_lower": "상체가 너무 세워진 쪽이면 엉덩이를 뒤로 빼고 바를 몸 가까이에 둬라.",
        },
        "elbow_angle": {
            "user_higher": "팔꿈치각은 보조 지표다. 팔을 당기려 하지 말고 길게 늘어뜨려라.",
            "user_lower": "팔꿈치가 접히는 쪽이면 팔로 당기지 말고 팔을 곧게 펴라.",
        },
        "head_hip_line": {
            "user_higher": "머리와 골반 축이 벌어졌으니 목을 중립으로 두고 몸통 축을 맞춰라.",
            "user_lower": "머리-골반 축은 크게 벗어나지 않았으니 중립을 유지해라.",
        },
    },
    "benchpress": {
        "elbow_angle": {
            "user_higher": "팔꿈치가 더 펴진 쪽이면 하강 구간에서 팔꿈치를 더 접어 가동범위를 확보해라.",
            "user_lower": "팔꿈치가 더 접힌 쪽이면 상단에서 팔꿈치를 더 펴 락아웃을 완성해라.",
        },
        "lockout_angle_min": {
            "user_higher": "락아웃각이 충분히 큰 쪽이면 팔꿈치 펴짐을 유지하되 어깨가 뜨지 않게 해라.",
            "user_lower": "락아웃각이 작으면 상단에서 팔꿈치를 더 펴고 바를 끝까지 밀어라.",
        },
        "wrist_elbow_x_diff": {
            "user_higher": "손목과 팔꿈치 수직선이 벌어졌으니 손목을 팔꿈치 위로 더 맞춰라.",
            "user_lower": "손목-팔꿈치 정렬은 크게 벗어나지 않았으니 손목을 꺾지 말고 유지해라.",
        },
        "bench_line_diff": {
            "user_higher": "상체 기준선이 흔들리므로 머리·어깨·엉덩이를 벤치에 더 고정해라.",
            "user_lower": "벤치라인은 크게 벗어나지 않았으니 상체 고정을 유지해라.",
        },
    },
}


GENERIC_CUES: Dict[str, str] = {
    "user_higher": "사용자 값이 전문가보다 크다. larger/smaller 의미를 보고 구체적인 동작을 줄이거나 늘리는 큐로 바꿔라.",
    "user_lower": "사용자 값이 전문가보다 작다. larger/smaller 의미를 보고 구체적인 동작을 줄이거나 늘리는 큐로 바꿔라.",
    "similar": "사용자 값과 전문가 값이 비슷하다. 중요한 차이가 없으면 이 지표를 우선 피드백으로 고르지 마라.",
}


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if out == out and abs(out) != float("inf") else None


def _round_value(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), ndigits)


def _format_metric_value(value: Optional[float], unit: str) -> str:
    if value is None:
        return "-"
    if unit == "deg":
        return f"{value:.1f}도"
    if unit == "confidence":
        return f"{value:.2f}"
    return f"{value:.3f}"


def _metric_sort_score(record: Dict[str, Any]) -> float:
    norm = record.get("normalized_abs_delta")
    if norm is not None:
        return float(norm)
    delta = record.get("abs_delta")
    return 0.0 if delta is None else float(delta)


def _delta_direction(delta: Optional[float]) -> str:
    if delta is None:
        return "unknown"
    if float(delta) > 1e-6:
        return "user_higher"
    if float(delta) < -1e-6:
        return "user_lower"
    return "similar"


def _recommended_cue(exercise: str, key: str, delta: Optional[float]) -> Optional[str]:
    direction = _delta_direction(delta)
    if direction not in {"user_higher", "user_lower", "similar"}:
        return None
    if direction == "similar":
        return GENERIC_CUES["similar"]
    return (
        COACHING_CUES.get(exercise, {}).get(key, {}).get(direction)
        or COACHING_CUES.get("*", {}).get(key, {}).get(direction)
        or GENERIC_CUES[direction]
    )


def build_metric_records(
    exercise: str,
    user_m: Dict[str, float],
    expert_m: Dict[str, float],
    deltas: Dict[str, float],
    *,
    max_metrics: Optional[int] = LLM_FEEDBACK_MAX_METRICS,
) -> List[Dict[str, Any]]:
    """Build LLM-ready metric records from all already-computed v4 metrics."""
    exercise = base.normalize_exercise_name(exercise)
    display_keys = list(base.DISPLAY_METRICS.get(exercise, []))
    keys: List[str] = []
    for key in display_keys + sorted(user_m.keys()) + sorted(expert_m.keys()) + sorted(deltas.keys()):
        if key not in keys:
            keys.append(key)

    thresholds = base.DELTA_THRESHOLDS.get(exercise, {})
    advisory = base.ADVISORY_METRICS.get(exercise, set())
    records: List[Dict[str, Any]] = []

    for key in keys:
        user_value = _finite_float(user_m.get(key))
        expert_value = _finite_float(expert_m.get(key))
        delta = _finite_float(deltas.get(key))
        if user_value is None and expert_value is None and delta is None:
            continue

        threshold = _finite_float(thresholds.get(key))
        abs_delta = None if delta is None else abs(delta)
        normalized_abs_delta = (
            None
            if abs_delta is None or threshold is None or threshold <= 0
            else abs_delta / threshold
        )
        meta = copy.deepcopy(METRIC_METADATA.get(key, {}))
        meta.setdefault("label_ko", base.METRIC_LABELS_KO.get(key, key))
        meta.setdefault("unit", "raw")
        meta.setdefault("description", "프로그램이 계산한 자세 비교 지표입니다.")

        is_advisory = key in advisory
        direction = _delta_direction(delta)
        cue = _recommended_cue(exercise, key, delta)
        records.append(
            {
                "name": key,
                "label_ko": meta["label_ko"],
                "description": meta["description"],
                "body_part": meta.get("body_part"),
                "unit": meta.get("unit", "raw"),
                "larger_value_generally_means": meta.get("larger_value_generally_means"),
                "smaller_value_generally_means": meta.get("smaller_value_generally_means"),
                "user_value": _round_value(user_value),
                "expert_value": _round_value(expert_value),
                "delta_user_minus_expert": _round_value(delta),
                "abs_delta": _round_value(abs_delta),
                "threshold": _round_value(threshold),
                "normalized_abs_delta": _round_value(normalized_abs_delta),
                "delta_direction": direction,
                "recommended_action_cue": cue,
                "cue_style_requirement": "각도 설명만 하지 말고 recommended_action_cue처럼 좁혀라/넓혀라/접어라/펴라/세워라/고정해라 형태의 동작 지시로 변환한다.",
                "is_advisory": bool(is_advisory),
                "reliability": 0.45 if is_advisory else 0.90,
                "landmarks": METRIC_LANDMARKS.get(key, []),
            }
        )

    records.sort(key=_metric_sort_score, reverse=True)
    if max_metrics is not None and max_metrics > 0:
        return records[:max_metrics]
    return records


def build_llm_feedback_payload(
    exercise: str,
    phase: str,
    user_m: Dict[str, float],
    expert_m: Dict[str, float],
    deltas: Dict[str, float],
    *,
    rep_count: Optional[int] = None,
    frame_info: Optional[str] = None,
    visual_issue: Optional[base.Issue] = None,
) -> Dict[str, Any]:
    """Serialize the exact evidence the LLM is allowed to use."""
    exercise = base.normalize_exercise_name(exercise)
    visual_issue_data = None
    if visual_issue is not None:
        visual_issue_data = {
            "key": visual_issue.key,
            "message_from_original_rules": visual_issue.message,
            "severity_from_original_rules": _round_value(visual_issue.severity),
            "landmarks_from_original_rules": list(visual_issue.landmarks),
            "contract": "Skeleton color/normal-range judgment uses this original v4 issue. LLM only rewrites the feedback text.",
        }
    return {
        "schema_version": "metric_feedback_v1",
        "task": "generate_exercise_feedback_from_computed_metrics",
        "guardrails": [
            "Use only the supplied metric records as evidence.",
            "The metrics list contains every computed user/expert/delta metric available for this frame/rep.",
            "Do not decide whether the skeleton should be red/green; visual_issue is already computed by the original v4 rules.",
            "Use visual_issue as the default priority topic for the feedback text unless another supplied metric is clearly more actionable.",
            "Do not infer unseen body positions, injury risk, diagnosis, or medical advice.",
            "Do not recalculate pose, phase, rep count, or alignment.",
            "Select the most important 1-3 metric differences; ignore tiny/noisy differences.",
            "If evidence is weak or advisory, say so briefly.",
            "Do not use generic phrases such as '운동자세 개선이 필요합니다' or '자세를 교정하세요'.",
            "The summary must name the priority metric or body part and the direction of the difference.",
            "main_feedback must include concrete evidence from user_value/expert_value/delta when available.",
            "Do not stop at angle descriptions. Convert the metric difference into a movement cue.",
            "Prefer verbs like 더 접어라, 더 펴라, 더 세워라, 더 낮춰라, 더 뒤로 빼라, 더 좁혀라, 더 맞춰라, 더 고정해라.",
            "Use each selected metric's recommended_action_cue as the primary coaching direction unless it conflicts with the numbers.",
            "correction_cues must be actionable and specific to the selected metric, not a generic encouragement.",
            "Vary wording across reps while staying grounded in the selected metric records.",
            "Return Korean feedback in the requested JSON schema.",
        ],
        "context": {
            "exercise": exercise,
            "phase": str(phase),
            "rep_count": rep_count,
            "frame_info": frame_info,
            "comparison": "user_vs_expert_same_phase",
            "visual_issue": visual_issue_data,
        },
        "metrics": build_metric_records(exercise, user_m, expert_m, deltas),
        "output_schema": {
            "summary": "one short Korean action cue, e.g. '무릎을 더 접어 깊이를 확보하세요'",
            "main_feedback": ["1-3 Korean sentences: metric evidence + what movement to increase/decrease"],
            "correction_cues": ["1-3 concise action cues using verbs like 접어라/펴라/세워라/낮춰라/맞춰라"],
            "priority_metric": "metric name or null",
            "selected_metrics": ["metric names used"],
            "severity": "none|low|medium|high",
            "confidence": "low|medium|high",
        },
    }


def _payload_cache_key(payload: Dict[str, Any]) -> str:
    compact = {
        "context": payload.get("context", {}),
        "metrics": [
            {
                "name": m.get("name"),
                "u": None if m.get("user_value") is None else round(float(m["user_value"]), 2),
                "e": None if m.get("expert_value") is None else round(float(m["expert_value"]), 2),
                "d": None
                if m.get("delta_user_minus_expert") is None
                else round(float(m["delta_user_minus_expert"]), 2),
            }
            for m in payload.get("metrics", [])
        ],
    }
    blob = json.dumps(compact, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _normalize_feedback_dict(raw: Dict[str, Any], *, provider: str) -> Dict[str, Any]:
    selected = raw.get("selected_metrics") or []
    if isinstance(selected, str):
        selected = [selected]
    main_feedback = raw.get("main_feedback") or []
    if isinstance(main_feedback, str):
        main_feedback = [main_feedback]
    cues = raw.get("correction_cues") or []
    if isinstance(cues, str):
        cues = [cues]

    severity = str(raw.get("severity") or "low").lower()
    if severity not in {"none", "low", "medium", "high"}:
        severity = "low"
    confidence = str(raw.get("confidence") or "medium").lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    return {
        "summary": str(raw.get("summary") or "제공된 지표 기준으로 큰 차이가 없습니다."),
        "main_feedback": [str(x) for x in main_feedback[:3]],
        "correction_cues": [str(x) for x in cues[:3]],
        "priority_metric": raw.get("priority_metric"),
        "selected_metrics": [str(x) for x in selected[:3]],
        "severity": severity,
        "confidence": confidence,
        "provider": provider,
    }


def _no_feedback(provider: str) -> Dict[str, Any]:
    return _normalize_feedback_dict(
        {
            "summary": "",
            "main_feedback": [],
            "correction_cues": [],
            "priority_metric": None,
            "selected_metrics": [],
            "severity": "none",
            "confidence": "medium",
        },
        provider=provider,
    )


def _is_generic_feedback_text(text: Any) -> bool:
    normalized = str(text or "").strip().replace(" ", "")
    if not normalized:
        return True
    for phrase in GENERIC_FEEDBACK_PHRASES:
        if phrase.replace(" ", "") in normalized:
            return True
    return False


def _compact_display_text(text: str, max_chars: int = 0) -> str:
    text = " ".join(str(text or "").strip().split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _metric_evidence_text(metric: Dict[str, Any]) -> str:
    label = metric.get("label_ko") or metric.get("name") or "지표"
    unit = str(metric.get("unit") or "raw")
    user_txt = _format_metric_value(metric.get("user_value"), unit)
    expert_txt = _format_metric_value(metric.get("expert_value"), unit)
    delta_txt = _format_metric_value(metric.get("delta_user_minus_expert"), unit)
    cue = str(metric.get("recommended_action_cue") or "").strip()
    if cue and not _is_generic_feedback_text(cue):
        return f"{cue} ({label}: U {user_txt} / E {expert_txt} / Δ {delta_txt})"
    return f"{label} 차이 확인: U {user_txt} / E {expert_txt} / Δ {delta_txt}"


def _feedback_display_message(feedback: Dict[str, Any], metric: Dict[str, Any]) -> str:
    candidates: List[str] = []
    for key in ("main_feedback", "correction_cues"):
        values = feedback.get(key) or []
        if isinstance(values, str):
            values = [values]
        candidates.extend(str(v) for v in values if str(v).strip())
    summary = str(feedback.get("summary") or "").strip()
    if summary:
        candidates.append(summary)

    for text in candidates:
        if not _is_generic_feedback_text(text):
            return _compact_display_text(text)
    return _compact_display_text(_metric_evidence_text(metric))


def _call_original_choose_issue(
    exercise: str,
    user_m: Dict[str, float],
    expert_m: Dict[str, float],
    deltas: Dict[str, float],
    phase: str,
    rep_count: Optional[int],
) -> Optional[base.Issue]:
    try:
        return ORIGINAL_CHOOSE_ISSUE(exercise, user_m, expert_m, deltas, phase, rep_count=rep_count)
    except TypeError:
        return ORIGINAL_CHOOSE_ISSUE(exercise, user_m, expert_m, deltas, phase)


def _payload_metric_by_name(payload: Dict[str, Any], name: Any) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    target = str(name)
    for metric in payload.get("metrics", []):
        if str(metric.get("name")) == target:
            return metric
    return None


def _metric_for_visual_issue(payload: Dict[str, Any], visual_issue: base.Issue) -> Optional[Dict[str, Any]]:
    direct = _payload_metric_by_name(payload, visual_issue.key)
    if direct is not None:
        return direct

    key = str(visual_issue.key)
    candidates: List[str] = []
    if "knee" in key:
        candidates.append("knee_angle")
    if "hip" in key:
        candidates.append("hip_angle")
    if "trunk" in key:
        candidates.append("trunk_lean")
    if "elbow" in key:
        candidates.append("elbow_angle")
    if "lockout" in key:
        candidates.extend(["lockout_angle_min", "elbow_angle"])
    if "wrist" in key:
        candidates.append("wrist_elbow_x_diff")
    if "bench" in key:
        candidates.append("bench_line_diff")
    if "head" in key:
        candidates.append("head_hip_line")

    for candidate in candidates:
        metric = _payload_metric_by_name(payload, candidate)
        if metric is not None:
            return metric
    return None


class LLMFeedbackEngine:
    """Rep-gated metric-to-feedback engine used as a drop-in choose_issue replacement."""

    def __init__(self) -> None:
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.last_feedback: Optional[Dict[str, Any]] = None
        self.last_api_rep_key: Optional[Tuple[str, int]] = None
        self.last_feedback_source = "none"
        self.logged_display_keys: set[Tuple[str, Optional[int], str, str]] = set()
        self.provider = LLM_FEEDBACK_PROVIDER
        self.api_key = os.environ.get(LLM_FEEDBACK_API_KEY_ENV, "").strip()

    @property
    def remote_enabled(self) -> bool:
        return not self.api_config_errors()

    def api_config_errors(self) -> List[str]:
        errors: List[str] = []
        if self.provider in DISABLED_LLM_PROVIDERS:
            errors.append(
                "LLM_FEEDBACK_PROVIDER는 local/none/off가 아니라 OpenAI-compatible API provider여야 합니다 "
                "(예: set LLM_FEEDBACK_PROVIDER=openai)."
            )
        if not self.api_key:
            errors.append(f"{LLM_FEEDBACK_API_KEY_ENV} 환경변수에 API key를 설정해야 합니다.")
        if not LLM_FEEDBACK_MODEL:
            errors.append("LLM_FEEDBACK_MODEL 환경변수에 사용할 모델명을 설정해야 합니다.")
        if not LLM_FEEDBACK_ENDPOINT:
            errors.append("LLM_FEEDBACK_ENDPOINT가 비어 있습니다.")
        return errors

    def require_api_config(self) -> None:
        errors = self.api_config_errors()
        if errors:
            setup_hint = (
                "예시: set LLM_FEEDBACK_PROVIDER=openai && "
                "set OPENAI_API_KEY=... && "
                "set LLM_FEEDBACK_MODEL=gpt-4o-mini"
            )
            raise RuntimeError("[LLM] API 설정이 필요합니다. local fallback은 비활성화되어 있습니다.\n- " + "\n- ".join(errors) + "\n" + setup_hint)

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.require_api_config()
        context = payload.get("context", {})
        exercise = str(context.get("exercise") or "").strip().lower()
        rep_count_raw = context.get("rep_count")
        rep_count = int(rep_count_raw) if rep_count_raw is not None else None

        if rep_count is not None:
            if rep_count <= 0:
                self.last_feedback_source = "waiting_for_completed_rep"
                return _no_feedback("waiting_for_completed_rep")

            rep_key = (exercise, rep_count)
            if self.last_api_rep_key == rep_key and self.last_feedback is not None:
                self.last_feedback_source = "rep_reuse"
                return self.last_feedback

        key = _payload_cache_key(payload)
        if key in self.cache:
            feedback = self.cache[key]
            self.last_feedback = feedback
            self.last_feedback_source = "cache"
            if rep_count is not None:
                self.last_api_rep_key = (exercise, rep_count)
            return feedback

        if LLM_FEEDBACK_PRINT_PAYLOAD:
            print("[LLM_PAYLOAD]", json.dumps(payload, ensure_ascii=False, indent=2))

        try:
            feedback = self._call_openai_compatible(payload)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            feedback = _no_feedback("api_error")
            self.last_feedback = feedback
            self.last_feedback_source = "api_error"
            if rep_count is not None:
                self.last_api_rep_key = (exercise, rep_count)
            _write_llm_debug_log(
                {
                    "event": "api_call_failed",
                    "payload_cache_key": key,
                    "context": context,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[LLM][WARN] API call failed for rep={rep_count}: {error_text}. Using original v4 text for this rep.")
            return feedback

        self.cache[key] = feedback
        self.last_feedback = feedback
        self.last_feedback_source = "api"
        if rep_count is not None:
            self.last_api_rep_key = (exercise, rep_count)
        return feedback

    def _call_openai_compatible(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "너는 운동 자세 피드백 문장 생성기다. "
            "제공된 JSON metrics에 근거해서만 판단한다. "
            "영상, 관절각, phase, rep를 새로 추정하지 않는다. "
            "없는 지표를 상상하지 않는다. "
            "의학적 진단이나 부상 단정은 하지 않는다. "
            "summary에는 반드시 선택한 지표명 또는 신체 부위와 차이 방향을 넣는다. "
            "'운동자세 개선이 필요합니다', '자세를 교정하세요' 같은 포괄 문구는 금지한다. "
            "각도 정보만 설명하지 말고, recommended_action_cue를 우선 참고해서 동작 지시로 바꾼다. "
            "더 접어라, 더 펴라, 더 세워라, 더 낮춰라, 더 뒤로 빼라, 더 좁혀라, 더 맞춰라, 더 고정해라 같은 구체 동사를 쓴다. "
            "main_feedback에는 가능한 경우 사용자값, 전문가값, delta 중 하나 이상과 동작 방향을 함께 포함한다. "
            "correction_cues는 선택 지표에 맞춘 한 가지 행동 지시로 쓴다. "
            "반복마다 같은 문장 패턴을 피하되, 숫자와 근거는 바꾸지 않는다. "
            "중요한 지표 1~3개만 골라 한국어 JSON만 출력한다."
        )
        body = {
            "model": LLM_FEEDBACK_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": LLM_FEEDBACK_TEMPERATURE,
        }
        payload_key = _payload_cache_key(payload)
        user_json = body["messages"][1]["content"]
        _write_llm_debug_log(
            {
                "event": "request",
                "payload_cache_key": payload_key,
                "request_body": body,
                "payload": payload,
                "char_counts": {
                    "system_prompt": len(system_prompt),
                    "user_json": len(user_json),
                    "request_body_json": len(json.dumps(body, ensure_ascii=False)),
                },
            }
        )
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            LLM_FEEDBACK_ENDPOINT,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=LLM_FEEDBACK_TIMEOUT_SEC) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            _write_llm_debug_log(
                {
                    "event": "http_error",
                    "payload_cache_key": payload_key,
                    "status_code": exc.code,
                    "detail": detail,
                }
            )
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            _write_llm_debug_log(
                {
                    "event": "network_error",
                    "payload_cache_key": payload_key,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "timeout_sec": LLM_FEEDBACK_TIMEOUT_SEC,
                }
            )
            raise RuntimeError(f"LLM network/timeout error: {type(exc).__name__}: {exc}") from exc

        parsed = json.loads(raw)
        content = parsed["choices"][0]["message"]["content"]
        if LLM_FEEDBACK_PRINT_RESPONSE:
            print("[LLM_RESPONSE]", content)
        try:
            feedback_raw = json.loads(content)
        except json.JSONDecodeError:
            # Some models wrap JSON in fences; recover the object body.
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end <= start:
                _write_llm_debug_log(
                    {
                        "event": "parse_error",
                        "payload_cache_key": payload_key,
                        "raw_response": raw,
                        "message_content": content,
                    }
                )
                raise
            feedback_raw = json.loads(content[start : end + 1])
        feedback = _normalize_feedback_dict(feedback_raw, provider=self.provider)
        _write_llm_debug_log(
            {
                "event": "response",
                "payload_cache_key": payload_key,
                "raw_response": raw,
                "message_content": content,
                "feedback_raw": feedback_raw,
                "feedback_normalized": feedback,
                "char_counts": {
                    "raw_response": len(raw),
                    "message_content": len(content),
                    "summary": len(str(feedback.get("summary") or "")),
                    "main_feedback_joined": len(" ".join(str(x) for x in feedback.get("main_feedback", []))),
                    "correction_cues_joined": len(" ".join(str(x) for x in feedback.get("correction_cues", []))),
                },
            }
        )
        return feedback

    def choose_issue(
        self,
        exercise: str,
        user_m: Dict[str, float],
        expert_m: Dict[str, float],
        deltas: Dict[str, float],
        phase: str,
        rep_count: Optional[int] = None,
    ) -> Optional[base.Issue]:
        # 시각화 판정은 원본 v4를 그대로 쓴다.
        # LLM priority_metric/severity/landmarks를 Issue에 반영하면 skeleton 색과
        # "정상 범위"가 LLM 응답에 따라 바뀌므로, 여기서는 원본 issue가 없을 때만 정상으로 둔다.
        visual_issue = _call_original_choose_issue(exercise, user_m, expert_m, deltas, phase, rep_count)
        if visual_issue is None:
            return None

        payload = build_llm_feedback_payload(
            exercise,
            phase,
            user_m,
            expert_m,
            deltas,
            rep_count=rep_count,
            visual_issue=visual_issue,
        )
        feedback = self.generate(payload)
        metric = _payload_metric_by_name(payload, feedback.get("priority_metric")) or _metric_for_visual_issue(
            payload, visual_issue
        )
        if metric is not None and feedback.get("severity") != "none":
            message = _feedback_display_message(feedback, metric)
        else:
            # API가 아직 호출되지 않았거나 LLM이 usable feedback을 못 준 경우에도
            # 시각 판정은 원본 issue 그대로 유지하고 문장만 원본으로 되돌린다.
            message = visual_issue.message
        payload_key = _payload_cache_key(payload)
        display_key = (base.normalize_exercise_name(exercise), rep_count, payload_key, message)
        if display_key not in self.logged_display_keys:
            self.logged_display_keys.add(display_key)
            _write_llm_debug_log(
                {
                    "event": "display_message",
                    "payload_cache_key": payload_key,
                    "feedback_source": self.last_feedback_source,
                    "context": payload.get("context", {}),
                    "priority_metric": feedback.get("priority_metric"),
                    "selected_metrics": feedback.get("selected_metrics"),
                    "metric_used_for_display": metric,
                    "visual_issue_message": visual_issue.message,
                    "display_message": message,
                    "display_message_chars": len(message),
                    "feedback_normalized": feedback,
                }
            )
        return base.Issue(
            visual_issue.key,
            message,
            visual_issue.severity,
            list(visual_issue.landmarks),
        )


ENGINE = LLMFeedbackEngine()


def _is_camera_like_realtime_source(source: Any) -> bool:
    if isinstance(source, int):
        return True
    return str(source).strip().isdigit()


def _resolve_realtime_video_source_path() -> Optional[Tuple[Path, Optional[str]]]:
    source = base.REALTIME_SOURCE
    if _is_camera_like_realtime_source(source):
        return None

    if base._is_video_config_realtime_source(source):  # type: ignore[attr-defined]
        source_path, exercise = base._resolve_video_config_realtime_source(base.REALTIME_INITIAL_EXERCISE)  # type: ignore[attr-defined]
        return Path(source_path), str(exercise)

    text = str(source).strip()
    if not text or "://" in text:
        return None
    source_path = base._resolve_project_path(text)  # type: ignore[attr-defined]
    if not source_path.exists():
        return None
    return Path(source_path), None


def _default_realtime_llm_output_path() -> Optional[Path]:
    resolved = _resolve_realtime_video_source_path()
    if resolved is None:
        return None
    source_path, exercise = resolved
    stem = source_path.stem
    if exercise and not stem.lower().startswith(exercise.lower()):
        stem = f"{exercise}_{stem}"
    return base.BASE_DIR / "infer" / "output" / f"{stem}_realtime_llm.mp4"


def _configure_realtime_video_recording() -> None:
    """For realtime mode backed by a video file, save the full input-length LLM result MP4."""
    if str(base.RUN_MODE).strip().lower() != "realtime":
        return
    if _is_camera_like_realtime_source(base.REALTIME_SOURCE):
        return

    # A video source should be processed once from frame 0 to EOF, not looped/truncated like an interactive webcam smoke run.
    # Force headless mode so cv2.waitKey / window events cannot stop file rendering before EOF.
    base.REALTIME_DISPLAY = False
    base.REALTIME_LOOP_VIDEO = False
    base.REALTIME_MAX_FRAMES = None

    if not base.REALTIME_OUTPUT:
        output_path = _default_realtime_llm_output_path()
        if output_path is not None:
            base.REALTIME_OUTPUT = str(output_path)


def install_llm_variant() -> None:
    """Patch v4 runtime objects so this file behaves like a separate version."""
    base.choose_issue = ENGINE.choose_issue  # type: ignore[assignment]
    base.REALTIME_WINDOW_NAME = "Unified Feedback Realtime LLM"

    for cfg in base.VIDEO_CONFIG.values():
        out = cfg.get("output")
        if not out:
            continue
        path = Path(str(out))
        if path.suffix and not path.stem.endswith("_llm"):
            cfg["output"] = str(path.with_name(f"{path.stem}_llm{path.suffix}"))

    if base.REALTIME_OUTPUT:
        path = Path(str(base.REALTIME_OUTPUT))
        if path.suffix and not path.stem.endswith("_llm"):
            base.REALTIME_OUTPUT = str(path.with_name(f"{path.stem}_llm{path.suffix}"))

    _configure_realtime_video_recording()


def main() -> None:
    ENGINE.require_api_config()
    install_llm_variant()
    print(
        "[LLM] metric-grounded feedback enabled: "
        f"provider={LLM_FEEDBACK_PROVIDER}, mode=api, model={LLM_FEEDBACK_MODEL}, endpoint={LLM_FEEDBACK_ENDPOINT}"
    )
    log_path = _llm_debug_log_path()
    if log_path is not None:
        print(f"[LLM] debug log: {log_path}")
    base.main()


if __name__ == "__main__":
    main()
