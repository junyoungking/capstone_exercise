"""Stage 2 measurement built on top of the original v4 metric function."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

import numpy as np

from feedback import config
from feedback.quality import observation_reason, window_reason
from unified_feedback_v4 import SideLock, compute_metrics, normalize_exercise_name


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _phase_detection(valid: np.ndarray, start: int, end: int, position: int | None = None) -> float:
    length = max(0, int(end) - int(start))
    if position is not None and length < 5:
        start = max(0, position - config.POSITION_WINDOW_RADIUS)
        end = min(len(valid), position + config.POSITION_WINDOW_RADIUS + 1)
        return float(np.count_nonzero(valid[start:end]) / 5)
    if length == 0:
        return 0.0
    return float(np.count_nonzero(valid[start:end]) / length)


def _tempo(phases: dict, fps: float) -> dict:
    safe_fps = float(fps) if fps else 0.0

    def seconds(phase: str) -> float | None:
        if safe_fps <= 0:
            return None
        start, end = phases[phase]
        return (int(end) - int(start)) / safe_fps

    down_s = seconds("down")
    bottom_s = seconds("bottom")
    up_s = seconds("up")
    ratio = None if down_s is None or up_s in (None, 0.0) else down_s / up_s
    return {"down_s": down_s, "bottom_s": bottom_s, "up_s": up_s, "ratio": ratio}


def _rep_bounds(rep: dict, total: int) -> range:
    start = max(0, int(rep["phases"]["down"][0]) - config.FRAME_PADDING)
    end = min(total, int(rep["phases"]["up"][1]) + config.FRAME_PADDING)
    return range(start, end)


def _frame_sides(lms: np.ndarray, valid: np.ndarray, segments: dict) -> list[str]:
    exercise = segments["exercise"]
    total = int(segments["total"])
    side_lock = SideLock(exercise)
    sides: list[str] = []
    for frame in range(total):
        current = lms[frame] if bool(valid[frame]) else None
        sides.append(side_lock.update(current))
    for rep in segments["reps"]:
        side = rep["side"]
        for frame in _rep_bounds(rep, total):
            sides[frame] = side
    return sides


def _frame_metrics(
    lms: np.ndarray,
    valid: np.ndarray,
    exercise: str,
    display_metrics: list[str],
    sides: list[str],
) -> dict[str, list[float | None]]:
    output = {metric: [] for metric in display_metrics}
    for frame, is_valid in enumerate(valid):
        values = compute_metrics(lms[frame], exercise, sides[frame]) if bool(is_valid) else {}
        for metric in display_metrics:
            output[metric].append(_json_number(values.get(metric)))
    return output


def _position_values(frame_metrics: dict[str, list[float | None]], frame: int) -> dict[str, float | None]:
    total = len(next(iter(frame_metrics.values()), []))
    start = max(0, int(frame) - config.POSITION_WINDOW_RADIUS)
    end = min(total, int(frame) + config.POSITION_WINDOW_RADIUS + 1)
    values: dict[str, float | None] = {}
    for metric, series in frame_metrics.items():
        samples = [value for value in series[start:end] if value is not None]
        values[metric] = float(median(samples)) if len(samples) >= config.MIN_POSITION_VALUES else None
    return values


def _exclude_reason(detect: dict[str, float], exercise: str = "squat") -> str | None:
    minimum = float(config.MIN_VALID_RATIO)
    for phase in config.PHASES:
        ratio = detect[phase]
        if ratio < minimum:
            return config.DETECTION_REASON_TEMPLATE.format(
                phase=config.phase_labels(exercise)[phase], ratio=ratio, minimum=minimum
            )
    return None


def build_metrics(pose: dict, segments: dict) -> dict:
    """Return the stage 2 metrics JSON for one video."""

    exercise = normalize_exercise_name(segments["exercise"])
    total = int(segments["total"])
    lms = np.asarray(pose["lms"])
    valid = np.asarray(pose["valid"], dtype=bool).copy()
    fps = float(segments["fps"])
    display_metrics = list(config.DISPLAY_METRICS[exercise])

    sides = _frame_sides(lms, valid, segments)
    frame_metrics = _frame_metrics(lms, valid, exercise, display_metrics, sides)
    # Quality uses the selected raw observations so EMA cannot conceal a joint
    # outside the image or raise its visibility across the acceptance boundary.
    plate = segments.get("source", "").startswith("plate")
    raw = np.asarray(pose.get("raw_lms", lms))
    frame_reasons = {metric: [] for metric in display_metrics}
    for frame in range(total):
        for metric in display_metrics:
            if not valid[frame]:
                reason = "포즈 미검출"
            elif plate:
                reason = observation_reason(raw[frame], metric, sides[frame],
                                            has_fingers=bool(pose.get("has_fingers", True)))
            else:
                reason = None
            if reason is None and frame_metrics[metric][frame] is None:
                reason = "지표 계산 불가"
            frame_reasons[metric].append(reason)
            if reason is not None:
                frame_metrics[metric][frame] = None
    phase_positions = {"down": "MID_DOWN", "bottom": "BOTTOM", "up": "MID_UP"}

    reps = []
    for source in segments["reps"]:
        phases = source["phases"]
        detect = {
            phase: _phase_detection(valid, int(phases[phase][0]), int(phases[phase][1]),
                                    source["positions"][phase_positions[phase]] if plate else None)
            for phase in config.PHASES
        }
        exclude_reason = _exclude_reason(detect, exercise)
        positions = {}
        for position in config.POSITIONS:
            frame = int(source["positions"][position])
            start = max(0, frame - config.POSITION_WINDOW_RADIUS)
            end = min(total, frame + config.POSITION_WINDOW_RADIUS + 1)
            counts = {metric: sum(value is not None for value in series[start:end])
                      for metric, series in frame_metrics.items()}
            positions[position] = {
                "frame": frame,
                "values": _position_values(frame_metrics, frame),
                "valid_frames": counts,
                "reasons": {metric: window_reason(frame_reasons[metric][start:end], counts[metric],
                                                  config.MIN_POSITION_VALUES)
                            for metric in display_metrics},
            }
        reps.append({
            "rep": int(source["rep"]),
            "detect": detect,
            "positions": positions,
            "tempo": _tempo(phases, fps),
            "excluded": exclude_reason is not None,
            "exclude_reason": exclude_reason,
        })

    return {
        "exercise": exercise,
        "fps": fps,
        "total": total,
        "frame_metrics": frame_metrics,
        "reps": reps,
    }


def measure_reps(pose: dict, segments: dict) -> dict:
    """Backward-compatible alias for earlier skeleton code."""

    return build_metrics(pose, segments)
