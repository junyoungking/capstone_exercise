"""수동 phase 라벨과 이미지 좌표의 엉덩이 높이로 포지션을 정한다."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from feedback import config


def smooth_hip_y(values: np.ndarray) -> np.ndarray:
    """양쪽 값이 있는 짧은 빈칸만 보간하고 중심 5프레임 이동평균을 낸다."""
    values = np.asarray(values, dtype=float).copy()
    known = np.flatnonzero(np.isfinite(values))
    for left, right in zip(known[:-1], known[1:]):
        gap = int(right - left - 1)
        if 0 < gap <= config.HIP_MAX_GAP_FRAMES:
            values[left + 1:right] = np.linspace(values[left], values[right], gap + 2)[1:-1]
    output = np.full(values.shape, np.nan)
    radius = config.HIP_SMOOTH_WINDOW // 2
    for frame in np.flatnonzero(np.isfinite(values)):
        window = values[max(0, frame - radius):min(len(values), frame + radius + 1)]
        output[frame] = np.mean(window[np.isfinite(window)])
    return output


def hip_y_trajectory(pose: dict) -> np.ndarray:
    from unified_feedback_v4 import SIDE_IDS, mid

    values = np.full(pose["total"], np.nan)
    for frame in np.flatnonzero(pose["valid"]):
        values[frame] = mid(pose["lms"][frame], SIDE_IDS["left"]["hip"], SIDE_IDS["right"]["hip"])[1]
    return smooth_hip_y(values)


def _validate_phases(phases: dict, total: int) -> dict:
    validated = {}
    previous_end = 0
    for phase in config.PHASES:
        span = phases[phase]
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or any(type(frame) is not int for frame in span)):
            raise ValueError({"phase": phase, "span": span})
        start, end = span
        if not previous_end <= start < end <= total:
            raise ValueError({"phase": phase, "span": span, "previous_end": previous_end, "total": total})
        validated[phase] = [start, end]
        previous_end = end
    return validated


def _validate_positions(positions: dict, total: int) -> None:
    frames = [positions[position] for position in config.POSITIONS]
    if any(type(frame) is not int or not 0 <= frame < total for frame in frames):
        raise ValueError({"positions": positions, "total": total})
    if any(left >= right for left, right in zip(frames[:-1], frames[1:])):
        raise ValueError({"positions": positions, "order": config.POSITIONS})


def positions_from_labels(hip_y: np.ndarray, phases: dict, changes: dict | None = None) -> dict:
    phases = _validate_phases(phases, len(hip_y))
    start = phases["down"][0]
    lockout = phases["up"][1] - 1
    bottom_start, bottom_end = phases["bottom"]
    bottom_values = hip_y[bottom_start:bottom_end]
    if np.isfinite(bottom_values).any():
        bottom = bottom_start + int(np.nanargmax(bottom_values))
    else:
        bottom = (bottom_start + bottom_end) // 2

    mid_down = sum(phases["down"]) // 2
    mid_up = sum(phases["up"]) // 2
    down_height = (hip_y[start] + hip_y[bottom]) / 2
    up_height = (hip_y[bottom] + hip_y[lockout]) / 2
    down_crossings = np.flatnonzero(hip_y[start:bottom + 1] >= down_height)
    up_crossings = np.flatnonzero(hip_y[bottom:lockout + 1] <= up_height)
    if down_crossings.size:
        mid_down = start + int(down_crossings[0])
    if up_crossings.size:
        mid_up = bottom + int(up_crossings[0])
    positions = dict(zip(config.POSITIONS, (start, mid_down, bottom, mid_up, lockout)))
    if changes is not None:
        return apply_override(positions, changes, phases, len(hip_y))
    _validate_positions(positions, len(hip_y))
    return positions


def apply_override(positions: dict, changes: dict, phases: dict, total: int) -> dict:
    if not isinstance(changes, dict) or set(changes) - set(config.POSITIONS):
        raise ValueError({"override": changes})
    updated = {**positions, **changes}
    _validate_positions(updated, total)
    lower = max(0, phases["down"][0] - config.FRAME_PADDING)
    upper = min(total, phases["up"][1] + config.FRAME_PADDING)
    if any(not lower <= frame < upper for frame in updated.values()):
        raise ValueError({"positions": updated, "bounds": [lower, upper]})
    return updated


def _rep_facing(sequence: list, start: int, side: str) -> str | None:
    from unified_feedback_v4 import SIDE_IDS

    ids = SIDE_IDS[side]
    differences = []
    for frame in sequence[start:start + config.FACING_SAMPLE_FRAMES]:
        if frame is not None:
            difference = frame[ids["foot"], 0] - frame[ids["heel"], 0]
            if np.isfinite(difference):
                differences.append(float(difference))
    if not differences:
        return None
    mean = float(np.mean(differences))
    if mean > 0:
        return "right"
    if mean < 0:
        return "left"
    return None


def build_segments(pose: dict, settings: dict, exercise: str, overrides: dict | None = None) -> dict:
    from export_overlay import dominant_side
    from unified_feedback_v4 import normalize_exercise_name

    exercise = normalize_exercise_name(exercise)
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or any(not isinstance(value, dict) for value in overrides.values()):
        raise ValueError({"overrides": overrides})
    labels = settings["reps"]
    indices = [label["index"] for label in labels]
    if any(type(index) is not int for index in indices) or len(set(indices)) != len(indices):
        raise ValueError({"indices": indices})
    if set(overrides) - {str(index) for index in indices}:
        raise ValueError({"overrides": overrides, "indices": indices})
    hip_y = hip_y_trajectory(pose)
    sequence = [frame if valid else None for frame, valid in zip(pose["lms"], pose["valid"])]
    reps = []
    facing_votes = []
    for label in labels:
        phases = _validate_phases(label["phases"], pose["total"])
        rep_key = str(label["index"])
        positions = positions_from_labels(hip_y, phases, overrides.get(rep_key))
        side = dominant_side(sequence, (phases["down"][0], phases["up"][1]), exercise)
        facing = _rep_facing(sequence, positions["START"], side)
        if facing is not None:
            facing_votes.append(facing)
        reps.append({"rep": label["index"], "phases": phases, "positions": positions, "side": side})
    votes = Counter(facing_votes)
    facing = votes.most_common(1)[0][0] if votes else None
    return {
        "video": Path(settings["video"]).as_posix(),
        "fps": pose["fps"],
        "size": [pose["w"], pose["h"]],
        "total": pose["total"],
        "exercise": exercise,
        "view": settings["view"],
        "facing": facing,
        "source": "labels+override" if overrides else "labels",
        "reps": reps,
    }


def _runs(values: list[int]) -> list[dict[str, int]]:
    if not values:
        return []
    runs = []
    start = 0
    current = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value != current:
            runs.append({"phase_id": current, "start": start, "end": index})
            start = index
            current = value
    runs.append({"phase_id": current, "start": start, "end": len(values)})
    return runs


def _finite_argmax(values: np.ndarray, offset: int) -> int | None:
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return None
    local = finite[int(np.nanargmax(values[finite]))]
    return offset + int(local)


def _median_torso_len(pose: dict, start: int, end: int) -> float:
    from unified_feedback_v4 import torso_len

    values = []
    for frame in range(max(0, start), min(end, pose["total"])):
        if pose["valid"][frame]:
            value = float(torso_len(pose["lms"][frame]))
            if np.isfinite(value):
                values.append(value)
    if not values:
        return float("nan")
    return float(np.median(values))


def hip_velocity(hip_y: np.ndarray) -> np.ndarray:
    """Central difference; video edges and unavailable neighbours stay unknown."""
    velocity = np.full(len(hip_y), np.nan)
    velocity[1:-1] = (hip_y[2:] - hip_y[:-2]) / 2
    return velocity


def _model_candidate(hip_y: np.ndarray, pose: dict, down: dict[str, int],
                     up: dict[str, int]) -> dict | None:
    window_start = max(0, int(down["start"]) - config.MODEL_CONTEXT_FRAMES)
    window_end = min(len(hip_y), int(up["end"]) + config.MODEL_CONTEXT_FRAMES)
    if window_start >= window_end:
        return None
    bottom = _finite_argmax(hip_y[window_start:window_end], window_start)
    if bottom is None:
        return None

    pre = hip_y[window_start:bottom + 1]
    post = hip_y[bottom:window_end]
    if not np.isfinite(pre).any() or not np.isfinite(post).any():
        return None
    pre_min = float(np.nanmin(pre))
    post_min = float(np.nanmin(post))
    bottom_y = float(hip_y[bottom])
    d1 = bottom_y - pre_min
    d2 = bottom_y - post_min
    amplitude = min(d1, d2)
    torso = _median_torso_len(pose, window_start, window_end)
    if not np.isfinite(amplitude) or not np.isfinite(torso) or torso <= 0:
        return None
    if amplitude / torso < config.MODEL_MIN_AMPLITUDE_RATIO:
        return None

    start_candidates = np.flatnonzero(
        np.isfinite(hip_y[window_start:bottom + 1])
        & (hip_y[window_start:bottom + 1] <= pre_min + config.MODEL_ENDPOINT_RATIO * d1)
    )
    if start_candidates.size == 0:
        return None
    start = window_start + int(start_candidates[-1])

    lockout_candidates = np.flatnonzero(
        np.isfinite(hip_y[bottom:window_end])
        & (hip_y[bottom:window_end] <= post_min + config.MODEL_ENDPOINT_RATIO * d2)
    )
    if lockout_candidates.size == 0:
        return None
    lockout = bottom + int(lockout_candidates[0])

    return {"window_start": window_start, "window_end": window_end, "bottom": bottom,
            "bottom_y": bottom_y, "pre_min": pre_min, "post_min": post_min,
            "d1": d1, "d2": d2, "amplitude": amplitude,
            "fallback_start": start, "fallback_lockout": lockout}


def _stationary_endpoint(hip_y: np.ndarray, velocity: np.ndarray, candidate: dict,
                         position: str, lower: int, upper: int,
                         speed_ratio: float) -> int | None:
    """Return the first stationary run encountered from BOTTOM, within [lower, upper)."""
    bottom = candidate["bottom"]
    is_start = position == "START"
    amplitude = candidate["d1" if is_start else "d2"]
    top = candidate["pre_min" if is_start else "post_min"]
    if is_start:
        frames = range(bottom - 1, lower - 1, -1)
    else:
        frames = range(bottom + 1, upper)
    eligible = (np.isfinite(hip_y) & np.isfinite(velocity)
                & (hip_y <= top + config.MODEL_TOP_BAND_RATIO * amplitude)
                & (np.abs(velocity) < speed_ratio * amplitude))
    consecutive = 0
    for frame in frames:
        consecutive = consecutive + 1 if eligible[frame] else 0
        if consecutive == config.MODEL_STOP_FRAMES:
            return (frame + config.MODEL_STOP_FRAMES - 1 if is_start
                    else frame - config.MODEL_STOP_FRAMES + 1)
    return None


def _select_endpoint(hip_y: np.ndarray, velocity: np.ndarray, candidate: dict,
                     position: str, lower: int, upper: int) -> tuple[int, str, float | None]:
    """Try each speed threshold over the whole search range before relaxing it."""
    for speed_ratio in config.MODEL_STOP_SPEED_RATIOS:
        frame = _stationary_endpoint(hip_y, velocity, candidate, position, lower, upper, speed_ratio)
        if frame is not None:
            return frame, config.MODEL_ENDPOINT_METHODS["velocity"], speed_ratio
    fallback = "fallback_start" if position == "START" else "fallback_lockout"
    return candidate[fallback], config.MODEL_ENDPOINT_METHODS["fallback"], None


def _model_positions_for_candidate(hip_y: np.ndarray, velocity: np.ndarray, candidate: dict,
                                   previous_lockout: int | None = None,
                                   next_start: int | None = None) -> tuple[dict, dict, dict, dict] | None:
    lower = max(0, candidate["window_start"] - config.MODEL_SEARCH_EXTRA_FRAMES)
    upper = min(len(hip_y), candidate["window_end"] + config.MODEL_SEARCH_EXTRA_FRAMES)
    if previous_lockout is not None:
        lower = max(lower, previous_lockout)
    if next_start is not None:
        upper = min(upper, next_start + 1)
    bottom = candidate["bottom"]
    if not lower < bottom < upper - 1:
        return None
    methods = {}
    speed_ratios = {}
    endpoints = {}
    for position in ("START", "LOCKOUT"):
        frame, methods[position], speed_ratios[position] = _select_endpoint(
            hip_y, velocity, candidate, position, lower, upper)
        if not lower <= frame < upper:
            return None
        endpoints[position] = frame
    start, lockout = endpoints["START"], endpoints["LOCKOUT"]
    bottom_y = candidate["bottom_y"]
    bottom_threshold = bottom_y - config.MODEL_BOTTOM_RATIO * candidate["amplitude"]
    bottom_start = bottom
    while bottom_start > start and np.isfinite(hip_y[bottom_start - 1]) and hip_y[bottom_start - 1] >= bottom_threshold:
        bottom_start -= 1
    bottom_end = bottom + 1
    while bottom_end <= lockout and np.isfinite(hip_y[bottom_end]) and hip_y[bottom_end] >= bottom_threshold:
        bottom_end += 1

    phases = {"down": [start, bottom_start], "bottom": [bottom_start, bottom_end], "up": [bottom_end, lockout + 1]}
    try:
        positions = positions_from_labels(hip_y, phases)
    except ValueError:
        return None
    return phases, positions, methods, speed_ratios


def build_model_segments(pose: dict, settings: dict, inference: dict[str, Any],
                         overrides: dict | None = None) -> dict:
    from export_overlay import dominant_side
    from unified_feedback_v4 import normalize_exercise_name

    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or any(not isinstance(value, dict) for value in overrides.values()):
        raise ValueError({"overrides": overrides})
    exercise = inference.get("exercise")
    if exercise is None:
        raise ValueError({"model_exercise": None})
    exercise = normalize_exercise_name(exercise)
    phase_ids = [int(value) for value in inference.get("clean_phase_ids", [])]
    if len(phase_ids) != pose["total"]:
        raise ValueError({"clean_phase_ids": len(phase_ids), "total": pose["total"]})

    hip_y = hip_y_trajectory(pose)
    velocity = hip_velocity(hip_y)
    sequence = [frame if valid else None for frame, valid in zip(pose["lms"], pose["valid"])]
    reps = []
    facing_votes = []
    runs = _runs(phase_ids)
    candidates = []
    for index, run in enumerate(runs[:-1]):
        next_run = runs[index + 1]
        if run["phase_id"] != 1 or next_run["phase_id"] != 2:
            continue
        candidate = _model_candidate(hip_y, pose, run, next_run)
        if candidate is None:
            continue
        candidates.append(candidate)

    # START look-ahead supplies a bound for the previous rep's LOCKOUT. Final
    # START searches also use the already-finalized previous LOCKOUT as a bound.
    provisional_starts = []
    for index, candidate in enumerate(candidates):
        lower = max(0, candidate["window_start"] - config.MODEL_SEARCH_EXTRA_FRAMES)
        if index:
            lower = max(lower, candidates[index - 1]["bottom"] + 1)
        start, _, _ = _select_endpoint(hip_y, velocity, candidate, "START", lower, candidate["bottom"])
        provisional_starts.append(start)
    for index, candidate in enumerate(candidates):
        previous_lockout = reps[-1]["positions"]["LOCKOUT"] if reps else None
        next_start = provisional_starts[index + 1] if index + 1 < len(candidates) else None
        selected = _model_positions_for_candidate(hip_y, velocity, candidate, previous_lockout, next_start)
        if selected is None:
            continue
        phases, positions, methods, speed_ratios = selected
        rep_index = len(reps) + 1
        rep_key = str(rep_index)
        if rep_key in overrides:
            positions = apply_override(positions, overrides[rep_key], phases, pose["total"])
            for position in methods:
                if position in overrides[rep_key]:
                    methods[position] = config.MODEL_ENDPOINT_METHODS["override"]
                    speed_ratios[position] = None
        side = dominant_side(sequence, (phases["down"][0], phases["up"][1]), exercise)
        facing = _rep_facing(sequence, positions["START"], side)
        if facing is not None:
            facing_votes.append(facing)
        reps.append({"rep": rep_index, "phases": phases, "positions": positions, "side": side,
                     "endpoint_methods": methods, "endpoint_speed_ratios": speed_ratios})

    if set(overrides) - {str(rep["rep"]) for rep in reps}:
        raise ValueError({"overrides": overrides, "indices": [rep["rep"] for rep in reps]})
    votes = Counter(facing_votes)
    facing = votes.most_common(1)[0][0] if votes else None
    return {
        "video": Path(settings["video"]).as_posix(),
        "fps": pose["fps"],
        "size": [pose["w"], pose["h"]],
        "total": pose["total"],
        "exercise": exercise,
        "view": settings["view"],
        "facing": facing,
        "source": "model+override" if overrides else "model",
        "reps": reps,
    }
