"""Stage 7 input checks and user-facing failure reasons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from feedback import config
from tools import check_video


class GateFailure(ValueError):
    """A pipeline-stopping input failure with a readable reason."""

    def __init__(self, check: int, reason: str, remedy: str | None = None):
        super().__init__(reason)
        self.check = int(check)
        self.reason = reason
        self.remedy = remedy or config.GATE_REMEDIES.get(self.check, config.FAILURE_DEFAULT_REMEDY)


def _role_label(role: str) -> str:
    return config.ROLE_LABELS.get(role, role)


def _view_label(view: str | None) -> str:
    if view is None:
        return config.GATE_TEXT["no_view"]
    return config.VIEW_LABELS.get(str(view), str(view))


def _path_for(settings: dict[str, Any], role: str) -> Path:
    try:
        path = Path(settings[role]["video"])
    except KeyError as exc:
        raise GateFailure(1, config.GATE_TEXT["setting_missing"].format(key=f"{role}.video")) from exc
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def _rate(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _probe_missing(reason: str) -> bool:
    lowered = reason.lower()
    return "ffprobe" in lowered and ("없" in reason or "not found" in lowered)


def _video_report(settings: dict[str, Any], role: str) -> tuple[dict[str, Any], list[str]]:
    label = _role_label(role)
    video = _path_for(settings, role)
    cv_info = check_video.opencv_info(video)
    if "_error" in cv_info:
        raise GateFailure(
            1,
            config.GATE_TEXT["video_open"].format(role=label, video=video, reason=cv_info["_error"]),
        )
    actual = int(cv_info.get("actual") or 0)
    if actual < 1:
        raise GateFailure(1, config.GATE_TEXT["video_empty"].format(role=label, actual=actual))

    warnings: list[str] = []
    probe = check_video.ffprobe(video)
    if "_error" in probe:
        reason = str(probe["_error"])
        if _probe_missing(reason):
            warnings.append(config.GATE_TEXT["probe_missing"].format(role=label))
            probe = {}
        else:
            raise GateFailure(1, config.GATE_TEXT["probe_failed"].format(role=label, reason=reason))

    codec = str(probe.get("codec") or "").lower()
    if codec in check_video.BROWSER_NO:
        raise GateFailure(1, config.GATE_TEXT["codec"].format(role=label, codec=codec))
    if codec and codec not in check_video.BROWSER_OK:
        warnings.append(config.GATE_TEXT["codec_warning"].format(role=label, codec=codec))

    pix_fmt = str(probe.get("pix_fmt") or "")
    profile = str(probe.get("profile") or "")
    if any(marker in pix_fmt for marker in config.VIDEO_TEN_BIT_MARKERS) or any(
        marker in profile for marker in config.VIDEO_TEN_BIT_PROFILES
    ):
        raise GateFailure(1, config.GATE_TEXT["ten_bit"].format(role=label, pix_fmt=pix_fmt, profile=profile))

    r_fps = _rate(probe.get("r_fps"))
    avg_fps = _rate(probe.get("avg_fps"))
    if r_fps and avg_fps and (probe.get("variable_fps") or abs(r_fps - avg_fps) > config.VIDEO_FPS_TOLERANCE):
        raise GateFailure(
            1,
            config.GATE_TEXT["variable_fps"].format(
                role=label, rate=r_fps, average=avg_fps, maximum=config.VIDEO_FPS_TOLERANCE
            ),
        )

    rotation = int(probe.get("rotation") or 0)
    if rotation:
        raise GateFailure(1, config.GATE_TEXT["rotation"].format(role=label, rotation=rotation))

    header_count = int(cv_info.get("count") or 0)
    if abs(header_count - actual) > config.VIDEO_FRAME_COUNT_TOLERANCE:
        raise GateFailure(
            1,
            config.GATE_TEXT["frame_count"].format(
                role=label, count=header_count, actual=actual, maximum=config.VIDEO_FRAME_COUNT_TOLERANCE
            ),
        )

    probe_size = (probe.get("w"), probe.get("h"))
    cv_size = (cv_info.get("w"), cv_info.get("h"))
    if all(probe_size) and probe_size != cv_size:
        raise GateFailure(
            1,
            config.GATE_TEXT["size"].format(
                role=label, probe_w=probe_size[0], probe_h=probe_size[1], cv_w=cv_size[0], cv_h=cv_size[1]
            ),
        )

    return {"video": video.as_posix(), "opencv": cv_info, "ffprobe": probe}, warnings


def check_inputs(settings: dict[str, Any], mode: str = "labels") -> dict[str, Any]:
    """Run pre-pose checks and return warning metadata for the report."""

    warnings: list[str] = []
    videos = {}
    for role in ("user", "ref"):
        videos[role], role_warnings = _video_report(settings, role)
        warnings.extend(role_warnings)
    if mode == "labels":
        exercise_warning = check_exercises(settings)
        if exercise_warning:
            warnings.append(exercise_warning)
        check_views(settings)
    return {"warnings": warnings, "videos": videos}


def _exercise_for(settings: dict[str, Any], role: str) -> str | None:
    role_settings = settings.get(role, {})
    value = role_settings.get("exercise", settings.get("exercise"))
    return None if value is None else str(value)


def check_exercises(settings: dict[str, Any] | str, ref_exercise: str | None = None) -> str | None:
    from unified_feedback_v4 import normalize_exercise_name

    if isinstance(settings, dict):
        user = _exercise_for(settings, "user")
        ref = _exercise_for(settings, "ref")
    else:
        user = None if settings is None else str(settings)
        ref = None if ref_exercise is None else str(ref_exercise)
    if user is None or ref is None:
        return None
    user_norm = normalize_exercise_name(user)
    ref_norm = normalize_exercise_name(ref)
    if user_norm != ref_norm:
        raise GateFailure(2, config.GATE_TEXT["exercise"].format(user=user_norm, ref=ref_norm))
    return None


def check_views(settings: dict[str, Any] | str, ref_view: str | None = None) -> None:
    if isinstance(settings, dict):
        views = {role: settings.get(role, {}).get("view") for role in ("user", "ref")}
    else:
        views = {"user": settings, "ref": ref_view}
    for role, view in views.items():
        if view not in config.VIEW_LABELS:
            raise GateFailure(3, config.GATE_TEXT["unknown_view"].format(role=_role_label(role), view=view))
    if views["user"] != views["ref"]:
        raise GateFailure(
            3,
            config.GATE_TEXT["view"].format(user=_view_label(views["user"]), ref=_view_label(views["ref"])),
        )


def check_detection(role: str, pose: dict[str, Any]) -> dict[str, Any]:
    valid = np.asarray(pose.get("valid", []), dtype=bool)
    total = int(pose.get("total", len(valid)))
    detected = int(np.count_nonzero(valid[:total]))
    ratio = detected / total if total > 0 else 0.0
    if ratio < config.MIN_VALID_RATIO:
        raise GateFailure(
            4,
            config.GATE_TEXT["detection"].format(
                role=_role_label(role), detected=detected, total=total, ratio=ratio, minimum=config.MIN_VALID_RATIO
            ),
        )
    return {"detected": detected, "total": total, "ratio": ratio}


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(number)


def _reference_available_ratio(rep: dict[str, Any], exercise: str) -> float:
    metrics = config.DISPLAY_METRICS[exercise]
    values = [
        rep["positions"][position]["values"].get(metric)
        for position in config.POSITIONS
        for metric in metrics
    ]
    return sum(_finite(value) for value in values) / len(values)


def check_reference(ref_metrics: dict[str, Any], ref_rep: int = 1) -> dict[str, Any]:
    reps = list(ref_metrics.get("reps", []))
    matches = [rep for rep in reps if rep.get("rep") == ref_rep]
    if len(matches) != 1:
        raise GateFailure(
            5,
            config.GATE_TEXT["ref_missing"].format(rep=ref_rep, matches=len(matches), total=len(reps)),
        )
    reference = matches[0]
    exercise = ref_metrics["exercise"]
    reason = reference.get("exclude_reason") or config.REF_EXCLUDE_REASON
    if reference.get("excluded") or any(value < config.MIN_VALID_RATIO for value in reference.get("detect", {}).values()):
        raise GateFailure(5, config.GATE_TEXT["ref_excluded"].format(rep=ref_rep, reason=reason))
    available = _reference_available_ratio(reference, exercise)
    if available < config.MIN_REP_METRIC_RATIO:
        metric_reason = config.METRIC_REASON_TEMPLATE.format(ratio=available, minimum=config.MIN_REP_METRIC_RATIO)
        raise GateFailure(5, config.GATE_TEXT["ref_excluded"].format(rep=ref_rep, reason=metric_reason))
    return reference


def check_comparable(
    findings: dict[str, Any], ref_metrics: dict[str, Any] | None = None, ref_rep: int = 1
) -> dict[str, int]:
    if ref_metrics is not None:
        check_reference(ref_metrics, ref_rep)
    reps = list(findings.get("reps", []))
    comparable = sum(1 for rep in reps if not rep.get("excluded"))
    if comparable < 1:
        raise GateFailure(5, config.GATE_TEXT["user_excluded"].format(comparable=comparable, total=len(reps)))
    return {"comparable": comparable, "total": len(reps)}


def _torso_ratio(lms: np.ndarray) -> float | None:
    from unified_feedback_v4 import torso_len

    shoulder_width = abs(float(lms[11, 0]) - float(lms[12, 0]))
    torso = float(torso_len(lms))
    if not np.isfinite(shoulder_width) or not np.isfinite(torso) or torso <= 0:
        return None
    return shoulder_width / torso


def classify_view(ratio: float) -> str:
    if ratio < config.VIEW_SIDE_MAX:
        return "side"
    if ratio > config.VIEW_FRONT_MIN:
        return "front"
    return "diagonal"


def observe_angle(role: str, pose: dict[str, Any], segments: dict[str, Any]) -> dict[str, Any]:
    lms = np.asarray(pose.get("lms", []))
    valid = np.asarray(pose.get("valid", []), dtype=bool)
    samples = []
    for rep in segments.get("reps", []):
        frame = rep.get("positions", {}).get("START")
        if type(frame) is not int or frame < 0 or frame >= len(valid) or not bool(valid[frame]):
            continue
        ratio = _torso_ratio(lms[frame])
        if ratio is None:
            continue
        samples.append({"rep": rep.get("rep"), "frame": frame, "ratio": ratio, "view": classify_view(ratio)})
    configured = segments.get("view")
    if not samples:
        return {
            "ratio": None,
            "view": None,
            "samples": [],
            "warning": config.GATE_TEXT["angle_missing"].format(role=_role_label(role)),
        }
    ratio = float(np.median([sample["ratio"] for sample in samples]))
    view = classify_view(ratio)
    warning = None
    if configured in config.VIEW_LABELS and configured != view:
        warning = config.GATE_TEXT["angle_warning"].format(
            role=_role_label(role), ratio=ratio, actual=_view_label(view), configured=_view_label(configured)
        )
    return {"ratio": ratio, "view": view, "samples": samples, "warning": warning}


def check_label_ranges(role: str, video_settings: dict[str, Any], total: int) -> None:
    label = _role_label(role)
    last = max(0, int(total) - 1)
    for rep_index, rep in enumerate(video_settings.get("reps", []), start=1):
        rep_number = rep.get("index", rep_index)
        phases = rep.get("phases", {})
        previous_end = 0
        for phase in config.PHASES:
            span = phases.get(phase)
            phase_label = config.PHASE_LABELS[phase]
            if span is None:
                raise GateFailure(1, config.GATE_TEXT["phase_missing"].format(role=label, rep=rep_number, phase=phase_label))
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or any(type(frame) is not int for frame in span)
            ):
                raise GateFailure(1, config.GATE_TEXT["phase_shape"].format(role=label, rep=rep_number, phase=phase_label))
            start, end = span
            if start < 0 or start >= total:
                raise GateFailure(
                    1,
                    config.GATE_TEXT["phase_start"].format(
                        role=label, rep=rep_number, phase=phase_label, start=start, last=last
                    ),
                )
            if end < 1 or end > total:
                raise GateFailure(
                    1,
                    config.GATE_TEXT["phase_end"].format(
                        role=label, rep=rep_number, phase=phase_label, end=end, total=total
                    ),
                )
            if start >= end:
                raise GateFailure(
                    1,
                    config.GATE_TEXT["phase_empty"].format(
                        role=label, rep=rep_number, phase=phase_label, start=start, end=end
                    ),
                )
            if start < previous_end:
                raise GateFailure(
                    1,
                    config.GATE_TEXT["phase_order"].format(
                        role=label, rep=rep_number, phase=phase_label, start=start, previous=previous_end
                    ),
                )
            previous_end = end


def _format_positions(positions: Any) -> str:
    if not isinstance(positions, dict):
        return str(positions)
    parts = []
    for position in config.POSITIONS:
        if position in positions:
            parts.append(f"{position} {positions[position]}")
    extras = [key for key in positions if key not in config.POSITIONS]
    parts.extend(f"{key} {positions[key]}" for key in extras)
    return ", ".join(parts)


def _phase_fallback_message(role: str, value: dict[str, Any]) -> str:
    label = _role_label(role)
    phase = value.get("phase")
    phase_label = config.PHASE_LABELS.get(phase, str(phase))
    span = value.get("span")
    rep = value.get("rep", 1)
    total = int(value.get("total") or 0)
    previous_end = int(value.get("previous_end") or 0)
    last = max(0, total - 1)
    if (
        not isinstance(span, (list, tuple))
        or len(span) != 2
        or any(type(frame) is not int for frame in span)
    ):
        return config.GATE_TEXT["phase_shape"].format(role=label, rep=rep, phase=phase_label)
    start, end = span
    if total and (start < 0 or start >= total):
        return config.GATE_TEXT["phase_start"].format(
            role=label, rep=rep, phase=phase_label, start=start, last=last
        )
    if total and (end < 1 or end > total):
        return config.GATE_TEXT["phase_end"].format(
            role=label, rep=rep, phase=phase_label, end=end, total=total
        )
    if start >= end:
        return config.GATE_TEXT["phase_empty"].format(
            role=label, rep=rep, phase=phase_label, start=start, end=end
        )
    if start < previous_end:
        return config.GATE_TEXT["phase_order"].format(
            role=label, rep=rep, phase=phase_label, start=start, previous=previous_end
        )
    return config.GATE_TEXT["invalid_data"].format(fields="phase, span")


def friendly_error(error: BaseException, role: str | None = None) -> str:
    if isinstance(error, GateFailure):
        return error.reason
    value = error.args[0] if getattr(error, "args", None) else None
    label = _role_label(role or "user")
    if isinstance(value, dict):
        if "phase" in value and "span" in value:
            return _phase_fallback_message(role or "user", value)
        if "positions" in value and "bounds" in value:
            lower, upper = value["bounds"]
            return config.GATE_TEXT["position_bounds"].format(
                role=label, lower=lower, last=upper - 1, positions=_format_positions(value["positions"])
            )
        if "positions" in value:
            return config.GATE_TEXT["position_order"].format(
                role=label, positions=_format_positions(value["positions"])
            )
        if "override" in value or "overrides" in value:
            return config.GATE_TEXT["override"].format(role=label)
        if "ref_rep" in value:
            return config.GATE_TEXT["ref_missing"].format(
                rep=value.get("ref_rep"), matches=value.get("matches", 0), total=value.get("total", 0)
            )
        if "user" in value and "ref" in value:
            return config.GATE_TEXT["exercise"].format(user=value["user"], ref=value["ref"])
        fields = ", ".join(str(key) for key in value.keys())
        return config.GATE_TEXT["invalid_data"].format(fields=fields)
    if isinstance(error, KeyError):
        return config.GATE_TEXT["setting_missing"].format(key=error.args[0])
    text = str(error)
    return text if text else config.FAILURE_DEFAULT_REMEDY
