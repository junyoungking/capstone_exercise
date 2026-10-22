"""Stage 5 report.json and injected local HTML export."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any

import numpy as np

from export_overlay import POSE_CONNECTIONS, SIDE_IDS, build_sprite, dump_frames
from feedback import config
from feedback.describe import describe_reps
from feedback.select import goal_joints


TEMPLATE_PATH = config.PROJECT_ROOT / "web" / "report_template.html"
INJECTION_MARKER = "/*__DATA__*/null"
POSE3D_SCRIPTS_MARKER = "<!--__POSE3D_SCRIPTS__-->"


def _safe_json(data: Any) -> str:
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _json_number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _resolve_video(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (config.PROJECT_ROOT / candidate).resolve()


def _relative_video(path: str | Path, out: Path) -> str:
    return Path(os.path.relpath(_resolve_video(path), out)).as_posix()


def _pose_sequence(pose: dict[str, Any]) -> list[Any]:
    lms = np.asarray(pose["lms"])
    valid = np.asarray(pose["valid"], dtype=bool)
    return [lms[index] if bool(valid[index]) else None for index in range(len(valid))]


def _clipped_span(rep: dict[str, Any], total: int) -> tuple[int, int]:
    start = int(rep["phases"]["down"][0]) - config.FRAME_PADDING
    end = int(rep["phases"]["up"][1]) + config.FRAME_PADDING
    return max(0, start), min(int(total), end)


def _rep_frames(rep: dict[str, Any], total: int) -> list[int]:
    start, end = _clipped_span(rep, total)
    return list(range(start, end))


def _metric_labels(exercise: str) -> dict[str, str]:
    return {metric: config.METRIC_LABELS_KO[metric] for metric in config.DISPLAY_METRICS[exercise]}


def _metric_joints(exercise: str) -> dict[str, dict[str, list[int]]]:
    return {
        metric: {side: goal_joints({"type": "single", "metric": metric}, side) for side in ("left", "right")}
        for metric in config.DISPLAY_METRICS[exercise]
    }


def _overlay_side_ids() -> dict[str, dict[str, int]]:
    return {
        side: {"hip": int(ids["hip"]), "ankle": int(ids["ankle"])}
        for side, ids in SIDE_IDS.items()
    }


def _description_map(findings: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["rep"]): item for item in describe_reps(findings)}


def _cell_label(cell: dict[str, Any], exercise: str = "squat") -> str:
    metric = config.METRIC_LABELS_KO.get(cell["metric"], cell["metric"])
    where = config.position_labels(exercise).get(cell["where"], cell["where"])
    return f"{where} · {metric}"


def _enriched_cells(cells: list[dict[str, Any]], exercise: str = "squat") -> list[dict[str, Any]]:
    enriched = []
    for cell in cells:
        item = deepcopy(cell)
        item["label"] = _cell_label(cell, exercise)
        enriched.append(item)
    return enriched


def _lookup_cells(cells: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(cell["where"], cell["metric"]): cell for cell in cells}


def _frame_for_goal(role_rep: dict[str, Any], goal: dict[str, Any] | None) -> int | None:
    if not goal or goal.get("type") == "none":
        return None
    return int(role_rep["positions"][goal["where"]])


def _enriched_goal(
    rep: dict[str, Any],
    user_segment: dict[str, Any],
    ref_segment: dict[str, Any],
    description: dict[str, Any],
    exercise: str = "squat",
) -> dict[str, Any] | None:
    goal = rep.get("goal")
    if goal is None:
        return None
    enriched = deepcopy(goal)
    enriched["sentence"] = description["goal_sentence"]
    enriched["frame"] = _frame_for_goal(user_segment, goal)
    enriched["ref_frame"] = _frame_for_goal(ref_segment, goal)
    enriched["joints"] = goal_joints(goal, user_segment.get("side", "left"))

    cells = _lookup_cells(rep.get("cells", []))
    evidence = []
    for reference in goal.get("evidence", []):
        cell = deepcopy(cells[(reference["where"], reference["metric"])])
        cell["label"] = _cell_label(cell, exercise)
        evidence.append(cell)
    enriched["evidence"] = evidence
    return enriched


def _enriched_reps(
    findings: dict[str, Any],
    user_segments: dict[str, Any],
    ref_segment: dict[str, Any],
) -> list[dict[str, Any]]:
    descriptions = _description_map(findings)
    by_user_rep = {int(rep["rep"]): rep for rep in user_segments["reps"]}
    reps = []
    for rep in findings["reps"]:
        rep_number = int(rep["rep"])
        description = descriptions[rep_number]
        if rep.get("excluded"):
            goal = None
            exclude_sentence = description["goal_sentence"]
        else:
            goal = _enriched_goal(rep, by_user_rep[rep_number], ref_segment, description, findings["exercise"])
            exclude_sentence = None
        reps.append({
            "rep": rep_number,
            "excluded": bool(rep.get("excluded")),
            "exclude_reason": rep.get("exclude_reason"),
            "exclude_sentence": exclude_sentence,
            "tab_label": description["tab_label"],
            "goal": goal,
            "cells": _enriched_cells(rep.get("cells", []), findings["exercise"]),
            "tempo": {
                "items": deepcopy(rep.get("tempo", [])),
                "ratio": deepcopy(rep.get("tempo_ratio", {})),
                "goal": deepcopy(rep.get("tempo_goal")),
                "sentence": description.get("tempo_sentence"),
            },
        })
    return reps


def _video_block(
    role: str,
    result: dict[str, Any],
    measured: dict[str, Any],
    out: Path,
    spans: list[tuple[int, int]],
) -> dict[str, Any]:
    segments = result["segments"]
    block = {
        "video": _relative_video(segments["video"], out),
        "fps": _json_number(segments["fps"]),
        "size": [int(segments["size"][0]), int(segments["size"][1])],
        "total": int(segments["total"]),
        "facing": segments.get("facing"),
        "frames": dump_frames(_pose_sequence(result["pose"]), spans),
        "frame_metrics": deepcopy(measured["frame_metrics"]),
        "reps": deepcopy(segments["reps"]),
    }
    if role == "ref":
        block["mirror"] = False
        block["rep"] = None
        block["sprite"] = None
    return block


def _selected_ref_segment(ref_segments: dict[str, Any], findings: dict[str, Any]) -> dict[str, Any]:
    ref_rep = int(findings["ref_rep"])
    matches = [rep for rep in ref_segments["reps"] if int(rep["rep"]) == ref_rep]
    if len(matches) != 1:
        raise ValueError({"ref_rep": ref_rep, "matches": len(matches)})
    return matches[0]


def build_report(
    results: dict[str, dict[str, Any]],
    measured: dict[str, dict[str, Any]],
    findings: dict[str, Any],
    out: Path,
    *,
    ref_width: int = config.DEFAULT_REF_WIDTH,
    ref_quality: int = config.DEFAULT_REF_QUALITY,
    warnings: list[str] | None = None,
    angles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the serializable report data used by report.html."""

    out = Path(out)
    exercise = findings["exercise"]
    user_segments = results["user"]["segments"]
    ref_segments = results["ref"]["segments"]
    ref_segment = _selected_ref_segment(ref_segments, findings)
    ref_frames = _rep_frames(ref_segment, int(ref_segments["total"]))

    user_spans = [_clipped_span(rep, int(user_segments["total"])) for rep in user_segments["reps"]]
    ref_span = _clipped_span(ref_segment, int(ref_segments["total"]))

    report = {
        "meta": {
            "exercise": exercise,
            "view": user_segments.get("view"),
            "threshold_note": "임계값은 임시값(v4)",
            "positions": list(config.POSITIONS),
            "position_labels": config.position_labels(exercise),
            "phase_labels": config.phase_labels(exercise),
            "metric_labels": _metric_labels(exercise),
            "thresholds": deepcopy(findings.get("thresholds", {})),
            "tempo_thresholds": deepcopy(findings.get("tempo_thresholds", {})),
            "warnings": list(warnings or []),
            "angles": deepcopy(angles or {}),
            "rules": {
                "position_window_radius": config.POSITION_WINDOW_RADIUS,
                "min_position_values": config.MIN_POSITION_VALUES,
                "frame_padding": config.FRAME_PADDING,
                "k_none_max": config.K_NONE_MAX,
                "k_yellow_max": config.K_YELLOW_MAX,
                "ratio_percent_scale": config.RATIO_PERCENT_SCALE,
            },
            "metrics": list(config.DISPLAY_METRICS[exercise]),
            "advisory_metrics": sorted(config.ADVISORY_METRICS.get(exercise, set())),
            "side_ids": _overlay_side_ids(),
            "pose3d": {
                "parents": list(config.POSE3D_PARENTS),
                "metric_joints": deepcopy(config.POSE3D_METRIC_JOINTS),
                "labels": dict(config.POSE3D_UI_LABELS),
                "roles": dict(config.ROLE_LABELS),
            },
        },
        "user": _video_block("user", results["user"], measured["user"], out, user_spans),
        "ref": _video_block("ref", results["ref"], measured["ref"], out, [ref_span]),
        "connections": [list(connection) for connection in POSE_CONNECTIONS],
        "metric_joints": _metric_joints(exercise),
        "findings": deepcopy(findings),
        "reps": _enriched_reps(findings, user_segments, ref_segment),
        "pose3d": {"user": "pose3d_user.js", "ref": "pose3d_ref.js"},
    }
    report["ref"]["mirror"] = report["user"].get("facing") != report["ref"].get("facing")
    report["ref"]["rep"] = int(findings["ref_rep"])
    report["ref"]["reps"] = [deepcopy(ref_segment)]
    report["ref"]["sprite"] = build_sprite(
        _resolve_video(ref_segments["video"]), ref_frames, int(ref_width), int(ref_quality)
    )
    return report


def export_report(
    results: dict[str, dict[str, Any]],
    measured: dict[str, dict[str, Any]],
    findings: dict[str, Any],
    out: Path,
    *,
    ref_width: int = config.DEFAULT_REF_WIDTH,
    ref_quality: int = config.DEFAULT_REF_QUALITY,
    warnings: list[str] | None = None,
    angles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write report.json and report.html, then return the report data."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = build_report(results, measured, findings, out, ref_width=ref_width, ref_quality=ref_quality,
                          warnings=warnings, angles=angles)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if INJECTION_MARKER not in template:
        raise ValueError({"template": str(TEMPLATE_PATH), "missing": INJECTION_MARKER})
    html = template.replace(INJECTION_MARKER, _safe_json(report), 1)
    scripts = "\n".join(
        f'<script src="{report["pose3d"][role]}"></script>'
        for role in ("user", "ref") if (out / report["pose3d"][role]).is_file()
    )
    html = html.replace(POSE3D_SCRIPTS_MARKER, scripts)
    vendor_source = config.PROJECT_ROOT / "web" / "vendor"
    vendor_out = out / "vendor"
    for name in ("three.min.js", "three.LICENSE.txt", "three.version.json"):
        source = vendor_source / name
        if source.is_file():
            vendor_out.mkdir(parents=True, exist_ok=True)
            target = vendor_out / name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
    (out / "report.html").write_text(html, encoding="utf-8")
    return report


def export_failure(reason: str, out: Path, command: list[str], *,
                   remedy: str | None = None, check: int | None = None) -> Path:
    """Render the chapter 6.7 screen using readable reasons and local data."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    text = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
    payload = {"failure": {"title": config.FAILURE_TITLE, "reason": reason, "command": text,
                           "remedy": remedy or config.FAILURE_DEFAULT_REMEDY,
                           "check": check, "check_label": config.FAILURE_CHECK_LABEL,
                           "check_count": len(config.GATE_REMEDIES),
                           "checklist": list(config.FAILURE_CHECKLIST)}}
    template = config.FAILURE_TEMPLATE_PATH.read_text(encoding="utf-8")
    if INJECTION_MARKER not in template:
        raise ValueError({"template": str(config.FAILURE_TEMPLATE_PATH), "missing": INJECTION_MARKER})
    path = out / "failure.html"
    path.write_text(template.replace(INJECTION_MARKER, _safe_json(payload), 1), encoding="utf-8")
    return path
