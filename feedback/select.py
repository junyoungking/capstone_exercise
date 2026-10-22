"""3단계: rep마다 목표 1개와 근거 선택."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from export_overlay import joints_for

from feedback import config


def _metric_order(exercise: str) -> dict[str, int]:
    return {name: index for index, name in enumerate(config.DISPLAY_METRICS[exercise])}


def _position_order() -> dict[str, int]:
    return {name: index for index, name in enumerate(config.POSITIONS)}


def _candidate_cells(rep: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        cell
        for cell in rep.get("cells", [])
        if cell.get("status") == "ok"
        and cell.get("k") is not None
        and float(cell["k"]) > config.K_NONE_MAX
    ]


def _cell_key(cell: dict[str, Any], metric_order: dict[str, int], position_order: dict[str, int]):
    return (
        -float(cell["k"]),
        position_order.get(cell.get("where"), len(position_order)),
        metric_order.get(cell.get("metric"), len(metric_order)),
    )


def _goal_key(goal: dict[str, Any], metric_order: dict[str, int], position_order: dict[str, int]):
    metrics = goal.get("metrics") or [goal.get("metric")]
    metric_rank = min(metric_order.get(metric, len(metric_order)) for metric in metrics)
    return (
        -float(goal["k"]),
        position_order.get(goal.get("where"), len(position_order)),
        metric_rank,
    )


def _evidence_refs(cells: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"where": cell["where"], "metric": cell["metric"]} for cell in cells[: config.MAX_EVIDENCE]]


def _evidence(
    goal: dict[str, Any],
    cells: list[dict[str, Any]],
    metric_order: dict[str, int],
    position_order: dict[str, int],
) -> list[dict[str, str]]:
    if goal["type"] == "group":
        metrics = set(goal["metrics"])
        goal_cells = [
            cell
            for cell in cells
            if cell.get("where") == goal["where"] and cell.get("metric") in metrics
        ]
        goal_cells.sort(key=lambda cell: (-float(cell["k"]), metric_order.get(cell["metric"], len(metric_order))))
        other_cells = [
            cell
            for cell in cells
            if cell.get("where") != goal["where"] and cell.get("metric") in metrics
        ]
    else:
        metrics = {goal["metric"]}
        goal_cells = [
            cell
            for cell in cells
            if cell.get("where") == goal["where"] and cell.get("metric") == goal["metric"]
        ]
        other_cells = [
            cell
            for cell in cells
            if cell.get("where") != goal["where"] and cell.get("metric") in metrics
        ]
    other_cells.sort(key=lambda cell: _cell_key(cell, metric_order, position_order))
    return _evidence_refs(goal_cells + other_cells)


def _group_candidate(
    key: str,
    group: dict[str, Any],
    by_place_metric: dict[tuple[str, str], dict[str, Any]],
    metric_order: dict[str, int],
) -> dict[str, Any] | None:
    where = group["where"]
    metrics = list(group["metrics"])
    members = [by_place_metric.get((where, metric)) for metric in metrics]
    if any(member is None for member in members):
        return None
    dirs = {member.get("dir") for member in members if member is not None}
    if len(dirs) != 1:
        return None
    max_member = sorted(
        members,
        key=lambda cell: (-float(cell["k"]), metric_order.get(cell["metric"], len(metric_order))),
    )[0]
    return {
        "type": "group",
        "key": key,
        "where": where,
        "metrics": metrics,
        "dir": max_member["dir"],
        "k": max_member["k"],
        "color": max_member["color"],
    }


def _single_candidate(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "single",
        "where": cell["where"],
        "metric": cell["metric"],
        "dir": cell["dir"],
        "k": cell["k"],
        "color": cell["color"],
    }


def _select_posture_goal(rep: dict[str, Any], exercise: str) -> dict[str, Any]:
    metric_order = _metric_order(exercise)
    position_order = _position_order()
    cells = _candidate_cells(rep)
    if not cells:
        return {"type": "none"}

    by_place_metric = {(cell["where"], cell["metric"]): cell for cell in cells}
    consumed: set[tuple[str, str]] = set()
    goals: list[dict[str, Any]] = []

    for key, group in (config.GROUPS.items() if exercise == "squat" else []):
        goal = _group_candidate(key, group, by_place_metric, metric_order)
        if goal is None:
            continue
        goals.append(goal)
        consumed.update((goal["where"], metric) for metric in goal["metrics"])

    for cell in cells:
        if (cell["where"], cell["metric"]) in consumed:
            continue
        goals.append(_single_candidate(cell))

    if not goals:
        return {"type": "none"}

    goal = deepcopy(sorted(goals, key=lambda item: _goal_key(item, metric_order, position_order))[0])
    goal["evidence"] = _evidence(goal, cells, metric_order, position_order)
    return goal


def _select_tempo_goal(rep: dict[str, Any]) -> dict[str, Any] | None:
    order = {name: index for index, name in enumerate(config.TEMPO_THRESHOLDS)}
    candidates = [
        item
        for item in rep.get("tempo", [])
        if item.get("k") is not None and float(item["k"]) > config.K_NONE_MAX
    ]
    if not candidates:
        return None
    return deepcopy(sorted(candidates, key=lambda item: (-float(item["k"]), order.get(item.get("metric"), len(order))))[0])


def select_goals(findings: dict[str, Any]) -> dict[str, Any]:
    """Return findings with per-rep posture and tempo goals added."""

    selected = deepcopy(findings)
    exercise = selected["exercise"]
    for rep in selected.get("reps", []):
        if rep.get("excluded"):
            rep["goal"] = None
            rep["tempo_goal"] = None
            continue
        rep["goal"] = _select_posture_goal(rep, exercise)
        rep["tempo_goal"] = _select_tempo_goal(rep)
    return selected


def goal_joints(goal: dict[str, Any] | None, side: str) -> list[int]:
    """Return the sorted union of joints used by a selected goal."""

    if not goal or goal.get("type") == "none":
        return []
    metrics = goal.get("metrics") or [goal["metric"]]
    joints: set[int] = set()
    for metric in metrics:
        joints.update(joints_for(metric, side))
    return sorted(joints)
