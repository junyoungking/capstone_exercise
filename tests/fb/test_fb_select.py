from copy import deepcopy

import pytest

from export_overlay import joints_for

from feedback.select import goal_joints, select_goals


def cell(where, metric, k, *, direction="+", status="ok", color="yellow"):
    diff = k * 10 if k is not None else None
    return {
        "where": where,
        "metric": metric,
        "status": status,
        "user": diff,
        "ref": 0.0,
        "diff": diff,
        "unit": "deg",
        "threshold": 10.0,
        "k": k,
        "dir": direction,
        "color": color,
        "range": [-10.0, 10.0],
    }


def tempo(metric, k, *, direction="+"):
    return {
        "metric": metric,
        "user": k,
        "ref": 0.0,
        "diff": k,
        "unit": "s",
        "threshold": 0.3,
        "k": k,
        "dir": direction,
        "color": "none" if k is None else "yellow" if k <= 1.5 else "red",
    }


def findings(*cells, tempo_items=(), excluded=False):
    return {
        "exercise": "squat",
        "ref_rep": 1,
        "thresholds": {},
        "tempo_thresholds": {"down_s": 0.3, "bottom_s": 0.2, "up_s": 0.3},
        "reps": [
            {
                "rep": 1,
                "excluded": excluded,
                "exclude_reason": "측정 부족" if excluded else None,
                "cells": list(cells),
                "tempo": list(tempo_items),
                "tempo_ratio": {"user": 1.2, "ref": 1.0},
            }
        ],
    }


def rep_goal(data):
    return select_goals(data)["reps"][0]["goal"]


@pytest.mark.parametrize("direction", ["+", "-"])
def test_depth_group_when_members_share_where_and_direction_consumes_singles(direction):
    goal = rep_goal(
        findings(
            cell("BOTTOM", "knee_angle", 1.2, direction=direction),
            cell("BOTTOM", "hip_angle", 1.6, direction=direction, color="red"),
            cell("LOCKOUT", "trunk_lean", 1.4, direction="+"),
        )
    )

    assert goal == {
        "type": "group",
        "key": "depth",
        "where": "BOTTOM",
        "metrics": ["knee_angle", "hip_angle"],
        "dir": direction,
        "k": 1.6,
        "color": "red",
        "evidence": [
            {"where": "BOTTOM", "metric": "hip_angle"},
            {"where": "BOTTOM", "metric": "knee_angle"},
        ],
    }


def test_depth_group_requires_same_direction_and_then_selects_single():
    goal = rep_goal(
        findings(
            cell("BOTTOM", "knee_angle", 1.2, direction="-"),
            cell("BOTTOM", "hip_angle", 1.6, direction="+", color="red"),
        )
    )

    assert goal["type"] == "single"
    assert goal["where"] == "BOTTOM"
    assert goal["metric"] == "hip_angle"
    assert goal["dir"] == "+"


def test_goal_tie_breaks_by_position_then_display_metric_order():
    position_goal = rep_goal(
        findings(
            cell("BOTTOM", "knee_angle", 1.4),
            cell("START", "trunk_lean", 1.4),
        )
    )
    metric_goal = rep_goal(
        findings(
            cell("MID_DOWN", "hip_angle", 1.4),
            cell("MID_DOWN", "knee_angle", 1.4),
        )
    )

    assert position_goal["where"] == "START"
    assert position_goal["metric"] == "trunk_lean"
    assert metric_goal["where"] == "MID_DOWN"
    assert metric_goal["metric"] == "knee_angle"


def test_none_when_only_advisory_non_candidate_or_k_at_threshold_or_null():
    goal = rep_goal(
        findings(
            cell("BOTTOM", "head_hip_line", 2.0, status="advisory", color="red"),
            cell("LOCKOUT", "trunk_lean", 1.0, color="none"),
            cell("START", "knee_angle", None),
        )
    )

    assert goal == {"type": "none"}


def test_evidence_is_goal_members_then_same_metric_other_positions_limited_to_three():
    goal = rep_goal(
        findings(
            cell("BOTTOM", "knee_angle", 2.0, direction="-", color="red"),
            cell("BOTTOM", "hip_angle", 1.8, direction="-", color="red"),
            cell("START", "knee_angle", 1.7, direction="-", color="red"),
            cell("MID_DOWN", "hip_angle", 1.6, direction="-", color="red"),
            cell("LOCKOUT", "knee_angle", 1.4, direction="-"),
        )
    )

    assert goal["type"] == "group"
    assert goal["evidence"] == [
        {"where": "BOTTOM", "metric": "knee_angle"},
        {"where": "BOTTOM", "metric": "hip_angle"},
        {"where": "START", "metric": "knee_angle"},
    ]


def test_single_evidence_uses_same_metric_other_positions_by_k_then_position():
    goal = rep_goal(
        findings(
            cell("BOTTOM", "trunk_lean", 1.9, color="red"),
            cell("MID_UP", "trunk_lean", 1.4),
            cell("START", "trunk_lean", 1.4),
            cell("LOCKOUT", "knee_angle", 1.8, color="red"),
        )
    )

    assert goal["type"] == "single"
    assert goal["metric"] == "trunk_lean"
    assert goal["evidence"] == [
        {"where": "BOTTOM", "metric": "trunk_lean"},
        {"where": "START", "metric": "trunk_lean"},
        {"where": "MID_UP", "metric": "trunk_lean"},
    ]


def test_tempo_goal_is_independent_from_posture_goal_and_ignores_k_at_threshold():
    data = findings(
        cell("BOTTOM", "knee_angle", 1.3),
        tempo_items=[
            tempo("down_s", 1.0),
            tempo("bottom_s", 1.8, direction="-"),
            tempo("up_s", None),
        ],
    )

    selected = select_goals(data)["reps"][0]
    assert selected["goal"]["metric"] == "knee_angle"
    assert selected["tempo_goal"]["metric"] == "bottom_s"
    assert selected["tempo_goal"]["dir"] == "-"

    no_tempo = select_goals(findings(tempo_items=[tempo("down_s", 1.0)]))["reps"][0]
    assert no_tempo["goal"] == {"type": "none"}
    assert no_tempo["tempo_goal"] is None


def test_excluded_rep_has_null_goals_and_input_is_not_mutated():
    data = findings(cell("BOTTOM", "knee_angle", 2.0), tempo_items=[tempo("down_s", 2.0)], excluded=True)
    before = deepcopy(data)
    selected = select_goals(data)

    assert data == before
    assert selected["reps"][0]["goal"] is None
    assert selected["reps"][0]["tempo_goal"] is None


def test_goal_joints_returns_sorted_union_without_adding_joints_to_goal_json():
    goal = rep_goal(
        findings(
            cell("BOTTOM", "knee_angle", 1.2, direction="-"),
            cell("BOTTOM", "hip_angle", 1.7, direction="-", color="red"),
        )
    )
    expected_left = sorted(set(joints_for("knee_angle", "left")) | set(joints_for("hip_angle", "left")))
    expected_right = sorted(set(joints_for("knee_angle", "right")) | set(joints_for("hip_angle", "right")))

    assert "joints" not in goal
    assert goal_joints(goal, "left") == expected_left
    assert goal_joints(goal, "right") == expected_right
    assert goal_joints({"type": "none"}, "left") == []
