"""영상 없이 손으로 만든 측정 값으로 비교 경계와 제외 규칙을 확인한다."""

from copy import deepcopy
import json

import pytest

from feedback import config
from feedback.compare import compare_reps, compare_values


def measured(exercise="squat", rep=1):
    return {"exercise": exercise, "reps": [{
        "rep": rep, "excluded": False, "exclude_reason": None,
        "detect": {phase: 1.0 for phase in config.PHASES},
        "positions": {position: {"frame": frame, "values": {
            metric: 1.0 for metric in config.DISPLAY_METRICS[exercise]}}
            for frame, position in enumerate(config.POSITIONS)},
        "tempo": {"down_s": 1.0, "bottom_s": 0.2, "up_s": 2.0, "ratio": 0.5},
    }]}


@pytest.mark.parametrize("diff,k,color", [(18, 1, "none"), (27, 1.5, "yellow"),
                                         (27.001, 27.001 / 18, "red"),
                                         (-27, 1.5, "yellow"), (0, 0, "none")])
def test_difference_k_color_and_direction_boundaries(diff, k, color):
    result = compare_values(diff, 0, 18, "deg")
    assert result["diff"] == diff
    assert result["k"] == pytest.approx(k)
    assert result["color"] == color
    assert result["dir"] == ("+" if diff > 0 else "-" if diff < 0 else "0")


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf")])
@pytest.mark.parametrize("role", ["user", "ref"])
def test_unavailable_nulls_all_numeric_fields_and_range(missing, role):
    user, ref = measured(), measured()
    (user if role == "user" else ref)["reps"][0]["positions"]["BOTTOM"]["values"]["knee_angle"] = missing
    findings = compare_reps(user, ref)
    cell = next(cell for cell in findings["reps"][0]["cells"]
                if cell["where"] == "BOTTOM" and cell["metric"] == "knee_angle")
    assert cell["status"] == "unavailable"
    assert all(cell[key] is None for key in ("user", "ref", "diff", "threshold", "k", "range", "dir"))
    assert cell["color"] == "none"
    json.dumps(findings, allow_nan=False)


def test_range_units_all_cells_reference_selection_and_input_preservation():
    user, ref = measured(), measured()
    alternate = deepcopy(ref["reps"][0])
    alternate["rep"] = 2
    alternate["positions"]["BOTTOM"]["values"]["foot_flatness"] = 0.12
    ref["reps"].append(alternate)
    user["reps"][0]["positions"]["BOTTOM"]["values"]["foot_flatness"] = 0.24
    before = deepcopy((user, ref))
    findings = compare_reps(user, ref, ref_rep=2)
    cells = findings["reps"][0]["cells"]
    assert len(cells) == len(config.POSITIONS) * len(config.DISPLAY_METRICS["squat"])
    cell = next(item for item in cells if item["where"] == "BOTTOM" and item["metric"] == "foot_flatness")
    assert cell["unit"] == "ratio"
    assert cell["diff"] == pytest.approx(0.12)
    assert cell["range"] == pytest.approx([0.04, 0.20])
    assert next(item for item in cells if item["metric"] == "knee_angle")["unit"] == "deg"
    assert findings["ref_rep"] == 2
    assert findings["thresholds"] == config.DELTA_THRESHOLDS["squat"]
    assert (user, ref) == before


def test_advisory_preserves_values_colors_and_zero_threshold_display_metric():
    user, ref = measured("benchpress"), measured("benchpress")
    user["reps"][0]["positions"]["BOTTOM"]["values"]["bench_line_diff"] = 1.4
    user["reps"][0]["positions"]["BOTTOM"]["values"]["bar_proxy_conf"] = 0.5
    cells = compare_reps(user, ref)["reps"][0]["cells"]
    advisory = next(item for item in cells if item["where"] == "BOTTOM" and item["metric"] == "bench_line_diff")
    assert advisory["status"] == "advisory"
    assert advisory["diff"] == pytest.approx(0.4)
    assert advisory["k"] == pytest.approx(2)
    assert advisory["color"] == "red"
    display = next(item for item in cells if item["where"] == "BOTTOM" and item["metric"] == "bar_proxy_conf")
    assert display["status"] == "advisory" and display["diff"] == -0.5
    assert display["k"] is None and display["color"] == "none"


@pytest.mark.parametrize("available,excluded", [(9, True), (10, False), (11, False)])
def test_fifty_percent_exclusion_boundary_includes_advisory(monkeypatch, available, excluded):
    names = ["knee_angle", "hip_angle", "trunk_lean", "foot_flatness"]
    monkeypatch.setattr(config, "DISPLAY_METRICS", {"squat": names})
    monkeypatch.setattr(config, "ADVISORY_METRICS", {"squat": {"foot_flatness"}})
    user, ref = measured(), measured()
    for index, (position, metric) in enumerate((position, metric) for position in config.POSITIONS for metric in names):
        if index >= available:
            user["reps"][0]["positions"][position]["values"][metric] = None
    result = compare_reps(user, ref)["reps"][0]
    assert result["excluded"] is excluded
    assert result["exclude_reason"] == ("측정 가능한 칸 45% (기준 50%)" if excluded else None)


def test_detection_exclusion_reason_has_priority_over_metric_exclusion():
    user, ref = measured(), measured()
    rep = user["reps"][0]
    rep["excluded"], rep["exclude_reason"] = True, "하강 구간 관절 검출 40% (기준 60%)"
    for item in rep["positions"].values():
        item["values"] = {}
    result = compare_reps(user, ref)["reps"][0]
    assert result["excluded"] is True
    assert result["exclude_reason"] == rep["exclude_reason"]


@pytest.mark.parametrize("cause", ["detection", "metrics"])
def test_reference_detection_or_metric_shortage_fails(cause):
    user, ref = measured(), measured()
    if cause == "detection":
        ref["reps"][0]["detect"]["up"] = 0.59
    else:
        for item in ref["reps"][0]["positions"].values():
            item["values"] = {}
    with pytest.raises(ValueError, match="레퍼런스 rep 검출 부족"):
        compare_reps(user, ref)


def test_tempo_uses_own_thresholds_and_keeps_ratio_for_display():
    user, ref = measured(), measured()
    user["reps"][0]["tempo"] = {"down_s": 1.3, "bottom_s": 0.5, "up_s": 1.7, "ratio": 1.3 / 1.7}
    rep = compare_reps(user, ref)["reps"][0]
    assert [item["metric"] for item in rep["tempo"]] == ["down_s", "bottom_s", "up_s"]
    assert [item["threshold"] for item in rep["tempo"]] == [0.3, 0.2, 0.3]
    assert all(item["unit"] == "s" for item in rep["tempo"])
    assert rep["tempo_ratio"] == {"user": 1.3 / 1.7, "ref": 0.5}
