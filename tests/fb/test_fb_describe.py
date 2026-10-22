"""설계 문장과 정확히 일치하는지 고정 입력으로 확인한다."""

from copy import deepcopy
import json
import re

import numpy as np
import pytest

from feedback import config, pose
from feedback.describe import describe_reps
import run_demo


def findings(metric="knee_angle", diff=-22, where="BOTTOM"):
    direction = "+" if diff > 0 else "-"
    unit = "ratio" if metric in config.RATIO_METRICS else "deg"
    return {"exercise": "squat", "reps": [{
        "rep": 1, "excluded": False, "exclude_reason": None,
        "goal": {"type": "single", "where": where, "metric": metric, "dir": direction},
        "cells": [{"where": where, "metric": metric, "diff": diff, "unit": unit}],
        "tempo_goal": None, "tempo_ratio": {"user": 1.3, "ref": 1.0},
    }]}


def description(data):
    before = deepcopy(data)
    result = describe_reps(data)[0]
    assert data == before
    for field in ("goal_sentence", "tempo_sentence", "tab_label"):
        text = result[field]
        if text:
            assert not any(word in text for word in config.BANNED)
    return result


@pytest.mark.parametrize("metric,diff,expected,label", [
    ("knee_angle", 22, "바닥에서 무릎을 레퍼런스보다 22° 덜 굽혔습니다.", "바닥 · 무릎각 +22°"),
    ("knee_angle", -22, "바닥에서 무릎을 레퍼런스보다 22° 더 굽혔습니다.", "바닥 · 무릎각 −22°"),
    ("hip_angle", 19, "바닥에서 엉덩이를 레퍼런스보다 19° 덜 접었습니다.", "바닥 · 엉덩이각 +19°"),
    ("hip_angle", -19, "바닥에서 엉덩이를 레퍼런스보다 19° 더 접었습니다.", "바닥 · 엉덩이각 −19°"),
    ("trunk_lean", 12, "바닥에서 상체를 레퍼런스보다 12° 더 숙였습니다.", "바닥 · 몸통기울기 +12°"),
    ("trunk_lean", -12, "바닥에서 상체를 레퍼런스보다 12° 덜 숙였습니다.", "바닥 · 몸통기울기 −12°"),
    ("foot_flatness", 0.12, "바닥에서 발 앞뒤 높이 차이가 레퍼런스보다 몸통 길이의 12%만큼 큽니다.", "바닥 · 발바닥밀착 +12%"),
    ("foot_flatness", -0.12, "바닥에서 발 앞뒤 높이 차이가 레퍼런스보다 몸통 길이의 12%만큼 작습니다.", "바닥 · 발바닥밀착 −12%"),
    ("head_hip_line", 0.25, "바닥에서 머리와 엉덩이의 가로 거리가 레퍼런스보다 몸통 길이의 25%만큼 멉니다.", "바닥 · 머리-엉덩이 +25%"),
    ("head_hip_line", -0.25, "바닥에서 머리와 엉덩이의 가로 거리가 레퍼런스보다 몸통 길이의 25%만큼 가깝습니다.", "바닥 · 머리-엉덩이 −25%"),
])
def test_all_five_metric_directions_match_verbatim_and_have_number_unit(metric, diff, expected, label):
    result = description(findings(metric, diff))
    assert result["goal_sentence"] == expected
    assert result["tab_label"] == label
    assert re.search(r"\d+(?:°|%)", result["goal_sentence"])
    assert result["tempo_sentence"] is None


@pytest.mark.parametrize("direction,expected,label", [
    ("-", "바닥에서 레퍼런스보다 더 깊게 앉았습니다 (무릎 22°, 엉덩이 19°).", "바닥 · 더 깊게"),
    ("+", "바닥에서 레퍼런스보다 덜 깊게 앉았습니다 (무릎 22°, 엉덩이 19°).", "바닥 · 덜 깊게"),
])
def test_depth_group_both_directions_match_verbatim(direction, expected, label):
    data = findings()
    rep = data["reps"][0]
    rep["goal"] = {"type": "group", "key": "depth", "where": "BOTTOM",
                   "metrics": ["knee_angle", "hip_angle"], "dir": direction}
    sign = 1 if direction == "+" else -1
    rep["cells"] = [{"where": "BOTTOM", "metric": "knee_angle", "diff": sign * 22, "unit": "deg"},
                    {"where": "BOTTOM", "metric": "hip_angle", "diff": sign * 19, "unit": "deg"}]
    result = description(data)
    assert result["goal_sentence"] == expected
    assert result["tab_label"] == label
    assert re.search(r"\d+°", result["goal_sentence"])


@pytest.mark.parametrize("metric,diff,expected", [
    ("down_s", 0.4, "하강이 레퍼런스보다 0.4초 느립니다 (하강:상승 1.3:1, 레퍼런스 1.0:1)."),
    ("down_s", -0.4, "하강이 레퍼런스보다 0.4초 빠릅니다 (하강:상승 1.3:1, 레퍼런스 1.0:1)."),
    ("up_s", 0.4, "상승이 레퍼런스보다 0.4초 느립니다 (하강:상승 1.3:1, 레퍼런스 1.0:1)."),
    ("up_s", -0.4, "상승이 레퍼런스보다 0.4초 빠릅니다 (하강:상승 1.3:1, 레퍼런스 1.0:1)."),
    ("bottom_s", 0.3, "바닥에서 레퍼런스보다 0.3초 더 머물렀습니다."),
    ("bottom_s", -0.3, "바닥에서 레퍼런스보다 0.3초 덜 머물렀습니다."),
])
def test_all_three_tempo_directions_match_verbatim_and_have_number_unit(metric, diff, expected):
    data = findings()
    data["reps"][0]["tempo_goal"] = {"metric": metric, "diff": diff, "dir": "+" if diff > 0 else "-"}
    sentence = description(data)["tempo_sentence"]
    assert sentence == expected
    assert re.search(r"\d+\.\d초", sentence)


def test_none_sentence_label_and_tempo_are_independent():
    data = findings()
    data["reps"][0]["goal"] = {"type": "none"}
    data["reps"][0]["tempo_goal"] = {"metric": "bottom_s", "diff": 0.3, "dir": "+"}
    result = description(data)
    assert result["goal_sentence"] == "모든 포지션에서 레퍼런스와의 차이가 임계값 안에 있습니다."
    assert result["tab_label"] == "차이 없음"
    assert result["tempo_sentence"] == "바닥에서 레퍼런스보다 0.3초 더 머물렀습니다."


def test_excluded_sentence_label_and_null_tempo():
    data = findings()
    data["reps"][0].update(excluded=True, exclude_reason="하강 구간 관절 검출 40% (기준 60%)",
                           goal=None, tempo_goal=None)
    result = description(data)
    assert result["goal_sentence"] == "이 rep은 비교에서 뺐습니다 — 하강 구간 관절 검출 40% (기준 60%)."
    assert result["tab_label"] == "제외"
    assert result["tempo_sentence"] is None


@pytest.mark.parametrize("where,label", [("START", "시작"), ("MID_DOWN", "하강 중간"),
                                       ("BOTTOM", "바닥"), ("MID_UP", "상승 중간"), ("LOCKOUT", "락아웃")])
def test_all_position_labels_and_python_rounding(where, label):
    result = description(findings("knee_angle", -22.5, where))
    assert result["goal_sentence"] == f"{label}에서 무릎을 레퍼런스보다 22° 더 굽혔습니다."
    assert result["tab_label"] == f"{label} · 무릎각 −22°"


def test_cli_synthetic_pipeline_generates_json_and_prints_descriptions(tmp_path, monkeypatch, capsys):
    from feedback import gate

    monkeypatch.setattr(gate, "check_inputs", lambda *args, **kwargs: {"warnings": []})
    lms = np.zeros((21, 33, 4), dtype=np.float32)
    lms[:, :, 3] = 1
    lms[:, 23:25, 1] = np.concatenate((np.arange(11), np.arange(9, -1, -1)))[:, None] / 10
    lms[:, 31:33, 0] = 0.1
    synthetic = {"lms": lms, "valid": np.ones(21, dtype=bool), "fps": 30.0,
                 "w": 640, "h": 480, "total": 21, "cache_hit": True}
    monkeypatch.setattr(pose, "extract_pose", lambda *args, **kwargs: synthetic)
    rep = {"index": 1, "phases": {"down": [0, 9], "bottom": [9, 12], "up": [12, 21]}}
    settings = {"exercise": "squat", "user": {"video": "user.mp4", "view": "side", "reps": [rep]},
                "ref": {"video": "ref.mp4", "view": "side", "reps": [rep], "rep": 1}}
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(settings), encoding="utf-8")
    out = tmp_path / "out"
    assert run_demo.main(["--config", str(config_path), "--out", str(out), "--until", "findings"]) == 0
    result = json.loads((out / "findings.json").read_text(encoding="utf-8"))
    assert result["reps"][0]["goal"] == {"type": "none"}
    assert result["reps"][0]["tempo_goal"] is None
    for role in ("user", "ref"):
        values = json.loads((out / f"metrics_{role}.json").read_text(encoding="utf-8"))
        assert values["total"] == 21
    output = capsys.readouterr().out
    assert "[metrics]" in output and "[compare: diff/k/color]" in output
    assert "tab_label=차이 없음" in output
    assert "goal_sentence=모든 포지션에서 레퍼런스와의 차이가 임계값 안에 있습니다." in output
    assert "tempo_sentence=null" in output
