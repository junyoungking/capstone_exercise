"""Synthetic, video-free checks for stage 5 report export."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from feedback import config
from feedback.compare import compare_reps
from feedback.compare import compare_values
import feedback.export as export_module
from feedback.export import build_report, export_report
from feedback.metrics import _position_values
import feedback.metrics as metrics_module
from feedback.select import select_goals
import run_demo


def _landmarks(total: int = 40, *, offset: float = 0.0) -> np.ndarray:
    lms = np.zeros((total, 33, 4), dtype=np.float32)
    lms[:, :, 3] = 1.0
    for frame in range(total):
        lms[frame, :, 0] = frame + offset
        lms[frame, :, 1] = (frame + offset) / 100
        lms[frame, 0, 0] = frame + offset
    return lms


def _pose(total: int = 40, *, offset: float = 0.0, invalid=()) -> dict:
    valid = np.ones(total, dtype=bool)
    for frame in invalid:
        valid[frame] = False
    return {"lms": _landmarks(total, offset=offset), "valid": valid, "fps": 30.0,
            "w": 640, "h": 480, "total": total, "cache_hit": True}


def _cli_pose() -> dict:
    pose = _pose(total=12)
    hip = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0])
    pose["lms"][:, 23:25, 1] = hip[:, None]
    pose["lms"][:, 31:33, 0] = 0.1
    return pose


def _segment(video: str | Path, *, total: int = 40, facing: str = "left", reps=None) -> dict:
    if reps is None:
        reps = [{
            "rep": 1,
            "phases": {"down": [5, 10], "bottom": [10, 12], "up": [12, 20]},
            "positions": {"START": 5, "MID_DOWN": 8, "BOTTOM": 11, "MID_UP": 15, "LOCKOUT": 19},
            "side": "left",
        }]
    return {"video": str(video), "fps": 30.0, "size": [640, 480], "total": total,
            "exercise": "squat", "view": "side", "facing": facing, "source": "labels",
            "reps": reps}


def _measured(*, total: int = 40, value: float = 10.0, positions=None) -> dict:
    names = config.DISPLAY_METRICS["squat"]
    if positions is None:
        positions = {"START": 5, "MID_DOWN": 8, "BOTTOM": 11, "MID_UP": 15, "LOCKOUT": 19}
    return {
        "exercise": "squat",
        "fps": 30.0,
        "total": total,
        "frame_metrics": {metric: [float(frame + value) for frame in range(total)] for metric in names},
        "reps": [{
            "rep": 1,
            "detect": {phase: 1.0 for phase in config.PHASES},
            "positions": {position: {"frame": frame,
                                      "values": {metric: float(frame + value) for metric in names}}
                          for position, frame in positions.items()},
            "tempo": {"down_s": 0.2, "bottom_s": 0.1, "up_s": 0.3, "ratio": 2 / 3},
            "excluded": False,
            "exclude_reason": None,
        }],
    }


def _template_script() -> str:
    template = Path("web/report_template.html").read_text(encoding="utf-8")
    match = re.search(r"<script>\s*(.*?)\s*</script>", template, flags=re.S)
    assert match is not None
    return match.group(1)


def _script_from_html(html: str) -> str:
    match = re.search(r"<script>\s*(.*?)\s*</script>", html, flags=re.S)
    assert match is not None
    return match.group(1)


def _injected_data(html: str) -> dict:
    match = re.search(r"const D = (.*?);\nconst clamp", html, flags=re.S)
    assert match is not None
    return json.loads(match.group(1))


def _assert_cell_projection(actual: dict, expected: dict) -> None:
    for key in ("where", "metric", "status", "user", "ref", "diff", "unit",
                "threshold", "k", "dir", "color", "range"):
        assert actual[key] == expected[key]
    assert actual["label"] == f"{config.POSITION_LABELS[expected['where']]} · {config.METRIC_LABELS_KO[expected['metric']]}"


def _assert_goal_projection(actual: dict | None, expected: dict | None, source_cells: list[dict]) -> None:
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    for key in ("type", "where", "metric", "metrics", "dir", "k", "color"):
        if key in expected:
            assert actual[key] == expected[key]
    assert "sentence" in actual and actual["sentence"]
    assert "frame" in actual and "ref_frame" in actual and "joints" in actual
    by_key = {(cell["where"], cell["metric"]): cell for cell in source_cells}
    assert len(actual["evidence"]) == len(expected.get("evidence", []))
    for evidence, reference in zip(actual["evidence"], expected.get("evidence", [])):
        _assert_cell_projection(evidence, by_key[(reference["where"], reference["metric"])])


def _compare_with_range(user, ref, threshold: float, unit: str) -> dict:
    cell = compare_values(user, ref, threshold, unit)
    cell["range"] = [cell["ref"] - cell["threshold"], cell["ref"] + cell["threshold"]] if cell["diff"] is not None else None
    return cell


def _cell(where: str, metric: str, *, user=130.0, ref=100.0, k=2.0, color="red", status="ok") -> dict:
    threshold = 15.0
    return {"where": where, "metric": metric, "status": status, "user": user, "ref": ref,
            "diff": user - ref, "unit": "deg", "threshold": threshold, "k": k,
            "dir": "+" if user > ref else "-", "color": color,
            "range": [ref - threshold, ref + threshold]}


def _findings(*, goal=None, excluded=False) -> dict:
    names = config.DISPLAY_METRICS["squat"]
    cells = [_cell(position, metric, k=0.5, color="none")
             for position in config.POSITIONS for metric in names]
    lookup = {(cell["where"], cell["metric"]): cell for cell in cells}
    lookup["BOTTOM", "knee_angle"].update(_cell("BOTTOM", "knee_angle", user=130, ref=100, k=2.0))
    lookup["BOTTOM", "hip_angle"].update(_cell("BOTTOM", "hip_angle", user=125, ref=100, k=1.7, color="red"))
    if goal is None:
        goal = {"type": "single", "where": "BOTTOM", "metric": "knee_angle",
                "dir": "+", "k": 2.0, "color": "red",
                "evidence": [{"where": "BOTTOM", "metric": "knee_angle"}]}
    if excluded:
        goal = None
    return {
        "exercise": "squat",
        "ref_rep": 1,
        "thresholds": deepcopy(config.DELTA_THRESHOLDS["squat"]),
        "tempo_thresholds": deepcopy(config.TEMPO_THRESHOLDS),
        "reps": [{
            "rep": 1,
            "excluded": excluded,
            "exclude_reason": "하강 구간 관절 검출 40% (기준 60%)" if excluded else None,
            "cells": cells,
            "tempo": [_cell("tempo", "down_s", user=0.8, ref=0.4, k=1.4, color="yellow")],
            "tempo_ratio": {"user": 1.0, "ref": 0.5},
            "goal": goal,
            "tempo_goal": None,
        }],
    }


def _node_report_math(payload: dict, body: str) -> dict:
    if subprocess.run(["node", "--version"], capture_output=True).returncode != 0:
        pytest.skip("node is not available")
    script = """
const vm = require("vm");
const payload = JSON.parse(%s);
const context = {console};
vm.createContext(context);
vm.runInContext(payload.source, context);
const M = context.ReportMath;
%s
""" % (json.dumps(json.dumps({"source": _template_script(), **payload})), body)
    completed = subprocess.run(["node"], input=script, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


@pytest.fixture
def fake_sprite(monkeypatch):
    calls = []

    def fake(path, frames, width, quality):
        calls.append({"path": Path(path), "frames": list(frames), "width": width, "quality": quality})
        return {"cols": 1, "rows": len(frames), "cell": [width, 2], "pos": {str(f): [0, i] for i, f in enumerate(frames)},
                "data": "data:image/jpeg;base64,test"}

    monkeypatch.setattr(export_module, "build_sprite", fake)
    return calls


def test_report_preserves_findings_enriches_descriptions_and_uses_relative_paths(tmp_path, fake_sprite):
    user_video = tmp_path / "video space 사용자.mp4"
    ref_video = tmp_path / "ref space.mp4"
    user_video.touch()
    ref_video.touch()
    results = {"user": {"pose": _pose(invalid=[4]), "segments": _segment(user_video)},
               "ref": {"pose": _pose(offset=100), "segments": _segment(ref_video, facing="right")}}
    measured = {"user": _measured(), "ref": _measured(value=0)}
    findings = _findings()

    report = build_report(results, measured, findings, tmp_path / "out", ref_width=123, ref_quality=45)

    assert report["findings"] == findings
    assert report["user"]["video"] == "../video space 사용자.mp4"
    assert report["ref"]["video"] == "../ref space.mp4"
    assert (tmp_path / "out" / report["user"]["video"]).resolve() == user_video.resolve()
    assert report["ref"]["mirror"] is True
    assert report["ref"]["rep"] == 1
    assert report["ref"]["reps"] == [results["ref"]["segments"]["reps"][0]]
    assert fake_sprite[0]["frames"] == list(range(0, 35))
    assert fake_sprite[0]["width"] == 123 and fake_sprite[0]["quality"] == 45
    assert "4" not in report["user"]["frames"]
    assert "5" in report["user"]["frames"]
    assert report["user"]["frame_metrics"] == measured["user"]["frame_metrics"]
    assert report["reps"][0]["goal"]["sentence"].endswith(".")
    assert report["reps"][0]["goal"]["frame"] == 11
    assert report["reps"][0]["goal"]["ref_frame"] == 11
    assert report["reps"][0]["goal"]["joints"] == [23, 25, 27]
    assert report["reps"][0]["goal"]["evidence"][0]["label"] == "바닥 · 무릎각"
    assert report["reps"][0]["cells"][0]["label"] == "시작 · 무릎각"
    assert "screen" not in report["meta"]
    assert "side_ids" not in report
    assert report["meta"]["rules"]["k_yellow_max"] == config.K_YELLOW_MAX
    assert report["meta"]["side_ids"] == {"left": {"hip": 23, "ankle": 27},
                                          "right": {"hip": 24, "ankle": 28}}
    assert report["meta"]["advisory_metrics"] == sorted(config.ADVISORY_METRICS.get("squat", set()))


def test_group_goal_excluded_rep_and_clipped_ref_sprite_window(tmp_path, fake_sprite):
    video = tmp_path / "clip.mp4"
    video.touch()
    ref_reps = [
        {"rep": 1, "phases": {"down": [0, 2], "bottom": [2, 3], "up": [3, 5]},
         "positions": {"START": 0, "MID_DOWN": 1, "BOTTOM": 2, "MID_UP": 3, "LOCKOUT": 4}, "side": "left"},
        {"rep": 2, "phases": {"down": [20, 22], "bottom": [22, 23], "up": [23, 39]},
         "positions": {"START": 20, "MID_DOWN": 21, "BOTTOM": 22, "MID_UP": 30, "LOCKOUT": 38}, "side": "left"},
    ]
    group = {"type": "group", "where": "BOTTOM", "metrics": ["knee_angle", "hip_angle"],
             "dir": "+", "k": 2.0, "color": "red",
             "evidence": [{"where": "BOTTOM", "metric": "knee_angle"},
                          {"where": "BOTTOM", "metric": "hip_angle"}]}
    findings = _findings(goal=group)
    findings["ref_rep"] = 2
    results = {"user": {"pose": _pose(), "segments": _segment(video)},
               "ref": {"pose": _pose(), "segments": _segment(video, reps=ref_reps)}}
    measured = {"user": _measured(), "ref": _measured()}

    report = build_report(results, measured, findings, tmp_path / "out")

    assert report["ref"]["rep"] == 2
    assert report["ref"]["reps"] == [ref_reps[1]]
    assert fake_sprite[0]["frames"] == list(range(5, 40))
    assert report["reps"][0]["goal"]["joints"] == [11, 23, 25, 27]
    assert [item["label"] for item in report["reps"][0]["goal"]["evidence"]] == ["바닥 · 무릎각", "바닥 · 엉덩이각"]

    excluded = _findings(excluded=True)
    excluded_report = build_report(results, measured, excluded, tmp_path / "out")
    assert excluded_report["reps"][0]["goal"] is None
    assert excluded_report["reps"][0]["exclude_sentence"] == "이 rep은 비교에서 뺐습니다 — 하강 구간 관절 검출 40% (기준 60%)."
    assert excluded_report["reps"][0]["tempo"]["goal"] is None


def test_export_report_injects_escaped_json_without_fetch_or_network(tmp_path, monkeypatch, fake_sprite):
    template = tmp_path / "template.html"
    template.write_text("<html><body><script>const D = /*__DATA__*/null;</script></body></html>", encoding="utf-8")
    monkeypatch.setattr(export_module, "TEMPLATE_PATH", template)
    video = tmp_path / "video.mp4"
    video.touch()
    findings = _findings()
    findings["reps"][0]["cells"][0]["label_probe"] = "</script><script>alert(1)</script>\u2028"
    results = {"user": {"pose": _pose(), "segments": _segment(video)},
               "ref": {"pose": _pose(), "segments": _segment(video)}}
    measured = {"user": _measured(), "ref": _measured()}

    report = export_report(results, measured, findings, tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    html = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")

    assert saved == report
    assert saved["findings"] == findings
    assert "fetch(" not in html
    assert "https://" not in html and "http://" not in html
    assert "</script><script>alert" not in html
    assert "\\u003c/script\\u003e" in html
    assert "\\u2028" in html


def test_actual_template_has_embedded_data_no_fetch_network_or_modules_and_roundtrips(tmp_path, fake_sprite):
    user_video = tmp_path / "user 실제.mp4"
    ref_video = tmp_path / "ref 실제.mp4"
    user_video.touch()
    ref_video.touch()
    results = {"user": {"pose": _pose(), "segments": _segment(user_video)},
               "ref": {"pose": _pose(), "segments": _segment(ref_video)}}
    findings = _findings()
    first = findings["reps"][0]["cells"][0]
    first.update({"user": None, "ref": None, "diff": None, "threshold": None,
                  "k": None, "dir": None, "range": None, "status": "unavailable",
                  "color": "none"})
    findings["reps"][0]["cells"][1]["status"] = "advisory"
    findings["reps"][0]["cells"][1]["color"] = "yellow"
    findings["reps"][0]["tempo_goal"] = deepcopy(findings["reps"][0]["tempo"][0])
    measured = {"user": _measured(), "ref": _measured(value=0)}

    report = export_report(results, measured, findings, tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    html = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    injected = _injected_data(html)

    assert saved == report == injected
    assert saved["findings"] == findings
    assert "/*__DATA__*/null" not in html
    assert "fetch(" not in html
    assert "https://" not in html and "http://" not in html
    assert 'type="module"' not in html and "type='module'" not in html
    for expected, actual in zip(findings["reps"][0]["cells"], report["reps"][0]["cells"]):
        _assert_cell_projection(actual, expected)
    _assert_goal_projection(report["reps"][0]["goal"], findings["reps"][0]["goal"], findings["reps"][0]["cells"])
    assert report["reps"][0]["tempo"]["items"] == findings["reps"][0]["tempo"]
    assert report["reps"][0]["tempo"]["ratio"] == findings["reps"][0]["tempo_ratio"]
    assert report["reps"][0]["tempo"]["goal"] == findings["reps"][0]["tempo_goal"]


def test_report_none_goal_keeps_sentence_and_original_goal_projection(tmp_path, fake_sprite):
    video = tmp_path / "none.mp4"
    video.touch()
    findings = _findings(goal={"type": "none"})
    results = {"user": {"pose": _pose(), "segments": _segment(video)},
               "ref": {"pose": _pose(), "segments": _segment(video)}}
    measured = {"user": _measured(), "ref": _measured()}

    report = build_report(results, measured, findings, tmp_path / "out")

    assert report["reps"][0]["goal"]["type"] == "none"
    assert report["reps"][0]["goal"]["sentence"] == "모든 포지션에서 레퍼런스와의 차이가 임계값 안에 있습니다."
    assert report["reps"][0]["goal"]["frame"] is None
    assert report["reps"][0]["goal"]["ref_frame"] is None
    assert report["reps"][0]["goal"]["joints"] == []


def test_template_report_math_matches_python_compare_and_frame_mapping():
    payload = {
        "cases": [
            {"user": 118, "ref": 100, "threshold": 18, "unit": "deg", "advisory": False},
            {"user": 127, "ref": 100, "threshold": 18, "unit": "deg", "advisory": False},
            {"user": 127.18, "ref": 100, "threshold": 18, "unit": "deg", "advisory": False},
            {"user": 73, "ref": 100, "threshold": 18, "unit": "deg", "advisory": False},
            {"user": 0.5, "ref": 1.0, "threshold": 0, "unit": "deg", "advisory": True},
            {"user": None, "ref": 100, "threshold": 18, "unit": "deg", "advisory": False},
            {"user": 100, "ref": None, "threshold": 18, "unit": "deg", "advisory": False},
        ],
        "rules": {"k_none_max": config.K_NONE_MAX, "k_yellow_max": config.K_YELLOW_MAX},
    }
    result = _node_report_math(payload, """
const names = ["START", "MID_DOWN", "BOTTOM", "MID_UP", "LOCKOUT"];
const userRep = {positions:{START:10,MID_DOWN:20,BOTTOM:30,MID_UP:50,LOCKOUT:70}, phases:{down:[10,30], bottom:[30,35], up:[35,71]}};
const refRep = {positions:{START:100,MID_DOWN:130,BOTTOM:150,MID_UP:180,LOCKOUT:210}};
const out = {
  compare: payload.cases.map(c => M.compareCell(c.user, c.ref, c.threshold, c.unit, c.advisory, payload.rules)),
  progress: [
    M.progressAt(10, userRep.positions, names),
    M.progressAt(25, userRep.positions, names),
    M.progressAt(60, userRep.positions, names)
  ],
  frameAt: [
    M.frameAt(0, refRep.positions, names),
    M.frameAt(1.5, refRep.positions, names),
    M.frameAt(4, refRep.positions, names)
  ],
  refSync: [
    M.referenceFrame(5, userRep, refRep, names, 30, 60, true),
    M.referenceFrame(25, userRep, refRep, names, 30, 60, true),
    M.referenceFrame(72, userRep, refRep, names, 30, 60, true)
  ],
  refFree: [
    M.referenceFrame(10, userRep, refRep, names, 30, 60, false),
    M.referenceFrame(15, userRep, refRep, names, 30, 60, false),
    M.referenceFrame(80, userRep, refRep, names, 30, 60, false)
  ]
};
process.stdout.write(JSON.stringify(out));
""")

    expected = []
    for case in payload["cases"]:
        cell = compare_values(case["user"], case["ref"], case["threshold"], case["unit"])
        available = cell["diff"] is not None
        cell["status"] = "advisory" if available and case["advisory"] else "ok" if available else "unavailable"
        cell["range"] = [cell["ref"] - cell["threshold"], cell["ref"] + cell["threshold"]] if available else None
        expected.append(cell)
    for actual, expected_cell in zip(result["compare"], expected):
        assert actual["status"] == expected_cell["status"]
        for key in ("user", "ref", "diff", "unit", "threshold", "k", "dir", "color", "range"):
            if isinstance(expected_cell[key], list):
                assert actual[key] == pytest.approx(expected_cell[key])
            else:
                assert actual[key] == expected_cell[key]
    assert result["progress"] == [0, 1.5, 3.5]
    assert result["frameAt"] == [100, 140, 210]
    assert result["refSync"] == [100, 140, 100]
    assert result["refFree"] == [100, 110, 210]


def test_template_position_median_matches_python_edges_even_and_invalid_values():
    frame_metrics = {
        "edge": [9.0, 5.0, 7.0, None, 20.0],
        "middle": [None, 9.0, 3.0, 4.0, 100.0, None, 12.0],
        "short": [None, 1.0, None, None, 4.0],
        "even": [None, 1.0, 3.0, 5.0, 7.0, None],
    }
    rules = {
        "position_window_radius": config.POSITION_WINDOW_RADIUS,
        "min_position_values": config.MIN_POSITION_VALUES,
    }
    result = _node_report_math({"frame_metrics": frame_metrics, "rules": rules}, """
const out = {};
for (const [name, series] of Object.entries(payload.frame_metrics)) {
  out[name] = M.positionMedian(series, name === "edge" ? 0 : name === "even" ? 2 : 3, payload.rules);
}
process.stdout.write(JSON.stringify(out));
""")

    expected = {
        "edge": _position_values({"edge": frame_metrics["edge"]}, 0)["edge"],
        "middle": _position_values({"middle": frame_metrics["middle"]}, 3)["middle"],
        "short": _position_values({"short": frame_metrics["short"]}, 3)["short"],
    }
    assert result["edge"] == expected["edge"]
    assert result["middle"] == expected["middle"]
    assert result["short"] == expected["short"]
    assert result["even"] == 4.0


def _edit_report(tmp_path: Path) -> tuple[dict, dict]:
    names = config.DISPLAY_METRICS["squat"]
    positions1 = {"START": 5, "MID_DOWN": 8, "BOTTOM": 11, "MID_UP": 15, "LOCKOUT": 19}
    positions2 = {"START": 21, "MID_DOWN": 24, "BOTTOM": 27, "MID_UP": 30, "LOCKOUT": 34}
    user_segments = _segment(tmp_path / "user.mp4", reps=[
        {"rep": 1, "phases": {"down": [5, 10], "bottom": [10, 12], "up": [12, 20]},
         "positions": dict(positions1), "side": "left"},
        {"rep": 2, "phases": {"down": [21, 26], "bottom": [26, 28], "up": [28, 35]},
         "positions": dict(positions2), "side": "left"},
    ])
    ref_segments = _segment(tmp_path / "ref.mp4", reps=[
        {"rep": 1, "phases": {"down": [5, 10], "bottom": [10, 12], "up": [12, 20]},
         "positions": dict(positions1), "side": "left"},
    ])
    user_metrics = _measured(positions=positions1)
    user_metrics["reps"].append(deepcopy(user_metrics["reps"][0]))
    user_metrics["reps"][1]["rep"] = 2
    user_metrics["reps"][1]["positions"] = {
        position: {"frame": frame, "values": {metric: float(frame + 20) for metric in names}}
        for position, frame in positions2.items()
    }
    for metric in names:
        user_metrics["frame_metrics"][metric] = [float(frame + (0 if metric == "knee_angle" else 100)) for frame in range(40)]
    ref_metrics = _measured(value=10, positions=positions1)
    for metric in names:
        ref_metrics["frame_metrics"][metric] = [float(frame + 10 + (0 if metric == "knee_angle" else 100)) for frame in range(40)]
    findings = select_goals(compare_reps(user_metrics, ref_metrics, ref_rep=1))
    report = {
        "meta": {
            "positions": list(config.POSITIONS),
            "rules": {
                "position_window_radius": config.POSITION_WINDOW_RADIUS,
                "min_position_values": config.MIN_POSITION_VALUES,
                "frame_padding": config.FRAME_PADDING,
                "k_none_max": config.K_NONE_MAX,
                "k_yellow_max": config.K_YELLOW_MAX,
                "ratio_percent_scale": config.RATIO_PERCENT_SCALE,
            },
            "thresholds": dict(config.DELTA_THRESHOLDS["squat"]),
            "advisory_metrics": sorted(config.ADVISORY_METRICS.get("squat", set())),
        },
        "user": {**user_segments, "frame_metrics": user_metrics["frame_metrics"]},
        "ref": {**ref_segments, "frame_metrics": ref_metrics["frame_metrics"]},
        "reps": deepcopy(findings["reps"]),
    }
    for finding in report["reps"]:
        user_segment = next(rep for rep in report["user"]["reps"] if rep["rep"] == finding["rep"])
        ref_segment = report["ref"]["reps"][0]
        finding["tab_label"] = f"rep {finding['rep']}"
        if finding["goal"]:
            finding["goal"]["sentence"] = f"sentence {finding['rep']}"
            if finding["goal"]["type"] != "none":
                finding["goal"]["frame"] = user_segment["positions"][finding["goal"]["where"]]
                finding["goal"]["ref_frame"] = ref_segment["positions"][finding["goal"]["where"]]
        finding["tempo"] = {
            "items": finding["tempo"],
            "ratio": finding["tempo_ratio"],
            "goal": finding["tempo_goal"],
            "sentence": None,
        }
    return report, deepcopy(report)


def _cell_lookup(rep: dict) -> dict[tuple[str, str], dict]:
    return {(cell["where"], cell["metric"]): cell for cell in rep["cells"]}


def test_apply_position_edit_user_recalculates_changed_column_preserves_text_and_changes_sparse(tmp_path):
    report, original = _edit_report(tmp_path)
    report["reps"][0]["tab_label"] = "fixed label"
    report["reps"][0]["goal"] = {
        "type": "single", "where": "BOTTOM", "metric": "knee_angle", "dir": "-",
        "k": 1.0, "color": "none", "sentence": "fixed sentence",
        "frame": 11, "ref_frame": 11,
        "evidence": [{"where": "BOTTOM", "metric": "knee_angle"}],
    }
    untouched = deepcopy(_cell_lookup(report["reps"][0])[("START", "knee_angle")])
    result = _node_report_math({"report": report, "original": original}, """
const report = payload.report, original = payload.original;
const beforeSentence = report.reps[0].goal.sentence, beforeLabel = report.reps[0].tab_label;
const edit = M.applyPositionEdit(report, "user", 1, "BOTTOM", 13);
const changes = M.positionChanges(report, original);
process.stdout.write(JSON.stringify({edit, changes, report, beforeSentence, beforeLabel}));
""")

    assert result["edit"] == {"ok": True}
    assert result["changes"] == {"user": {"1": {"BOTTOM": 13}}, "ref": {}}
    edited = result["report"]["reps"][0]
    assert edited["goal"]["sentence"] == result["beforeSentence"]
    assert edited["tab_label"] == result["beforeLabel"]
    assert _cell_lookup(edited)[("START", "knee_angle")] == untouched
    expected_user = _position_values({"knee_angle": report["user"]["frame_metrics"]["knee_angle"]}, 13)["knee_angle"]
    expected_ref = _position_values({"knee_angle": report["ref"]["frame_metrics"]["knee_angle"]}, 11)["knee_angle"]
    expected = _compare_with_range(expected_user, expected_ref, config.DELTA_THRESHOLDS["squat"]["knee_angle"], "deg")
    cell = _cell_lookup(edited)[("BOTTOM", "knee_angle")]
    for key in ("user", "ref", "diff", "threshold", "k", "dir", "color", "range"):
        if isinstance(expected[key], list):
            assert cell[key] == pytest.approx(expected[key])
        else:
            assert cell[key] == expected[key]
    assert edited["goal"]["frame"] == 13
    assert edited["goal"]["evidence"][0] == cell


def test_apply_position_edit_rejects_bounds_and_order_without_mutating(tmp_path):
    report, original = _edit_report(tmp_path)
    result = _node_report_math({"report": report, "original": original}, """
const report = payload.report, original = payload.original;
const before = JSON.stringify(report);
const bound = M.applyPositionEdit(report, "user", 1, "START", 36);
const order = M.applyPositionEdit(report, "user", 1, "BOTTOM", 8);
process.stdout.write(JSON.stringify({bound, order, same: before === JSON.stringify(report), changes: M.positionChanges(report, original)}));
""")

    assert result["bound"]["ok"] is False
    assert "수정 범위" in result["bound"]["reason"]
    assert result["order"]["ok"] is False
    assert "포지션 순서" in result["order"]["reason"]
    assert result["same"] is True
    assert result["changes"] == {"user": {}, "ref": {}}


def test_apply_position_edit_ref_updates_same_column_for_multiple_user_reps(tmp_path):
    report, original = _edit_report(tmp_path)
    before_rep1_start = deepcopy(_cell_lookup(report["reps"][0])[("START", "knee_angle")])
    before_rep2_start = deepcopy(_cell_lookup(report["reps"][1])[("START", "knee_angle")])
    result = _node_report_math({"report": report, "original": original}, """
const report = payload.report, original = payload.original;
const edit = M.applyPositionEdit(report, "ref", 1, "BOTTOM", 13);
const changes = M.positionChanges(report, original);
process.stdout.write(JSON.stringify({edit, changes, report}));
""")

    assert result["edit"] == {"ok": True}
    assert result["changes"] == {"user": {}, "ref": {"1": {"BOTTOM": 13}}}
    for index, user_frame in enumerate((11, 27)):
        cell = _cell_lookup(result["report"]["reps"][index])[("BOTTOM", "knee_angle")]
        expected_user = _position_values({"knee_angle": report["user"]["frame_metrics"]["knee_angle"]}, user_frame)["knee_angle"]
        expected_ref = _position_values({"knee_angle": report["ref"]["frame_metrics"]["knee_angle"]}, 13)["knee_angle"]
        expected = _compare_with_range(expected_user, expected_ref, config.DELTA_THRESHOLDS["squat"]["knee_angle"], "deg")
        assert cell["user"] == expected["user"]
        assert cell["ref"] == expected["ref"]
        assert cell["diff"] == expected["diff"]
        assert cell["k"] == expected["k"]
    assert _cell_lookup(result["report"]["reps"][0])[("START", "knee_angle")] == before_rep1_start
    assert _cell_lookup(result["report"]["reps"][1])[("START", "knee_angle")] == before_rep2_start


def test_export_failure_escapes_reason_roundtrips_payload_and_template_branch(tmp_path):
    command = ["python", "run_demo.py", "--config", str(tmp_path / "demo with space.json")]
    reason = "invalid frame 999 </script><script>alert(1)</script>"
    path = export_module.export_failure(reason, tmp_path / "out", command)
    html = path.read_text(encoding="utf-8")
    injected = _injected_data(html)

    assert path.name == "failure.html"
    assert injected["failure"]["reason"] == reason
    assert "demo with space.json" in injected["failure"]["command"]
    assert "fetch(" not in html
    assert "https://" not in html and "http://" not in html
    assert "</script><script>alert" not in html
    assert "\\u003c/script\\u003e" in html

    result = _node_report_math({"html_script": _script_from_html(html)}, """
let captured = "";
const app = {};
Object.defineProperty(app, "innerHTML", {set(value){captured = value;}, get(){return captured;}});
const doc = {title:"", getElementById(id){return id === "app" ? app : {textContent:""};}};
const failureContext = {console, document: doc};
vm.createContext(failureContext);
vm.runInContext(payload.html_script, failureContext);
process.stdout.write(JSON.stringify({title: doc.title, html: captured}));
""")
    assert result["title"] == "분석할 수 없습니다"
    assert "분석할 수 없습니다" in result["html"]
    assert "invalid frame 999" in result["html"]
    assert "<script>alert" not in result["html"]
    assert "python" in result["html"] and "run_demo.py" in result["html"]


def test_cli_report_error_writes_failure_html_with_numeric_reason(tmp_path, monkeypatch):
    from feedback import gate, pose

    monkeypatch.setattr(gate, "check_inputs", lambda *args, **kwargs: {"warnings": []})

    video = tmp_path / "synthetic.mp4"
    video.touch()
    rep = {"index": 1, "phases": {"down": [0, 5], "bottom": [5, 7], "up": [7, 50]}}
    settings = {"exercise": "squat",
                "user": {"video": str(video), "view": "side", "reps": [rep]},
                "ref": {"video": str(video), "view": "side", "reps": [rep], "rep": 1}}
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr(pose, "extract_pose", lambda *args, **kwargs: _cli_pose())
    out = tmp_path / "out"

    with pytest.raises(SystemExit) as error:
        run_demo.main(["--config", str(config_path), "--out", str(out)])

    assert error.value.code == 2
    html = (out / "failure.html").read_text(encoding="utf-8")
    failure = _injected_data(html)["failure"]
    assert "50" in failure["reason"] and "12" in failure["reason"]
    assert "python" in failure["command"] and "run_demo.py" in failure["command"]


def test_cli_report_reruns_metrics_after_position_override(tmp_path, monkeypatch):
    from feedback import gate, pose

    monkeypatch.setattr(gate, "check_inputs", lambda *args, **kwargs: {"warnings": []})

    video = tmp_path / "synthetic.mp4"
    video.touch()
    rep = {"index": 1, "phases": {"down": [0, 5], "bottom": [5, 7], "up": [7, 12]}}
    settings = {"exercise": "squat",
                "user": {"video": str(video), "view": "side", "reps": [rep]},
                "ref": {"video": str(video), "view": "side", "reps": [rep], "rep": 1}}
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr(pose, "extract_pose", lambda *args, **kwargs: _cli_pose())

    def fake_compute(lms, _exercise, _side):
        frame_value = float(lms[0, 0])
        return {metric: frame_value for metric in config.DISPLAY_METRICS["squat"]}

    monkeypatch.setattr(metrics_module, "compute_metrics", fake_compute)
    monkeypatch.setattr(export_module, "build_sprite", lambda *args, **kwargs: {"pos": {}, "data": "data:"})
    template = tmp_path / "template.html"
    template.write_text("<script>window.REPORT=/*__DATA__*/null;</script>", encoding="utf-8")
    monkeypatch.setattr(export_module, "TEMPLATE_PATH", template)

    out = tmp_path / "out"
    assert run_demo.main(["--config", str(config_path), "--out", str(out)]) == 0
    first = json.loads((out / "metrics_user.json").read_text(encoding="utf-8"))
    assert first["reps"][0]["positions"]["BOTTOM"]["frame"] == 6
    assert first["reps"][0]["positions"]["BOTTOM"]["values"]["knee_angle"] == 6.0

    override_path = tmp_path / "override.json"
    override_path.write_text(json.dumps({"user": {"1": {"BOTTOM": 8}},
                                         "ref": {"1": {"BOTTOM": 8}}}), encoding="utf-8")
    assert run_demo.main(["--config", str(config_path), "--out", str(out),
                          "--override", str(override_path)]) == 0
    second = json.loads((out / "metrics_user.json").read_text(encoding="utf-8"))
    second_ref = json.loads((out / "metrics_ref.json").read_text(encoding="utf-8"))
    findings = json.loads((out / "findings.json").read_text(encoding="utf-8"))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert second["reps"][0]["positions"]["BOTTOM"]["frame"] == 8
    assert second["reps"][0]["positions"]["BOTTOM"]["values"]["knee_angle"] == 8.0
    assert second_ref["reps"][0]["positions"]["BOTTOM"]["frame"] == 8
    assert second_ref["reps"][0]["positions"]["BOTTOM"]["values"]["knee_angle"] == 8.0
    cell = next(item for item in findings["reps"][0]["cells"]
                if item["where"] == "BOTTOM" and item["metric"] == "knee_angle")
    assert cell["user"] == 8.0 and cell["ref"] == 8.0 and cell["diff"] == 0.0
    assert report["user"]["reps"][0]["positions"]["BOTTOM"] == 8
    assert report["ref"]["reps"][0]["positions"]["BOTTOM"] == 8
