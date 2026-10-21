"""0단계 뼈대 검증. 실제 영상이나 포즈 추출을 사용하지 않는다."""

import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from feedback import config
import run_demo


def test_feedback_modules_import_without_running_the_pipeline():
    for name in ("pose", "segment", "metrics", "compare", "select", "describe",
                 "export", "gate", "stgcn_infer"):
        importlib.import_module(f"feedback.{name}")


def test_reused_settings_are_the_original_objects():
    import offline_compare
    import unified_feedback_v4

    for name in ("DISPLAY_METRICS", "DELTA_THRESHOLDS", "METRIC_LABELS_KO", "ADVISORY_METRICS"):
        assert getattr(config, name) is getattr(unified_feedback_v4, name)
    for name in ("RATIO_METRICS", "MIN_VALID_RATIO", "MIN_REP_METRIC_RATIO"):
        assert getattr(config, name) is getattr(offline_compare, name)


def test_cli_defaults_and_path_types():
    args = run_demo.build_parser().parse_args([])
    assert args.config == Path("data/demo.json")
    assert args.out == Path("out/demo")
    assert args.until == "report"
    assert args.segment is None
    assert args.override is None
    assert args.ref_width == 300
    assert args.ref_quality == 72
    assert args.no_cache is False


def test_cli_accepts_all_pipeline_options():
    args = run_demo.build_parser().parse_args([
        "--config", "data/example.json", "--out", "out/example",
        "--until", "segments", "--segment", "model",
        "--override", "positions_override.json", "--ref-width", "400",
        "--ref-quality", "80", "--no-cache",
    ])
    assert args.config == Path("data/example.json")
    assert args.out == Path("out/example")
    assert args.until == "segments"
    assert args.segment == "model"
    assert args.override == Path("positions_override.json")
    assert args.ref_width == 400
    assert args.ref_quality == 80
    assert args.no_cache is True


@pytest.mark.parametrize("option", ["--until", "--segment"])
def test_cli_rejects_unknown_stage_or_segment(option):
    with pytest.raises(SystemExit) as error:
        run_demo.build_parser().parse_args([option, "unknown"])
    assert error.value.code == 2


@pytest.mark.parametrize("ref_key", ["ref", "expert"])
def test_labels_select_the_default_mode_with_both_reference_keys(ref_key):
    rep = {"index": 1, "phases": {"down": [0, 3], "bottom": [3, 4], "up": [4, 8]}}
    settings = {"user": {"reps": [rep]}, ref_key: {"reps": [rep]}}
    assert run_demo.segment_mode(settings) == "labels"
    assert run_demo.segment_mode(settings, "model") == "model"


def test_model_is_the_default_without_manual_labels():
    settings = {"user": {"video": "user.mp4"}, "ref": {"video": "ref.mp4"}}
    assert run_demo.segment_mode(settings) == "model"
    assert run_demo.segment_mode(settings, "labels") == "labels"


def test_cli_help_works_from_another_directory_without_loading_config(tmp_path):
    result = subprocess.run(
        [sys.executable, str(config.PROJECT_ROOT / "run_demo.py"), "--help"],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert result.returncode == 0, result.stderr
    for option in ("--config", "--out", "--until", "--segment", "--override",
                   "--ref-width", "--ref-quality", "--no-cache"):
        assert option in result.stdout


def test_cli_reads_utf8_config_with_synthetic_pose(tmp_path, monkeypatch, capsys):
    import numpy as np

    from feedback import gate, pose

    monkeypatch.setattr(gate, "check_inputs", lambda *args, **kwargs: {"warnings": []})

    lms = np.zeros((21, 33, 4), dtype=np.float32)
    lms[:, :, 3] = 1
    lms[:, 23:25, 1] = np.concatenate((np.arange(11), np.arange(9, -1, -1)))[:, None] / 10
    lms[:, 31:33, 0] = 0.1
    synthetic = {"lms": lms, "valid": np.ones(21, dtype=bool), "fps": 30.0,
                 "w": 640, "h": 480, "total": 21, "cache_hit": False}
    monkeypatch.setattr(pose, "extract_pose", lambda *args, **kwargs: synthetic)
    rep = {"index": 1, "phases": {"down": [0, 9], "bottom": [9, 12], "up": [12, 21]}}
    settings_path = tmp_path / "설정.json"
    settings_path.write_text(json.dumps({
        "exercise": "squat", "user": {"video": "사용자.mp4", "view": "side", "reps": [rep]},
        "ref": {"video": "레퍼런스.mp4", "view": "side", "reps": [rep]},
    }, ensure_ascii=False), encoding="utf-8")
    output_path = tmp_path / "결과"
    assert run_demo.main(["--config", str(settings_path), "--out", str(output_path),
                          "--until", "segments"]) == 0
    for role, name in (("user", "사용자.mp4"), ("ref", "레퍼런스.mp4")):
        segments = json.loads((output_path / f"segments_{role}.json").read_text(encoding="utf-8"))
        assert segments["video"] == name
        assert segments["total"] == 21
        assert segments["reps"][0]["positions"]["START"] == 0
        assert segments["reps"][0]["positions"]["LOCKOUT"] == 20
    output = capsys.readouterr().out
    assert "total=21 detected=21 facing=right" in output
    assert "hip_y=" in output
