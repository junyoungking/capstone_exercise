#!/usr/bin/env python3
"""세트 후 리포트 CLI. labels·model 모드의 처리 단계를 순서대로 연결한다."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from feedback import config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=config.CLI_DESCRIPTION)
    parser.add_argument("--config", type=Path, default=config.DEFAULT_CONFIG,
                        help=config.CLI_HELP["config"])
    parser.add_argument("--out", type=Path, default=config.DEFAULT_OUT,
                        help=config.CLI_HELP["out"])
    parser.add_argument("--until", choices=config.STAGES, default=config.DEFAULT_UNTIL,
                        help=config.CLI_HELP["until"])
    parser.add_argument("--segment", choices=config.SEGMENT_MODES,
                        help=config.CLI_HELP["segment"])
    parser.add_argument("--ckpt", type=Path, help=config.CLI_HELP["ckpt"])
    parser.add_argument("--override", type=Path, help=config.CLI_HELP["override"])
    parser.add_argument("--ref-width", type=int, default=config.DEFAULT_REF_WIDTH,
                        help=config.CLI_HELP["ref_width"])
    parser.add_argument("--ref-quality", type=int, default=config.DEFAULT_REF_QUALITY,
                        help=config.CLI_HELP["ref_quality"])
    parser.add_argument("--no-cache", action="store_true", help=config.CLI_HELP["no_cache"])
    return parser


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def segment_mode(settings: dict, explicit: str | None = None) -> str:
    if explicit is not None:
        return explicit
    ref = settings.get("ref", settings.get("expert", {}))
    if "reps" in settings.get("user", {}) and "reps" in ref:
        return "labels"
    return "model"


def _read_json(path: Path, parser: argparse.ArgumentParser) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        parser.error(config.CONFIG_ERROR_TEMPLATE.format(path=path, reason=error))


def _run_segments(settings: dict, args: argparse.Namespace, overrides: dict,
                  mode: str = "labels", *, validate: bool = False) -> dict:
    from feedback.pose import extract_pose
    from feedback.segment import build_segments, hip_y_trajectory

    if not isinstance(overrides, dict) or set(overrides) - {"user", "ref"}:
        raise ValueError({"overrides": overrides})
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = {"user": settings["user"], "ref": settings.get("ref", settings.get("expert"))}
    results = {}
    for role, video in videos.items():
        cache_path = out / f"pose_{role}.npz"
        pose = extract_pose(project_path(Path(video["video"])), cache_path, settings["exercise"],
                            no_cache=args.no_cache)
        results[role] = {"pose": pose}
        cache_status = "hit" if pose["cache_hit"] else "miss"
        print(f"{role}: cache={cache_status} file={cache_path.name}")
    if mode == "model":
        from feedback.stgcn_infer import infer_phases

        for result in results.values():
            result["inference"] = infer_phases(result["pose"], project_path(args.ckpt))
    if validate:
        from feedback import gate

        exercises = [results[role]["inference"]["exercise"] if mode == "model"
                     else settings["exercise"] for role in videos]
        gate.check_exercises(*exercises)
        gate.check_views(*(video["view"] for video in videos.values()))
        for role, result in results.items():
            gate.check_detection(role, result["pose"])
        if mode == "labels":
            for role, video in videos.items():
                gate.check_label_ranges(role, video, results[role]["pose"]["total"])
    for role, video in videos.items():
        pose = results[role]["pose"]
        inference = results[role].get("inference")
        try:
            if mode == "model":
                from feedback.segment import build_model_segments

                segments = build_model_segments(pose, video, inference, overrides.get(role))
            else:
                segments = build_segments(pose, video, settings["exercise"], overrides.get(role))
        except (ValueError, KeyError) as error:
            if validate:
                raise ValueError(gate.friendly_error(error, role)) from error
            raise
        results[role]["segments"] = segments
        if inference is not None:
            print(f"{role}: model exercise={inference['exercise']} "
                  f"ratio={inference['exercise_ratio']:.3f} reps={len(segments['reps'])}")
            print(f"{role}: class_votes={inference['votes']} class_ratios={inference['ratios']}")
            print(f"{role}: checkpoint={inference['checkpoint']}")
        segment_path = out / f"segments_{role}.json"
        segment_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2,
                                           allow_nan=False) + "\n", encoding="utf-8")
        print(f"{role}: video={segments['video']} total={pose['total']} "
              f"detected={int(pose['valid'].sum())} facing={segments['facing'] or 'null'}")
        hip_y = hip_y_trajectory(pose)
        for rep in segments["reps"]:
            phases = " ".join(f"{phase}=[{rep['phases'][phase][0]},{rep['phases'][phase][1]})"
                              for phase in config.PHASES)
            positions = []
            for position in config.POSITIONS:
                frame = rep["positions"][position]
                height = f"{hip_y[frame]:.3f}" if math.isfinite(hip_y[frame]) else "null"
                positions.append(f"{position}={frame}(hip_y={height})")
            print(f"{role}: rep={rep['rep']} side={rep['side']} phases={phases} " + " ".join(positions))
    return results


def _number(value, digits: int = 3) -> str:
    return "null" if value is None else f"{value:.{digits}f}"


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _run_metrics(results: dict, out: Path) -> dict:
    from feedback.metrics import build_metrics

    print("[metrics]")
    measured = {}
    for role, result in results.items():
        metrics = build_metrics(result["pose"], result["segments"])
        measured[role] = metrics
        _write_json(out / f"metrics_{role}.json", metrics)
        names = config.DISPLAY_METRICS[metrics["exercise"]]
        for rep in metrics["reps"]:
            print(f"{role}: rep={rep['rep']}")
            print("position(frame) " + " ".join(names))
            for position in config.POSITIONS:
                item = rep["positions"][position]
                values = " ".join(_number(item["values"][name]) for name in names)
                print(f"{position}({item['frame']}) {values}")
            print("tempo " + " ".join(f"{name}={_number(value)}"
                                        for name, value in rep["tempo"].items()))
            print("detect " + " ".join(f"{phase}={_number(rep['detect'][phase])}"
                                         for phase in config.PHASES))
            print(f"excluded={str(rep['excluded']).lower()} reason={rep['exclude_reason'] or 'null'}")
    return measured


def _run_findings(measured: dict, settings: dict, out: Path) -> dict:
    from feedback.compare import compare_reps
    from feedback.select import select_goals

    reference = settings.get("ref", settings.get("expert"))
    findings = select_goals(compare_reps(measured["user"], measured["ref"], reference.get("rep", 1)))
    _write_json(out / "findings.json", findings)
    names = config.DISPLAY_METRICS[findings["exercise"]]
    print("[compare: diff/k/color]")
    for rep in findings["reps"]:
        print(f"user: rep={rep['rep']}")
        print("position " + " ".join(names))
        cells = {(cell["where"], cell["metric"]): cell for cell in rep["cells"]}
        for position in config.POSITIONS:
            row = []
            for metric in names:
                cell = cells[position, metric]
                row.append(f"{_number(cell['diff'])}/{_number(cell['k'])}/{cell['color']}")
            print(position + " " + " ".join(row))
        print(f"excluded={str(rep['excluded']).lower()} reason={rep['exclude_reason'] or 'null'}")
    print("[select]")
    for rep in findings["reps"]:
        goal = rep["goal"]
        prefix = f"user: rep={rep['rep']}"
        if goal is None or goal["type"] == "none":
            print(f"{prefix} goal type={goal['type'] if goal else 'null'}")
        else:
            metrics = goal.get("metrics", [goal.get("metric")])
            print(f"{prefix} goal type={goal['type']} where={goal['where']} "
                  f"metrics={','.join(metrics)} dir={goal['dir']} k={_number(goal['k'])} color={goal['color']}")
            lookup = {(cell["where"], cell["metric"]): cell for cell in rep["cells"]}
            evidence = []
            for item in goal["evidence"]:
                cell = lookup[item["where"], item["metric"]]
                evidence.append(f"{cell['where']}.{cell['metric']}(diff={_number(cell['diff'])} "
                                f"k={_number(cell['k'])} color={cell['color']})")
            print("evidence " + "; ".join(evidence))
        tempo_goal = rep["tempo_goal"]
        if tempo_goal:
            print(f"tempo_goal metric={tempo_goal['metric']} dir={tempo_goal['dir']} "
                  f"diff={_number(tempo_goal['diff'])} k={_number(tempo_goal['k'])} color={tempo_goal['color']}")
        else:
            print("tempo_goal=null")
    return findings


def _print_descriptions(findings: dict) -> None:
    from feedback.describe import describe_reps

    print("[describe]")
    for rep in describe_reps(findings):
        print(f"user: rep={rep['rep']} tab_label={rep['tab_label']}")
        print(f"goal_sentence={rep['goal_sentence']}")
        print(f"tempo_sentence={rep['tempo_sentence'] or 'null'}")


def _run_report(results: dict, measured: dict, findings: dict,
                args: argparse.Namespace, *, warnings: list[str] | None = None,
                angles: dict | None = None) -> None:
    from feedback.export import export_report

    out = project_path(args.out)
    report = export_report(results, measured, findings, out,
                           ref_width=args.ref_width, ref_quality=args.ref_quality,
                           warnings=warnings, angles=angles)
    print("[report]")
    print(f"json={out / 'report.json'}")
    print(f"html={out / 'report.html'}")
    print(f"videos user={report['user']['video']} ref={report['ref']['video']}")
    print(f"set_bar_warnings={report['meta']['warnings']}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = _read_json(project_path(args.config), parser)
    if "ref" not in settings and "expert" in settings:
        settings["ref"] = settings["expert"]
    mode = segment_mode(settings, args.segment)
    if mode == "model" and args.ckpt is None:
        parser.error(config.MODEL_CHECKPOINT_REQUIRED)
    overrides = _read_json(project_path(args.override), parser) if args.override else {}
    from feedback import gate

    try:
        inputs = gate.check_inputs(settings, mode)
        warnings = list(inputs.get("warnings", []))
        results = _run_segments(settings, args, overrides, mode, validate=True)
        angles = {}
        for role, result in results.items():
            angle = gate.observe_angle(role, result["pose"], result["segments"])
            angles[role] = angle
            print(f"{role}: angle_ratio={_number(angle['ratio'])} "
                  f"angle_view={config.VIEW_LABELS.get(angle['view'], 'null')}")
            if angle.get("warning"):
                warnings.append(angle["warning"])
        for warning in warnings:
            print(f"warning={warning}")
        if args.until != "segments":
            measured = _run_metrics(results, project_path(args.out))
            if args.until in ("findings", "report"):
                reference = settings.get("ref", settings.get("expert"))
                gate.check_reference(measured["ref"], reference.get("rep", 1))
                findings = _run_findings(measured, settings, project_path(args.out))
                gate.check_comparable(findings, measured["ref"], reference.get("rep", 1))
                _print_descriptions(findings)
                if args.until == "report":
                    _run_report(results, measured, findings, args, warnings=warnings, angles=angles)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        reason = gate.friendly_error(error)
        if args.until == "report":
            from feedback.export import export_failure

            command = ["python", "run_demo.py", "--config", args.config.as_posix(),
                       "--out", args.out.as_posix()]
            if args.override:
                command.extend(["--override", args.override.as_posix()])
            if args.segment:
                command.extend(["--segment", args.segment])
            if args.ckpt:
                command.extend(["--ckpt", args.ckpt.as_posix()])
            try:
                failure = export_failure(reason, project_path(args.out), command,
                                         remedy=getattr(error, "remedy", None),
                                         check=getattr(error, "check", None))
                print(f"failure={failure}")
            except (OSError, ValueError):
                pass
        parser.error(reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
