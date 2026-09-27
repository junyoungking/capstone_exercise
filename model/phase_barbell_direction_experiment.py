"""Barbell-only phase-label direction experiment runner.

This runner implements the approved ``phase_label_scheme=bar_direction`` plan:

1. Re-evaluate the existing barbell checkpoint without training it again, using
   ``trained_phase_label_scheme=as_labeled`` and
   ``eval_phase_label_scheme=bar_direction``.
2. Train the same full barbell-position MLP contract with only the target label
   scheme changed to ``bar_direction``.
3. Export diagnostics and a stable comparison CSV under
   ``phase_experiments/barbell_yolo_world/bar_direction/``.

The model path intentionally stays simple: barbell one-keypoint position input,
``phase_head_type=mlp``, ``derivative_mode=pose``, ``phase_pooling=temporal_avg``.
No TCN and no new dependencies are introduced.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:  # `import model.phase_barbell_direction_experiment`
    from . import phase_first_diagnostics as diagnostics
    from . import train_ablation as ta
except ImportError:  # `python model/phase_barbell_direction_experiment.py` / PyCharm script mode
    import phase_first_diagnostics as diagnostics  # type: ignore
    import train_ablation as ta  # type: ignore


RUN_KIND = "full"  # use "smoke" for a 1-epoch pipeline check
SMOKE_EPOCHS = 1
FULL_EPOCHS = 60
SMOKE_MAX_VIDEOS_PER_SPLIT_TYPE = 2
SMOOTH_WINDOW = 5

OUTPUT_ROOT = Path("phase_experiments") / "barbell_yolo_world" / "bar_direction"
HISTORICAL_RESULTS_CSV = Path("phase_experiments") / "barbell_yolo_world" / "full" / "ablation_results.csv"
BASELINE_ARTIFACT_LABEL = "trained_as_labeled_eval_bar_direction"
TRAINED_ARTIFACT_LABEL = "trained_bar_direction_eval_bar_direction"
COMPARISON_FILENAME = "phase_barbell_direction_comparison.csv"
NOTE = (
    "Deadlift movement segments are reinterpreted as physical bar direction "
    "(start:middle=up, middle:finish=down). Squat and benchpress keep the "
    "legacy down/up mapping. The baseline row is eval-only and writes no "
    "checkpoint."
)

# Resume-safe defaults.  The new bar_direction model resumes/skips completed
# checkpoints by default; pass --force-retrain for a deliberately fresh run.
FORCE_RETRAIN = False
FRESH_RERUN = False
RUN_TAG = None


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def diagnostic_summary(diagnostic_artifacts: Mapping[str, Path]) -> Dict[str, Any]:
    metrics_path = Path(diagnostic_artifacts["metrics"])
    if not metrics_path.exists():
        return {}
    metrics = pd.read_csv(metrics_path)
    out: Dict[str, Any] = {}
    for _, row in metrics.iterrows():
        mode = str(row.get("mode", "raw"))
        prefix = "" if mode == "raw" else f"{mode}_"
        for key, value in row.items():
            if key in {"mode", "n"}:
                continue
            out[f"{prefix}{key}"] = value
    return out


def normalize_optional_path(value: Optional[str | Path]) -> Optional[Path]:
    if value in {None, "", "None"}:
        return None
    return Path(value).expanduser()


def build_train_cfg(
    *,
    run_kind: str,
    output_root: Path,
    epochs: Optional[int] = None,
    max_videos_per_split_type: Optional[int] = None,
    force_retrain: bool = FORCE_RETRAIN,
    fresh_rerun: bool = FRESH_RERUN,
    run_tag: Optional[str] = RUN_TAG,
) -> Dict[str, Any]:
    run_kind = str(run_kind).lower()
    if run_kind not in {"smoke", "full"}:
        raise ValueError(f"run_kind must be 'smoke' or 'full', got {run_kind}")
    if force_retrain and fresh_rerun and not run_tag:
        run_tag = f"rerun_{utc_stamp()}"

    cfg = copy.deepcopy(ta.DEFAULT_EXPERIMENT_CONFIG)
    cfg.update(
        {
            "model_type": "mlp",
            "phase_head_type": ta.PHASE_HEAD_MLP,
            "phase_pooling": "temporal_avg",
            "derivative_mode": "pose",
            "phase_label_scheme": ta.PHASE_LABEL_SCHEME_BAR_DIRECTION,
            "pose_backend": ta.POSE_BACKEND_BARBELL,
            "joint_subset": ta.JOINT_SUBSET_ALL,
            "hidden": 128,
            "clip_len": 16,
            "train_stride": 2,
            "dropout": 0.3,
            "aug": True,
            "epochs": epochs if epochs is not None else (SMOKE_EPOCHS if run_kind == "smoke" else FULL_EPOCHS),
            "batch": ta.DEFAULT_EXPERIMENT_CONFIG["batch"],
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "phase_loss_alpha": 2.0,
            "num_workers": 0,
            "run_kind": run_kind,
            "output_root": str(output_root / run_kind),
            "write_meta_csv": False,
            "extract_missing_pose": False,
            "resume": not force_retrain,
            "skip_completed": not force_retrain,
            "force_retrain": force_retrain,
            "fresh_run_tag": run_tag if force_retrain and fresh_rerun else None,
            "overwrite_existing": False,
            "max_videos_per_split_type": (
                max_videos_per_split_type
                if max_videos_per_split_type is not None
                else (SMOKE_MAX_VIDEOS_PER_SPLIT_TYPE if run_kind == "smoke" else None)
            ),
            "pin_memory": True,
        }
    )
    return ta.normalize_cfg(cfg)


def path_from_row(row: pd.Series, key: str) -> Optional[Path]:
    value = row.get(key)
    if pd.isna(value) or value in {None, "", "None"}:
        return None
    path = Path(str(value))
    if path.exists():
        return path
    return None


def candidate_checkpoint_paths(row: pd.Series, historical_csv: Path) -> Iterable[Path]:
    checkpoint = path_from_row(row, "checkpoint_path")
    if checkpoint is not None:
        yield checkpoint

    exp_dir = path_from_row(row, "exp_dir")
    if exp_dir is not None:
        for candidate in sorted(exp_dir.glob("best_ep*.pt")):
            yield candidate
        latest = exp_dir / "latest.pt"
        if latest.exists():
            yield latest

    exp_name = str(row.get("exp_name", "")).strip()
    if exp_name:
        relative_dir = historical_csv.parent / exp_name
        for candidate in sorted(relative_dir.glob("best_ep*.pt")):
            yield candidate
        latest = relative_dir / "latest.pt"
        if latest.exists():
            yield latest


def find_baseline_checkpoint(
    *,
    explicit_checkpoint: Optional[Path],
    historical_csv: Path,
) -> Tuple[Path, Optional[pd.Series]]:
    if explicit_checkpoint is not None:
        if not explicit_checkpoint.exists():
            raise FileNotFoundError(f"baseline checkpoint not found: {explicit_checkpoint}")
        return explicit_checkpoint, None

    if historical_csv.exists():
        results = pd.read_csv(historical_csv)
        if not results.empty:
            filtered = results.copy()
            for col, expected in {
                "completion_status": "complete",
                "pose_backend": ta.POSE_BACKEND_BARBELL,
                "phase_head_type": ta.PHASE_HEAD_MLP,
                "derivative_mode": "pose",
                "joint_subset": ta.JOINT_SUBSET_ALL,
            }.items():
                if col in filtered.columns:
                    filtered = filtered[filtered[col].astype(str).str.lower() == expected]
            if "phase_label_scheme" in filtered.columns:
                filtered = filtered[
                    filtered["phase_label_scheme"]
                    .fillna(ta.PHASE_LABEL_SCHEME_AS_LABELED)
                    .astype(str)
                    .map(ta.normalize_phase_label_scheme)
                    == ta.PHASE_LABEL_SCHEME_AS_LABELED
                ]
            sort_col = "video_phase_f1" if "video_phase_f1" in filtered.columns else "best_val_score"
            if sort_col in filtered.columns and not filtered.empty:
                filtered = filtered.sort_values(sort_col, ascending=False, na_position="last")
            for _, row in filtered.iterrows():
                for candidate in candidate_checkpoint_paths(row, historical_csv):
                    if candidate.exists():
                        return candidate, row

    search_root = historical_csv.parent
    candidates = sorted(search_root.glob("**/best_ep*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(search_root.glob("**/latest.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0], None
    raise FileNotFoundError(
        "Could not resolve an as_labeled barbell baseline checkpoint. "
        f"Checked explicit path and historical CSV root: {historical_csv}"
    )


def checkpoint_cfg(checkpoint_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=ta.DEVICE)
    cfg = ckpt.get("cfg")
    if not isinstance(cfg, Mapping):
        raise ValueError(f"checkpoint has no cfg mapping: {checkpoint_path}")
    return ta.normalize_cfg(dict(cfg))


def eval_checkpoint_video_metrics(
    *,
    checkpoint_path: Path,
    trained_cfg: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
    context: ta.ExperimentContext,
) -> Dict[str, float]:
    ckpt = torch.load(checkpoint_path, map_location=ta.DEVICE)
    model = ta.build_model(trained_cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return ta.eval_videos(
        model,
        context.val_meta,
        int(eval_cfg["clip_len"]),
        int(eval_cfg["train_stride"]),
        labels=context.labels,
        pose_backend=str(eval_cfg["pose_backend"]),
        joint_subset=str(eval_cfg["joint_subset"]),
        phase_label_scheme=str(eval_cfg["phase_label_scheme"]),
    )


def baseline_eval_cfg(
    *,
    trained_cfg: Mapping[str, Any],
    output_root: Path,
    max_videos_per_split_type: Optional[int],
) -> Dict[str, Any]:
    cfg = dict(trained_cfg)
    cfg.update(
        {
            "phase_label_scheme": ta.PHASE_LABEL_SCHEME_BAR_DIRECTION,
            "run_kind": "baseline_eval",
            "output_root": str(output_root / "baseline_eval"),
            "write_meta_csv": False,
            "extract_missing_pose": False,
            "resume": True,
            "skip_completed": True,
            "force_retrain": False,
            "fresh_run_tag": None,
            "max_videos_per_split_type": max_videos_per_split_type,
        }
    )
    return ta.normalize_cfg(cfg)


def run_baseline_eval_only(
    *,
    checkpoint_path: Path,
    output_root: Path,
    context: ta.ExperimentContext,
    smooth_window: int,
    diagnostic_max_videos: Optional[int],
    historical_row: Optional[pd.Series],
) -> Dict[str, Any]:
    trained_cfg = checkpoint_cfg(checkpoint_path)
    trained_cfg["phase_label_scheme"] = ta.normalize_phase_label_scheme(
        trained_cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED)
    )
    if trained_cfg["phase_label_scheme"] != ta.PHASE_LABEL_SCHEME_AS_LABELED:
        raise ValueError(
            "eval-only baseline must use a checkpoint trained with "
            f"phase_label_scheme={ta.PHASE_LABEL_SCHEME_AS_LABELED!r}; "
            f"got {trained_cfg['phase_label_scheme']!r} from {checkpoint_path}"
        )
    eval_cfg = baseline_eval_cfg(
        trained_cfg=trained_cfg,
        output_root=output_root,
        max_videos_per_split_type=context.cfg.get("max_videos_per_split_type"),
    )
    out_dir = output_root / "baseline_eval" / BASELINE_ARTIFACT_LABEL
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    video_metrics = eval_checkpoint_video_metrics(
        checkpoint_path=checkpoint_path,
        trained_cfg=trained_cfg,
        eval_cfg=eval_cfg,
        context=context,
    )
    diagnostic_artifacts = diagnostics.run_diagnostics_for_checkpoint(
        checkpoint_path,
        cfg=eval_cfg,
        context=context,
        output_dir=out_dir,
        run_id=BASELINE_ARTIFACT_LABEL,
        smooth_window=smooth_window,
        max_videos=diagnostic_max_videos,
        historical_row=historical_row,
        manifest_name="diagnostic_manifest.json",
        trained_cfg=trained_cfg,
        artifact_label=BASELINE_ARTIFACT_LABEL,
    )
    elapsed_min = round((time.time() - started) / 60, 3)
    metric_summary = diagnostic_summary(diagnostic_artifacts)
    result = {
        "artifact_label": BASELINE_ARTIFACT_LABEL,
        "exp_name": BASELINE_ARTIFACT_LABEL,
        "run_kind": "baseline_eval",
        "completion_status": "complete",
        "fresh_training_run": False,
        "checkpoint_path": str(checkpoint_path),
        "trained_phase_label_scheme": trained_cfg.get("phase_label_scheme"),
        "eval_phase_label_scheme": eval_cfg.get("phase_label_scheme"),
        "elapsed_min": elapsed_min,
        **video_metrics,
        **metric_summary,
    }
    manifest = {
        "run_id": BASELINE_ARTIFACT_LABEL,
        "created_at_utc": utc_stamp(),
        "completion_status": "complete",
        "fresh_training_run": False,
        "artifact_label": BASELINE_ARTIFACT_LABEL,
        "checkpoint_path": str(checkpoint_path),
        "trained_config": trained_cfg,
        "eval_config": eval_cfg,
        "trained_phase_label_scheme": trained_cfg.get("phase_label_scheme"),
        "eval_phase_label_scheme": eval_cfg.get("phase_label_scheme"),
        "metrics": result,
        "artifacts": {k: str(v) for k, v in diagnostic_artifacts.items()},
        "note": NOTE,
    }
    write_json(out_dir / "baseline_eval_metrics.json", result)
    write_json(out_dir / "run_manifest.json", manifest)
    return {"result": result, "manifest": manifest, "diagnostics": diagnostic_artifacts}


def training_manifest(
    *,
    cfg: Mapping[str, Any],
    result: Mapping[str, Any],
    context: ta.ExperimentContext,
    diagnostic_artifacts: Mapping[str, Path],
) -> Dict[str, Any]:
    metric_summary = diagnostic_summary(diagnostic_artifacts)
    metrics = {
        **{k: result.get(k) for k in result if k.startswith("video_") or k.startswith("mae_") or k.startswith("obo_")},
        **metric_summary,
    }
    return {
        "run_id": f"{cfg['run_kind']}_{TRAINED_ARTIFACT_LABEL}_{utc_stamp()}",
        "created_at_utc": utc_stamp(),
        "completion_status": result.get("completion_status", "complete"),
        "fresh_training_run": not bool(result.get("skipped", False)),
        "artifact_label": TRAINED_ARTIFACT_LABEL,
        "checkpoint_path": result.get("checkpoint_path"),
        "latest_path": result.get("latest_path"),
        "history_path": result.get("history_path"),
        "config": dict(cfg),
        "trained_phase_label_scheme": cfg.get("phase_label_scheme"),
        "eval_phase_label_scheme": cfg.get("phase_label_scheme"),
        "data_split_hash": ta.data_split_fingerprint(context.meta_df),
        "metrics": metrics,
        "artifacts": {k: str(v) for k, v in diagnostic_artifacts.items()},
        "note": NOTE,
    }


def run_bar_direction_training(
    *,
    cfg: Dict[str, Any],
    context: ta.ExperimentContext,
    output_root: Path,
    smooth_window: int,
    diagnostic_max_videos: Optional[int],
) -> Dict[str, Any]:
    result = ta.run_one_experiment(cfg, context=context)
    diagnostics_dir = output_root / "diagnostics" / str(cfg["run_kind"]) / str(result["exp_name"])
    diagnostic_artifacts = diagnostics.run_diagnostics_for_checkpoint(
        result["checkpoint_path"],
        cfg=cfg,
        context=context,
        output_dir=diagnostics_dir,
        run_id=f"{cfg['run_kind']}_{TRAINED_ARTIFACT_LABEL}",
        smooth_window=smooth_window,
        max_videos=diagnostic_max_videos,
        manifest_name="diagnostic_manifest.json",
        trained_cfg=cfg,
        artifact_label=TRAINED_ARTIFACT_LABEL,
    )
    manifest = training_manifest(cfg=cfg, result=result, context=context, diagnostic_artifacts=diagnostic_artifacts)
    manifest_path = Path(result["exp_dir"]) / "run_manifest.json"
    write_json(manifest_path, manifest)
    return {"result": dict(result), "manifest": manifest, "diagnostics": diagnostic_artifacts}


def historical_row_for_comparison(row: Optional[pd.Series]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    payload = row.to_dict()
    return {
        "artifact_label": "historical_as_labeled_eval_as_labeled",
        "exp_name": payload.get("exp_name"),
        "run_kind": payload.get("run_kind", "historical"),
        "completion_status": payload.get("completion_status", "complete"),
        "fresh_training_run": False,
        "checkpoint_path": payload.get("checkpoint_path"),
        "trained_phase_label_scheme": ta.PHASE_LABEL_SCHEME_AS_LABELED,
        "eval_phase_label_scheme": ta.PHASE_LABEL_SCHEME_AS_LABELED,
        "best_epoch": payload.get("best_epoch"),
        "best_val_score": payload.get("best_val_score"),
        "video_action_acc": payload.get("video_action_acc"),
        "video_action_f1": payload.get("video_action_f1"),
        "video_phase_acc": payload.get("video_phase_acc"),
        "video_phase_f1": payload.get("video_phase_f1"),
        "phase_acc": payload.get("phase_acc"),
        "phase_macro_f1": payload.get("phase_macro_f1"),
        "video_count_mae": payload.get("video_count_mae"),
        "video_count_obo": payload.get("video_count_obo"),
        "ready_f1": payload.get("ready_f1"),
        "down_f1": payload.get("down_f1"),
        "up_f1": payload.get("up_f1"),
    }


def comparison_row(result: Mapping[str, Any], *, artifact_label: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "artifact_label": artifact_label,
        "exp_name": result.get("exp_name", artifact_label),
        "run_kind": result.get("run_kind"),
        "completion_status": result.get("completion_status"),
        "fresh_training_run": result.get("fresh_training_run", artifact_label == TRAINED_ARTIFACT_LABEL),
        "checkpoint_path": result.get("checkpoint_path"),
        "trained_phase_label_scheme": result.get("trained_phase_label_scheme", result.get("phase_label_scheme")),
        "eval_phase_label_scheme": result.get("eval_phase_label_scheme", result.get("phase_label_scheme")),
        "best_epoch": result.get("best_epoch"),
        "best_val_score": result.get("best_val_score"),
        "elapsed_min": result.get("elapsed_min"),
    }
    for key, value in result.items():
        if (
            key.startswith("video_")
            or key.startswith("mae_")
            or key.startswith("obo_")
            or key.startswith("phase_acc")
            or key.startswith("phase_macro_f1")
            or key in {"ready_f1", "down_f1", "up_f1"}
        ):
            row[key] = value
    return row


def write_comparison(
    *,
    output_root: Path,
    rows: List[Dict[str, Any]],
) -> Dict[str, Path]:
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = comparison_dir / COMPARISON_FILENAME
    comparison_json = comparison_dir / "phase_barbell_direction_comparison.json"
    frame = pd.DataFrame(rows)
    if not frame.empty:
        sort_cols = [c for c in ["eval_phase_label_scheme", "artifact_label"] if c in frame.columns]
        if sort_cols:
            frame = frame.sort_values(sort_cols)
    frame.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    comparison_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    return {"comparison_csv": comparison_csv, "comparison_json": comparison_json}


def run(
    *,
    run_kind: str = RUN_KIND,
    output_root: Path = OUTPUT_ROOT,
    epochs: Optional[int] = None,
    max_videos_per_split_type: Optional[int] = None,
    baseline_checkpoint: Optional[Path] = None,
    historical_results_csv: Path = HISTORICAL_RESULTS_CSV,
    baseline_only: bool = False,
    skip_baseline: bool = False,
    force_retrain: bool = FORCE_RETRAIN,
    fresh_rerun: bool = FRESH_RERUN,
    run_tag: Optional[str] = RUN_TAG,
    smooth_window: int = SMOOTH_WINDOW,
    diagnostic_max_videos: Optional[int] = None,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = build_train_cfg(
        run_kind=run_kind,
        output_root=output_root,
        epochs=epochs,
        max_videos_per_split_type=max_videos_per_split_type,
        force_retrain=force_retrain,
        fresh_rerun=fresh_rerun,
        run_tag=run_tag,
    )
    print(
        f"[phase_barbell_direction_experiment] run_kind={cfg['run_kind']} epochs={cfg['epochs']} "
        f"pose_backend={cfg['pose_backend']} head={cfg['phase_head_type']} input={cfg['derivative_mode']} "
        f"pooling={cfg['phase_pooling']} label_scheme={cfg['phase_label_scheme']} "
        f"output_root={output_root}",
        flush=True,
    )
    context = ta.prepare_context(cfg, verbose=True, update_globals=True)

    comparison_rows: List[Dict[str, Any]] = []
    artifacts: Dict[str, Any] = {}
    baseline_payload: Optional[Dict[str, Any]] = None

    if not skip_baseline:
        checkpoint_path, historical_row = find_baseline_checkpoint(
            explicit_checkpoint=baseline_checkpoint,
            historical_csv=historical_results_csv,
        )
        historical_cmp = historical_row_for_comparison(historical_row)
        if historical_cmp is not None:
            comparison_rows.append(historical_cmp)
        baseline_payload = run_baseline_eval_only(
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            context=context,
            smooth_window=smooth_window,
            diagnostic_max_videos=diagnostic_max_videos,
            historical_row=historical_row,
        )
        comparison_rows.append(comparison_row(baseline_payload["result"], artifact_label=BASELINE_ARTIFACT_LABEL))
        artifacts["baseline_eval"] = baseline_payload["manifest"]["artifacts"]

    training_payload: Optional[Dict[str, Any]] = None
    if not baseline_only:
        training_payload = run_bar_direction_training(
            cfg=cfg,
            context=context,
            output_root=output_root,
            smooth_window=smooth_window,
            diagnostic_max_videos=diagnostic_max_videos,
        )
        comparison_rows.append(comparison_row(training_payload["result"], artifact_label=TRAINED_ARTIFACT_LABEL))
        artifacts["training"] = training_payload["manifest"]["artifacts"]

    comparison_artifacts = write_comparison(output_root=output_root, rows=comparison_rows)
    artifacts["comparison"] = {k: str(v) for k, v in comparison_artifacts.items()}
    manifest = {
        "run_id": f"barbell_direction_{cfg['run_kind']}_{utc_stamp()}",
        "created_at_utc": utc_stamp(),
        "completion_status": "complete",
        "run_kind": cfg["run_kind"],
        "baseline_only": baseline_only,
        "skip_baseline": skip_baseline,
        "config": cfg,
        "data_split_hash": ta.data_split_fingerprint(context.meta_df),
        "baseline_artifact_label": BASELINE_ARTIFACT_LABEL,
        "trained_artifact_label": TRAINED_ARTIFACT_LABEL,
        "artifacts": artifacts,
        "comparison_rows": comparison_rows,
        "note": NOTE,
    }
    write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps({"run_manifest": str(output_root / "run_manifest.json"), **artifacts["comparison"]}, ensure_ascii=False, indent=2))
    return manifest


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate the bar_direction barbell phase model.")
    parser.add_argument("--mode", choices=["smoke", "full"], default=RUN_KIND, help="smoke=1-epoch limited split, full=60 epochs")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count for the selected mode.")
    parser.add_argument(
        "--max-videos-per-split-type",
        type=int,
        default=None,
        help="Limit videos per split/exercise type. Defaults to 2 for smoke and unlimited for full.",
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Stable experiment output root.")
    parser.add_argument("--baseline-checkpoint", default=None, help="Explicit as_labeled checkpoint for eval-only baseline.")
    parser.add_argument("--historical-results-csv", default=str(HISTORICAL_RESULTS_CSV), help="Historical baseline result CSV.")
    parser.add_argument("--baseline-only", action="store_true", help="Run only the eval-only baseline path; do not train.")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline re-evaluation and train only.")
    parser.add_argument("--force-retrain", action="store_true", help="Do not resume/skip existing bar_direction checkpoints.")
    parser.add_argument("--fresh-rerun", action="store_true", help="When forcing retrain, add a fresh rerun tag.")
    parser.add_argument("--run-tag", default=None, help="Explicit fresh rerun tag when --force-retrain --fresh-rerun are used.")
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW, help="Offline smoothing window for diagnostics.")
    parser.add_argument("--diagnostic-max-videos", type=int, default=None, help="Limit diagnostic prediction export videos.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if args.baseline_only and args.skip_baseline:
        raise SystemExit("--baseline-only and --skip-baseline cannot be used together")
    return run(
        run_kind=args.mode,
        output_root=Path(args.output_root),
        epochs=args.epochs,
        max_videos_per_split_type=args.max_videos_per_split_type,
        baseline_checkpoint=normalize_optional_path(args.baseline_checkpoint),
        historical_results_csv=Path(args.historical_results_csv),
        baseline_only=bool(args.baseline_only),
        skip_baseline=bool(args.skip_baseline),
        force_retrain=bool(args.force_retrain),
        fresh_rerun=bool(args.fresh_rerun),
        run_tag=args.run_tag,
        smooth_window=int(args.smooth_window),
        diagnostic_max_videos=args.diagnostic_max_videos,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
