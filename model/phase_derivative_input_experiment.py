"""PyCharm-runnable ST-GCN phase input-channel ablation.

This runner keeps the existing phase model/hyperparameter shape fixed and
changes only the pose-window input channels:

- ``pose``: normalized ``[x, y, visibility]`` (3 channels)
- ``velocity``: pose + first temporal difference ``[dx, dy]`` (5 channels)
- ``acceleration``: pose + second temporal difference ``[ddx, ddy]`` (5 channels)
- ``velocity_acceleration``: pose + both derivative groups (7 channels)

Default behavior is resume-safe: interrupted runs can be continued by rerunning
this file because each variant writes ``latest.pt`` in a stable experiment
directory.  Set ``FORCE_RETRAIN=True`` and ``FRESH_RERUN=True`` only when a
fresh, non-resumed comparison is intentionally required.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:  # `import model.phase_derivative_input_experiment`
    from . import phase_first_diagnostics as diagnostics
    from . import train_ablation as ta
except ImportError:  # `python model/phase_derivative_input_experiment.py` / PyCharm script mode
    import phase_first_diagnostics as diagnostics  # type: ignore
    import train_ablation as ta  # type: ignore


RUN_KIND = "full"  # use "smoke" only for a quick 1-epoch pipeline check
SMOKE_EPOCHS = 1
FULL_EPOCHS = 60
SMOKE_MAX_VIDEOS_PER_SPLIT_TYPE = 2
DERIVATIVE_VARIANTS = ["pose", "velocity", "acceleration", "velocity_acceleration"]
OUTPUT_ROOT = Path("phase_experiments") / "derivative_input"
SMOOTH_WINDOW = 5
FAIRNESS_NOTE = (
    "Only input channels differ across variants; model type, pooling, clip_len, "
    "stride, loss weight, dropout, optimizer, and split are held constant."
)

# Resume-safe defaults.  Rerunning the file resumes incomplete latest.pt files or
# skips completed variants.  Flip these for a deliberately fresh rerun.
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


def build_variant_cfg(derivative_mode: str, run_kind: str, run_tag: Optional[str] = None) -> Dict[str, Any]:
    derivative_mode = ta.normalize_derivative_mode(derivative_mode)
    if derivative_mode not in DERIVATIVE_VARIANTS:
        raise ValueError(f"unexpected derivative variant for this runner: {derivative_mode}")
    run_kind = str(run_kind).lower()
    if run_kind not in {"smoke", "full"}:
        raise ValueError(f"run_kind must be 'smoke' or 'full', got {run_kind}")
    if FORCE_RETRAIN and FRESH_RERUN and not run_tag:
        run_tag = f"rerun_{utc_stamp()}"
    cfg = copy.deepcopy(ta.DEFAULT_EXPERIMENT_CONFIG)
    cfg.update(
        {
            "model_type": "mlp",
            "phase_pooling": "temporal_avg",
            "derivative_mode": derivative_mode,
            "hidden": 128,
            "clip_len": 16,
            "train_stride": 2,
            "dropout": 0.3,
            "aug": True,
            "epochs": SMOKE_EPOCHS if run_kind == "smoke" else FULL_EPOCHS,
            "batch": ta.DEFAULT_EXPERIMENT_CONFIG["batch"],
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "phase_loss_alpha": 2.0,
            "num_workers": 0,
            "run_kind": run_kind,
            "output_root": str(OUTPUT_ROOT / run_kind),
            "write_meta_csv": False,
            "extract_missing_pose": False,
            "resume": not FORCE_RETRAIN,
            "skip_completed": not FORCE_RETRAIN,
            "force_retrain": FORCE_RETRAIN,
            "fresh_run_tag": run_tag if FORCE_RETRAIN and FRESH_RERUN else None,
            "overwrite_existing": False,
            "max_videos_per_split_type": SMOKE_MAX_VIDEOS_PER_SPLIT_TYPE if run_kind == "smoke" else None,
            "pin_memory": True,
        }
    )
    return ta.normalize_cfg(cfg)


def manifest_path_for(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["output_root"]) / ta.make_run_exp_name(cfg) / "run_manifest.json"


def final_manifest_path_for_result(result: Dict[str, Any]) -> Path:
    return Path(result["exp_dir"]) / "run_manifest.json"


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def base_manifest(cfg: Dict[str, Any], status: str) -> Dict[str, Any]:
    return {
        "run_id": f"{cfg['run_kind']}_{cfg['derivative_mode']}_{utc_stamp()}",
        "created_at_utc": utc_stamp(),
        "run_kind": cfg["run_kind"],
        "completion_status": status,
        "derivative_mode": cfg["derivative_mode"],
        "input_channels": int(cfg["input_channels"]),
        "checkpoint_path": None,
        "config": cfg,
        "data_split_source": str(cfg.get("team_meta_csv")),
        "data_split_hash": None,
        "artifacts": {},
        "metrics": {},
        "note": FAIRNESS_NOTE,
    }


def diagnostic_summary(diagnostic_artifacts: Dict[str, Path]) -> Dict[str, Any]:
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


def completed_manifest(
    cfg: Dict[str, Any],
    result: Dict[str, Any],
    context: ta.ExperimentContext,
    diagnostic_artifacts: Dict[str, Path],
) -> Dict[str, Any]:
    artifacts = {
        "latest": result.get("latest_path"),
        "checkpoint": result.get("checkpoint_path"),
        "history": result.get("history_path"),
        "config": str(Path(result["exp_dir"]) / "config.json"),
        "phase_diagnostics": str(diagnostic_artifacts["metrics"]),
        "phase_diagnostics_by_exercise": str(diagnostic_artifacts["per_exercise"]),
        "phase_confusion_matrix": str(diagnostic_artifacts["confusion_matrix"]),
        "phase_boundary_summary": str(diagnostic_artifacts["boundary_summary"]),
        "raw_predictions": str(diagnostic_artifacts["raw_predictions"]),
        "diagnostic_manifest": str(diagnostic_artifacts["manifest"]),
    }
    metric_summary = diagnostic_summary(diagnostic_artifacts)
    manifest = base_manifest(cfg, result.get("completion_status", "complete"))
    manifest.update(
        {
            "run_id": f"{cfg['run_kind']}_{cfg['derivative_mode']}_{utc_stamp()}",
            "input_channels": int(result["input_channels"]),
            "checkpoint_path": result.get("checkpoint_path"),
            "data_split_hash": ta.data_split_fingerprint(context.meta_df),
            "artifacts": artifacts,
            "metrics": {
                **{k: result.get(k) for k in result if k.startswith("video_") or k.startswith("mae_") or k.startswith("obo_")},
                **metric_summary,
            },
        }
    )
    return manifest


def failed_manifest(cfg: Dict[str, Any], error: BaseException) -> Dict[str, Any]:
    manifest = base_manifest(cfg, "failed")
    manifest["error"] = f"{type(error).__name__}: {error}"
    return manifest


def aggregate_row_from_manifest(
    manifest: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = manifest.get("config", {})
    row = {
        "derivative_mode": manifest.get("derivative_mode"),
        "input_channels": manifest.get("input_channels"),
        "run_kind": manifest.get("run_kind"),
        "completion_status": manifest.get("completion_status"),
        "checkpoint_path": manifest.get("checkpoint_path"),
        "phase_pooling": cfg.get("phase_pooling"),
        "phase_loss_alpha": cfg.get("phase_loss_alpha"),
        "epochs": cfg.get("epochs"),
        "batch": cfg.get("batch"),
        "output_root": cfg.get("output_root"),
        "note": manifest.get("note"),
    }
    if result:
        for key, value in result.items():
            if (
                key.startswith("video_")
                or key.startswith("mae_")
                or key.startswith("obo_")
                or key.startswith("phase_acc_")
                or key.startswith("phase_macro_f1_")
                or key in {
                    "exp_name",
                    "best_epoch",
                    "best_val_score",
                    "elapsed_min",
                    "num_params_total",
                    "phase_head_params",
                    "pooling_feature_dim",
                    "phase_acc",
                    "phase_macro_f1",
                    "ready_f1",
                    "down_f1",
                    "up_f1",
                }
            ):
                row[key] = value
    row.update(manifest.get("metrics") or {})
    if manifest.get("error"):
        row["error"] = manifest["error"]
    return row


def update_aggregates(output_root: Path, rows: List[Dict[str, Any]], per_exercise_rows: List[pd.DataFrame]) -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    results_csv = output_root / "derivative_input_results.csv"
    results_json = output_root / "derivative_input_results.json"
    per_ex_csv = output_root / "derivative_input_per_exercise.csv"

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        sort_col = "phase_macro_f1" if "phase_macro_f1" in result_df.columns else "video_phase_f1"
        if sort_col in result_df.columns:
            result_df = result_df.sort_values(sort_col, ascending=False, na_position="last")
    result_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    results_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

    if per_exercise_rows:
        per_ex_df = pd.concat(per_exercise_rows, ignore_index=True)
    else:
        per_ex_df = pd.DataFrame(columns=["derivative_mode", "input_channels", "mode", "type", "phase_acc", "phase_macro_f1"])
    per_ex_df.to_csv(per_ex_csv, index=False, encoding="utf-8-sig")

    if not result_df.empty:
        show_cols = [
            "derivative_mode",
            "input_channels",
            "run_kind",
            "completion_status",
            "best_epoch",
            "phase_acc",
            "phase_macro_f1",
            "ready_f1",
            "down_f1",
            "up_f1",
            "video_count_obo",
        ]
        print("\n[derivative_input_results summary]")
        print(result_df[[c for c in show_cols if c in result_df.columns]].to_string(index=False))
    return {
        "derivative_input_results_csv": results_csv,
        "derivative_input_results_json": results_json,
        "derivative_input_per_exercise_csv": per_ex_csv,
    }


def main(run_kind: str = RUN_KIND, variants: Optional[Iterable[str]] = None) -> Dict[str, Path]:
    run_kind = str(run_kind).lower()
    normalized_variants = [ta.normalize_derivative_mode(v) for v in (variants or DERIVATIVE_VARIANTS)]
    run_tag = RUN_TAG
    if FORCE_RETRAIN and FRESH_RERUN and not run_tag:
        run_tag = f"rerun_{utc_stamp()}"
    cfgs = [build_variant_cfg(mode, run_kind, run_tag=run_tag) for mode in normalized_variants]
    output_root = Path(cfgs[0]["output_root"])
    print(
        f"[phase_derivative_input_experiment] run_kind={run_kind} "
        f"epochs={cfgs[0]['epochs']} variants={normalized_variants} "
        f"force_retrain={cfgs[0]['force_retrain']} resume={cfgs[0]['resume']} "
        f"output_root={cfgs[0]['output_root']}",
        flush=True,
    )
    context = ta.prepare_context(cfgs[0], verbose=True, update_globals=True)

    aggregate_rows: List[Dict[str, Any]] = []
    per_exercise_frames: List[pd.DataFrame] = []
    failures: List[str] = []

    for cfg in cfgs:
        manifest_path = manifest_path_for(cfg)
        write_manifest(manifest_path, base_manifest(cfg, "started"))
        result: Optional[Dict[str, Any]] = None
        try:
            result = ta.run_one_experiment(cfg, context=context)
            diag_artifacts = diagnostics.run_diagnostics_for_checkpoint(
                result["checkpoint_path"],
                cfg=cfg,
                context=context,
                output_dir=Path(result["exp_dir"]),
                run_id=f"{run_kind}_{cfg['derivative_mode']}",
                smooth_window=SMOOTH_WINDOW,
                manifest_name="diagnostic_manifest.json",
            )
            manifest = completed_manifest(cfg, result, context, diag_artifacts)
            final_manifest_path = final_manifest_path_for_result(result)
            write_manifest(final_manifest_path, manifest)
            if final_manifest_path.resolve() != manifest_path.resolve():
                redirect = base_manifest(cfg, "redirected")
                redirect["final_manifest"] = str(final_manifest_path)
                write_manifest(manifest_path, redirect)
            aggregate_rows.append(aggregate_row_from_manifest(manifest, result=result))

            per_ex = pd.read_csv(diag_artifacts["per_exercise"])
            per_ex.insert(0, "derivative_mode", cfg["derivative_mode"])
            per_ex.insert(1, "input_channels", int(cfg["input_channels"]))
            per_ex.insert(2, "run_kind", run_kind)
            per_ex.insert(3, "exp_name", result["exp_name"])
            per_exercise_frames.append(per_ex)
        except Exception as exc:
            manifest = failed_manifest(cfg, exc)
            write_manifest(manifest_path, manifest)
            aggregate_rows.append(aggregate_row_from_manifest(manifest, result=result))
            failures.append(f"{cfg['derivative_mode']}: {type(exc).__name__}: {exc}")
            print(f"[FAILED] {cfg['derivative_mode']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

        update_aggregates(output_root, aggregate_rows, per_exercise_frames)

    artifacts = update_aggregates(output_root, aggregate_rows, per_exercise_frames)
    print(json.dumps({k: str(v) for k, v in artifacts.items()}, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError("Derivative input experiment failed for required variant(s): " + "; ".join(failures))
    return artifacts


if __name__ == "__main__":
    main()
