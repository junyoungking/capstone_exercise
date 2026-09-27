"""PyCharm-runnable ST-GCN+MLP phase pooling ablation.

Default RUN_KIND is full so pressing Run starts the real 60-epoch experiment.
Set RUN_KIND = "smoke" only when you intentionally want a 1-epoch pipeline check.
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

try:  # `import model.phase_pooling_experiment`
    from . import phase_first_diagnostics as diagnostics
    from . import train_ablation as ta
except ImportError:  # `python model/phase_pooling_experiment.py` / PyCharm script mode
    import phase_first_diagnostics as diagnostics  # type: ignore
    import train_ablation as ta  # type: ignore


RUN_KIND = "full"  # use "smoke" only for a quick 1-epoch pipeline check
SMOKE_EPOCHS = 1
FULL_EPOCHS = 60
SMOKE_MAX_VIDEOS_PER_SPLIT_TYPE = 2
POOLING_VARIANTS = ["temporal_flatten", "last", "avg_last_concat"]
OUTPUT_ROOT = Path("phase_experiments") / "pooling_ablation"
SMOOTH_WINDOW = 5
FAIRNESS_NOTE = "temporal_flatten is not parameter-count matched; compare with pooling_feature_dim and phase_head_params."

# PyCharm에서 이 파일을 다시 실행할 때 기존 best/latest checkpoint 때문에
# eval-only로 끝나지 않도록 기본적으로 새 학습 run을 만든다.
# 기존 checkpoint는 덮어쓰지 않고 `..._rerun_YYYYMMDDTHHMMSS` 디렉터리에 저장한다.
FORCE_RETRAIN = True
FRESH_RERUN = True
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


def build_variant_cfg(phase_pooling: str, run_kind: str, run_tag: Optional[str] = None) -> Dict[str, Any]:
    if phase_pooling not in ta.POOLING_EXPERIMENT_VARIANTS:
        raise ValueError(f"unexpected pooling variant for this runner: {phase_pooling}")
    run_kind = str(run_kind).lower()
    if run_kind not in {"smoke", "full"}:
        raise ValueError(f"run_kind must be 'smoke' or 'full', got {run_kind}")
    if FORCE_RETRAIN and FRESH_RERUN and not run_tag:
        run_tag = f"rerun_{utc_stamp()}"
    cfg = copy.deepcopy(ta.DEFAULT_POOLING_EXPERIMENT)
    cfg.update(
        {
            "model_type": "mlp",
            "phase_pooling": phase_pooling,
            "hidden": 128,
            "clip_len": 16,
            "train_stride": 2,
            "dropout": 0.3,
            "aug": True,
            "epochs": SMOKE_EPOCHS if run_kind == "smoke" else FULL_EPOCHS,
            "batch": 256,
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
    exp_name = cfg.get("exp_name") or ta.make_exp_name(cfg)
    return Path(cfg["output_root"]) / str(exp_name) / "run_manifest.json"


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def base_manifest(cfg: Dict[str, Any], status: str) -> Dict[str, Any]:
    return {
        "run_id": f"{cfg['run_kind']}_{cfg['phase_pooling']}_{utc_stamp()}",
        "created_at_utc": utc_stamp(),
        "run_kind": cfg["run_kind"],
        "completion_status": status,
        "phase_pooling": cfg["phase_pooling"],
        "pooling_feature_dim": None,
        "num_params_total": None,
        "phase_head_params": None,
        "checkpoint_path": None,
        "config": cfg,
        "data_split_source": str(cfg.get("team_meta_csv")),
        "data_split_hash": None,
        "artifacts": {},
        "note": FAIRNESS_NOTE,
    }


def completed_manifest(cfg: Dict[str, Any], result: Dict[str, Any], context: ta.ExperimentContext, diagnostic_artifacts: Dict[str, Path]) -> Dict[str, Any]:
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
    manifest = base_manifest(cfg, result.get("completion_status", "complete"))
    manifest.update(
        {
            "run_id": f"{cfg['run_kind']}_{cfg['phase_pooling']}_{utc_stamp()}",
            "pooling_feature_dim": int(result["pooling_feature_dim"]),
            "num_params_total": int(result["num_params_total"]),
            "phase_head_params": int(result["phase_head_params"]),
            "checkpoint_path": result.get("checkpoint_path"),
            "data_split_hash": ta.data_split_fingerprint(context.meta_df),
            "artifacts": artifacts,
            "metrics": {k: result.get(k) for k in result if k.startswith("video_") or k.startswith("mae_") or k.startswith("obo_")},
        }
    )
    return manifest


def failed_manifest(cfg: Dict[str, Any], error: BaseException) -> Dict[str, Any]:
    manifest = base_manifest(cfg, "failed")
    manifest["error"] = f"{type(error).__name__}: {error}"
    return manifest


def update_aggregates(output_root: Path, rows: List[Dict[str, Any]], per_exercise_rows: List[pd.DataFrame]) -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    results_csv = output_root / "pooling_results.csv"
    results_json = output_root / "pooling_results.json"
    per_ex_csv = output_root / "pooling_per_exercise.csv"

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        if "video_phase_f1" in result_df.columns:
            result_df = result_df.sort_values("video_phase_f1", ascending=False, na_position="last")
    result_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    results_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

    if per_exercise_rows:
        per_ex_df = pd.concat(per_exercise_rows, ignore_index=True)
    else:
        per_ex_df = pd.DataFrame(columns=["phase_pooling", "mode", "type", "phase_acc", "phase_macro_f1"])
    per_ex_df.to_csv(per_ex_csv, index=False, encoding="utf-8-sig")
    if not result_df.empty:
        show_cols = [
            "phase_pooling",
            "run_kind",
            "completion_status",
            "best_epoch",
            "best_val_score",
            "video_phase_f1",
            "pooling_feature_dim",
        ]
        print("\n[pooling_results summary]")
        print(result_df[[c for c in show_cols if c in result_df.columns]].to_string(index=False))
    return {"pooling_results_csv": results_csv, "pooling_results_json": results_json, "pooling_per_exercise_csv": per_ex_csv}


def aggregate_row_from_manifest(manifest: Dict[str, Any], result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = manifest.get("config", {})
    row = {
        "phase_pooling": manifest.get("phase_pooling"),
        "run_kind": manifest.get("run_kind"),
        "completion_status": manifest.get("completion_status"),
        "pooling_feature_dim": manifest.get("pooling_feature_dim"),
        "num_params_total": manifest.get("num_params_total"),
        "phase_head_params": manifest.get("phase_head_params"),
        "checkpoint_path": manifest.get("checkpoint_path"),
        "phase_loss_alpha": cfg.get("phase_loss_alpha"),
        "epochs": cfg.get("epochs"),
        "batch": cfg.get("batch"),
        "output_root": cfg.get("output_root"),
        "note": manifest.get("note"),
    }
    if result:
        for key, value in result.items():
            if key.startswith("video_") or key.startswith("mae_") or key.startswith("obo_") or key in {"exp_name", "best_epoch", "best_val_score", "elapsed_min"}:
                row[key] = value
    if manifest.get("error"):
        row["error"] = manifest["error"]
    return row


def main(run_kind: str = RUN_KIND, variants: Optional[Iterable[str]] = None) -> Dict[str, Path]:
    run_kind = str(run_kind).lower()
    variants = list(variants or POOLING_VARIANTS)
    output_root = Path(OUTPUT_ROOT / run_kind)
    run_tag = RUN_TAG
    if FORCE_RETRAIN and FRESH_RERUN and not run_tag:
        run_tag = f"rerun_{utc_stamp()}"
    cfgs = [build_variant_cfg(pooling, run_kind, run_tag=run_tag) for pooling in variants]
    print(
        f"[phase_pooling_experiment] run_kind={run_kind} "
        f"epochs={cfgs[0]['epochs']} variants={variants} "
        f"force_retrain={cfgs[0]['force_retrain']} output_root={cfgs[0]['output_root']}",
        flush=True,
    )
    context = ta.prepare_context(cfgs[0], verbose=True, update_globals=True)

    aggregate_rows: List[Dict[str, Any]] = []
    per_exercise_frames: List[pd.DataFrame] = []

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
                run_id=f"{run_kind}_{cfg['phase_pooling']}",
                smooth_window=SMOOTH_WINDOW,
                manifest_name="diagnostic_manifest.json",
            )
            manifest = completed_manifest(cfg, result, context, diag_artifacts)
            write_manifest(manifest_path, manifest)
            aggregate_rows.append(aggregate_row_from_manifest(manifest, result=result))
            per_ex = pd.read_csv(diag_artifacts["per_exercise"])
            per_ex.insert(0, "phase_pooling", cfg["phase_pooling"])
            per_ex.insert(1, "run_kind", run_kind)
            per_ex.insert(2, "exp_name", result["exp_name"])
            per_exercise_frames.append(per_ex)
        except Exception as exc:
            manifest = failed_manifest(cfg, exc)
            write_manifest(manifest_path, manifest)
            aggregate_rows.append(aggregate_row_from_manifest(manifest, result=result))
            print(f"[FAILED] {cfg['phase_pooling']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

        update_aggregates(output_root, aggregate_rows, per_exercise_frames)

    artifacts = update_aggregates(output_root, aggregate_rows, per_exercise_frames)
    print(json.dumps({k: str(v) for k, v in artifacts.items()}, ensure_ascii=False, indent=2))
    return artifacts


if __name__ == "__main__":
    main()
