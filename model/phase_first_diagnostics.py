"""Phase-first diagnostic artifact generator.

Can run as a script for existing checkpoints and can also be called by
phase_pooling_experiment.py after each training variant.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:  # works for `import model.phase_first_diagnostics`
    from . import phase_metrics
    from . import train_ablation as ta
except ImportError:  # works for `python model/phase_first_diagnostics.py`
    import phase_metrics  # type: ignore
    import train_ablation as ta  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "phase_experiments"
REPORT_DIR = REPO_ROOT / ".omx" / "reports"


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


def run_git(args: List[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.STDOUT, text=True, encoding="utf-8").strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"<git unavailable: {exc}>"


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def cfg_from_row(row: pd.Series) -> Dict[str, Any]:
    return ta.normalize_cfg(
        {
            "model_type": str(row.get("model_type", "mlp")),
            "phase_head_type": str(row.get("phase_head_type", "mlp")),
            "phase_conditioning": row.get("phase_conditioning", None),
            "phase_pooling": str(row.get("phase_pooling", "temporal_avg")),
            "derivative_mode": str(row.get("derivative_mode", "pose")),
            "phase_label_scheme": str(row.get("phase_label_scheme", "as_labeled")),
            "pose_backend": str(row.get("pose_backend", "mediapipe")),
            "joint_subset": str(row.get("joint_subset", "all")),
            "barbell_edge_policy": str(row.get("barbell_edge_policy", ta.BARBELL_EDGE_POLICY_NONE)),
            "hidden": int(row.get("hidden", 128)),
            "lstm_layers": int(row.get("lstm_layers", 1)),
            "clip_len": int(row.get("clip_len", 16)),
            "train_stride": int(row.get("train_stride", 2)),
            "dropout": float(row.get("dropout", 0.3)),
            "aug": coerce_bool(row.get("aug", True)),
            "phase_loss_alpha": float(row.get("phase_loss_alpha", row.get("pw", 2.0))),
        }
    )


def cfg_from_checkpoint(ckpt: Dict[str, Any], fallback_row: Optional[pd.Series] = None) -> Dict[str, Any]:
    cfg = ckpt.get("cfg")
    if isinstance(cfg, dict):
        return ta.normalize_cfg(cfg)
    if fallback_row is not None:
        return cfg_from_row(fallback_row)
    raise ValueError("checkpoint has no cfg and no fallback row was provided")


def load_results_table() -> pd.DataFrame:
    if not ta.ABLATION_LOG_CSV.exists():
        raise FileNotFoundError(f"ablation result CSV not found: {ta.ABLATION_LOG_CSV}")
    return pd.read_csv(ta.ABLATION_LOG_CSV)


def find_checkpoint(exp_name: Optional[str] = None, checkpoint: Optional[Path] = None) -> Tuple[Path, Dict[str, Any], Optional[pd.Series]]:
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=ta.DEVICE)
        return checkpoint, cfg_from_checkpoint(ckpt), None

    results = load_results_table()
    if exp_name:
        matches = results[results["exp_name"] == exp_name]
        if matches.empty:
            raise ValueError(f"exp_name not found in {ta.ABLATION_LOG_CSV}: {exp_name}")
        row = matches.iloc[0]
    else:
        sort_col = "video_phase_f1" if "video_phase_f1" in results.columns else "best_val_score"
        row = results.sort_values(sort_col, ascending=False).iloc[0]
        exp_name = str(row["exp_name"])

    exp_dir = ta.ABLATION_ROOT / str(exp_name)
    bests = sorted(exp_dir.glob("best_ep*.pt"))
    if not bests:
        latest = exp_dir / "latest.pt"
        if latest.exists():
            bests = [latest]
    if not bests:
        raise FileNotFoundError(f"no checkpoint found under {exp_dir}")
    checkpoint_path = bests[-1]
    ckpt = torch.load(checkpoint_path, map_location=ta.DEVICE)
    return checkpoint_path, cfg_from_checkpoint(ckpt, row), row


def boundary_distance(frame_idx: int, reps: Iterable[Tuple[int, int, int]]) -> float:
    points: List[int] = []
    for s, b, f in reps:
        points.extend([int(s), int(b), int(f)])
    if not points:
        return float("nan")
    return float(min(abs(int(frame_idx) - p) for p in points))


def load_model(checkpoint_path: Path, cfg: Dict[str, Any]) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=ta.DEVICE)
    model = ta.build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def prediction_records(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    smooth_window: int,
    max_videos: Optional[int] = None,
    context: Optional[ta.ExperimentContext] = None,
) -> pd.DataFrame:
    context = context or ta.prepare_context(cfg, verbose=False, update_globals=True)
    records: List[Dict[str, Any]] = []
    meta = context.val_meta.reset_index(drop=True)
    if max_videos is not None:
        meta = meta.head(max_videos)

    norm_cfg = ta.normalize_cfg(cfg)
    head_type = ta.normalize_phase_head_type(norm_cfg.get("phase_head_type", ta.PHASE_HEAD_MLP))
    conditioning = ta.normalize_phase_conditioning(norm_cfg.get("phase_conditioning"), head_type)
    exercise_id_source = ta.exercise_id_source_for_conditioning(conditioning)
    target_scheme = ta.normalize_phase_label_scheme(norm_cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED))
    edge_policy = ta.normalize_barbell_edge_policy(norm_cfg.get("barbell_edge_policy"), norm_cfg.get("pose_backend"))

    for _, row in meta.iterrows():
        load_result = ta.load_model_keypoints_from_row(row, norm_cfg)
        kpts = load_result.kpts
        T = min(int(row.get("T_used", row.get("T", len(kpts)))), len(kpts))
        out = ta.predict_from_kpts(
            model,
            kpts[:T],
            clip_len=int(norm_cfg["clip_len"]),
            stride=int(norm_cfg["train_stride"]),
            smooth_window=smooth_window,
            derivative_mode=str(norm_cfg["derivative_mode"]),
            pose_backend=str(norm_cfg["pose_backend"]),
            joint_subset=str(norm_cfg["joint_subset"]),
            phase_head_type=head_type,
            barbell_box_features=kpts[:T] if ta.uses_direct_barbell_box_head(head_type) else None,
            barbell_edge_policy=edge_policy,
            model_ready_features=True,
            exercise_id=int(row["cls"]) if conditioning == ta.PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL else None,
            phase_conditioning=conditioning,
            allow_predicted_action_conditioning=conditioning == ta.PHASE_CONDITIONING_PREDICTED_ACTION,
        )
        gt_key = (row["type"], row["name"])
        reps = context.labels[gt_key]["reps"]
        gt_phase = ta.make_phase_target(
            T,
            reps,
            exercise_type=row["type"],
            phase_label_scheme=target_scheme,
        )
        valid_mask = np.asarray(out["valid_mask"], dtype=bool)

        for frame_idx in np.where(valid_mask)[0]:
            probs = np.asarray(out["phase_prob"][frame_idx], dtype=float)
            rec = {
                "type": row["type"],
                "name": row["name"],
                "split": row.get("split", "val"),
                "frame_idx": int(frame_idx),
                "clip_len": int(norm_cfg["clip_len"]),
                "stride": int(norm_cfg["train_stride"]),
                "phase_conditioning": conditioning,
                "exercise_id_source": exercise_id_source,
                "conditioning_exercise_id": int(row["cls"]) if conditioning == ta.PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL else "",
                "gt_phase": int(gt_phase[frame_idx]),
                "gt_phase_name": ta.PHASE_NAMES[int(gt_phase[frame_idx])],
                "pred_phase_raw": int(out["phase_raw"][frame_idx]),
                "pred_phase_raw_name": ta.PHASE_NAMES[int(out["phase_raw"][frame_idx])],
                "pred_phase_offline_smooth": int(out["phase_smooth"][frame_idx]),
                "pred_phase_offline_smooth_name": ta.PHASE_NAMES[int(out["phase_smooth"][frame_idx])],
                "phase_prob_ready": float(probs[ta.PHASE_READY]),
                "phase_prob_down": float(probs[ta.PHASE_DOWN]),
                "phase_prob_up": float(probs[ta.PHASE_UP]),
                "boundary_distance": boundary_distance(int(frame_idx), reps),
                "action_pred": out["pred_class"],
                "action_prob_squat": float(out["action_probs"][ta.CLASS_TO_ID["squat"]]),
                "action_prob_benchpress": float(out["action_probs"][ta.CLASS_TO_ID["benchpress"]]),
                "action_prob_deadlift": float(out["action_probs"][ta.CLASS_TO_ID["deadlift"]]),
            }
            records.append(rec)
    if not records:
        raise RuntimeError("no prediction records were generated")
    return pd.DataFrame(records)


def write_report(
    report_path: Path,
    run_id: str,
    checkpoint_path: Path,
    cfg: Dict[str, Any],
    summaries: pd.DataFrame,
    per_exercise: pd.DataFrame,
    boundary: pd.DataFrame,
    artifacts: Dict[str, Path],
    historical_row: Optional[pd.Series],
    max_videos: Optional[int],
) -> None:
    def md_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_empty_"
        shown = frame.copy()
        for col in shown.columns:
            if pd.api.types.is_float_dtype(shown[col]):
                shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        headers = [str(c) for c in shown.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in shown.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in shown.columns) + " |")
        return "\n".join(lines)

    raw = summaries[summaries["mode"] == "raw"].iloc[0]
    source_line = "- Source row: explicit checkpoint path; no ablation CSV row was used."
    if historical_row is not None:
        source_line = f"- Source historical row: `{historical_row.get('exp_name')}`."
    max_video_note = f"- Max videos limit: {max_videos}." if max_videos else "- Max videos limit: none."
    report = f"""# Phase Diagnostic Report — {run_id}

## Provenance

- Checkpoint: `{checkpoint_path}`
- Config: `{json.dumps(cfg, ensure_ascii=False, default=json_default)}`
- Phase conditioning: `{cfg.get("phase_conditioning")}` / exercise_id_source: `{cfg.get("exercise_id_source")}`.
{source_line}
{max_video_note}

## Global Phase Metrics

{md_table(summaries)}

## Per-Exercise Phase Metrics

{md_table(per_exercise)}

## Boundary Error Summary

{md_table(boundary)}

## Summary

- Raw phase accuracy: {raw['phase_acc']:.4f}
- Raw phase macro-F1: {raw['phase_macro_f1']:.4f}
- Offline smoothing is reported separately and should not be treated as a causal real-time claim.

## Artifact References

"""
    for label, path in artifacts.items():
        report += f"- {label}: `{path}`\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def run_diagnostics_for_checkpoint(
    checkpoint_path: Path | str,
    cfg: Optional[Dict[str, Any]] = None,
    context: Optional[ta.ExperimentContext] = None,
    output_dir: Optional[Path | str] = None,
    output_root: Optional[Path | str] = None,
    run_id: Optional[str] = None,
    smooth_window: int = 5,
    max_videos: Optional[int] = None,
    historical_row: Optional[pd.Series] = None,
    manifest_name: str = "diagnostic_manifest.json",
    trained_cfg: Optional[Dict[str, Any]] = None,
    artifact_label: Optional[str] = None,
) -> Dict[str, Path]:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=ta.DEVICE)
    norm_cfg = ta.normalize_cfg(cfg or cfg_from_checkpoint(ckpt))
    norm_trained_cfg = ta.normalize_cfg(trained_cfg) if trained_cfg is not None else None
    run_id = run_id or f"phase_diag_{utc_stamp()}"
    if output_dir is None:
        out_dir = Path(output_root or DEFAULT_OUTPUT_ROOT) / run_id
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    context = context or ta.prepare_context(norm_cfg, verbose=False, update_globals=True)
    model = load_model(checkpoint_path, norm_cfg)
    preds = prediction_records(model, norm_cfg, smooth_window=smooth_window, max_videos=max_videos, context=context)
    metric_artifacts = phase_metrics.combined_metric_artifacts(preds, ta.PHASE_NAMES)

    raw_path = out_dir / "raw_phase_predictions.csv"
    metrics_path = out_dir / "phase_diagnostics.csv"
    baseline_metrics_path = out_dir / "baseline_metrics.csv"
    per_ex_path = out_dir / "phase_diagnostics_by_exercise.csv"
    cm_path = out_dir / "phase_confusion_matrix.csv"
    boundary_path = out_dir / "phase_boundary_summary.csv"
    baseline_config_path = out_dir / "baseline_config.json"
    manifest_path = out_dir / manifest_name
    report_path = REPORT_DIR / f"phase-diagnostic-{run_id}.md"

    preds.to_csv(raw_path, index=False, encoding="utf-8-sig")
    metric_artifacts["summaries"].to_csv(metrics_path, index=False, encoding="utf-8-sig")
    metric_artifacts["summaries"].to_csv(baseline_metrics_path, index=False, encoding="utf-8-sig")
    metric_artifacts["per_exercise"].to_csv(per_ex_path, index=False, encoding="utf-8-sig")
    metric_artifacts["confusion"].to_csv(cm_path, index=True, encoding="utf-8-sig")
    metric_artifacts["boundary"].to_csv(boundary_path, index=False, encoding="utf-8-sig")

    artifacts = {
        "raw_predictions": raw_path,
        "metrics": metrics_path,
        "per_exercise": per_ex_path,
        "confusion_matrix": cm_path,
        "boundary_summary": boundary_path,
        "baseline_metrics": baseline_metrics_path,
        "baseline_config": baseline_config_path,
        "report": report_path,
    }
    baseline_config = {
        "run_id": run_id,
        "checkpoint": str(checkpoint_path),
        "config": norm_cfg,
        "eval_config": norm_cfg,
        "phase_conditioning": norm_cfg.get("phase_conditioning"),
        "exercise_id_source": norm_cfg.get("exercise_id_source"),
        "trained_config": norm_trained_cfg,
        "eval_phase_label_scheme": norm_cfg.get("phase_label_scheme"),
        "trained_phase_label_scheme": (
            norm_trained_cfg.get("phase_label_scheme") if norm_trained_cfg is not None else None
        ),
        "artifact_label": artifact_label,
        "source": "checkpoint_diagnostic_export",
        "fresh_training_run": False,
    }
    baseline_config_path.write_text(json.dumps(baseline_config, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_at_utc": utc_stamp(),
        "script": str(Path(__file__).resolve()),
        "python_executable": sys.executable,
        "checkpoint": str(checkpoint_path),
        "config": norm_cfg,
        "eval_config": norm_cfg,
        "phase_conditioning": norm_cfg.get("phase_conditioning"),
        "exercise_id_source": norm_cfg.get("exercise_id_source"),
        "trained_config": norm_trained_cfg,
        "eval_phase_label_scheme": norm_cfg.get("phase_label_scheme"),
        "trained_phase_label_scheme": (
            norm_trained_cfg.get("phase_label_scheme") if norm_trained_cfg is not None else None
        ),
        "artifact_label": artifact_label,
        "fresh_training_run": False,
        "diagnostic_export_fresh": True,
        "max_videos": max_videos,
        "git_rev": run_git(["rev-parse", "--short", "HEAD"]),
        "git_status_short": run_git(["status", "--short"]),
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    write_report(report_path, run_id, checkpoint_path, norm_cfg, metric_artifacts["summaries"], metric_artifacts["per_exercise"], metric_artifacts["boundary"], artifacts, historical_row, max_videos)
    return {"output_dir": out_dir, "manifest": manifest_path, **artifacts}


def run_diagnostics(args: argparse.Namespace) -> Dict[str, Path]:
    run_id = args.run_id or f"phase_diag_{utc_stamp()}"
    checkpoint_path, cfg, historical_row = find_checkpoint(
        exp_name=args.exp_name,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
    )
    if getattr(args, "phase_label_scheme", None):
        cfg = {**cfg, "phase_label_scheme": args.phase_label_scheme}
    return run_diagnostics_for_checkpoint(
        checkpoint_path,
        cfg=cfg,
        output_root=Path(args.output_root),
        run_id=run_id,
        smooth_window=args.smooth_window,
        max_videos=args.max_videos,
        historical_row=historical_row,
        manifest_name="run_manifest.json",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate phase-first diagnostics from a trained checkpoint.")
    parser.add_argument("--run-id", default=None, help="Run id used under phase_experiments/ and .omx/reports/.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Artifact root directory.")
    parser.add_argument("--exp-name", default=None, help="Existing ablation experiment name to inspect. Defaults to best video_phase_f1 row.")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path. Overrides --exp-name.")
    parser.add_argument("--phase-label-scheme", default=None, help="Override eval target label scheme.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Offline smoothing window for separated comparison.")
    parser.add_argument("--max-videos", type=int, default=None, help="Optional smoke limit for quick validation.")
    args = parser.parse_args(argv)
    artifacts = run_diagnostics(args)
    print(json.dumps({k: str(v) for k, v in artifacts.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
