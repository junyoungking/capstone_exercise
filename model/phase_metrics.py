"""Pure phase-metric helpers for ST-GCN phase diagnostics.

This module deliberately does not import train_ablation.  Callers pass phase
names and column names explicitly so the helpers can be reused by training,
diagnostics, and experiment runners without circular side effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def classification_metric_block(
    df: pd.DataFrame,
    phase_names: Iterable[str],
    true_col: str = "gt_phase",
    pred_col: str = "pred_phase_raw",
    mode: str = "raw",
) -> Dict[str, Any]:
    names = list(phase_names)
    labels = list(range(len(names)))
    y_true = df[true_col].astype(int).to_numpy()
    y_pred = df[pred_col].astype(int).to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    summary: Dict[str, Any] = {
        "mode": mode,
        "n": int(len(df)),
        "phase_acc": float(accuracy_score(y_true, y_pred)) if len(df) else float("nan"),
        "phase_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(df) else float("nan"),
    }
    for i, name in enumerate(names):
        summary[f"{name}_precision"] = float(precision[i])
        summary[f"{name}_recall"] = float(recall[i])
        summary[f"{name}_f1"] = float(f1[i])
        summary[f"{name}_support"] = int(support[i])
    return summary


def per_exercise_metric_block(
    df: pd.DataFrame,
    phase_names: Iterable[str],
    true_col: str = "gt_phase",
    pred_col: str = "pred_phase_raw",
    mode: str = "raw",
    exercise_col: str = "type",
) -> pd.DataFrame:
    rows = []
    names = list(phase_names)
    labels = list(range(len(names)))
    for exercise, sub in df.groupby(exercise_col):
        y_true = sub[true_col].astype(int).to_numpy()
        y_pred = sub[pred_col].astype(int).to_numpy()
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
        row: Dict[str, Any] = {
            "mode": mode,
            exercise_col: exercise,
            "n": int(len(sub)),
            "phase_acc": float(accuracy_score(y_true, y_pred)) if len(sub) else float("nan"),
            "phase_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(sub) else float("nan"),
        }
        for i, phase_name in enumerate(names):
            row[f"{phase_name}_precision"] = float(precision[i])
            row[f"{phase_name}_recall"] = float(recall[i])
            row[f"{phase_name}_f1"] = float(f1[i])
            row[f"{phase_name}_support"] = int(support[i])
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_matrix_block(
    df: pd.DataFrame,
    phase_names: Iterable[str],
    true_col: str = "gt_phase",
    pred_col: str = "pred_phase_raw",
    mode: str = "raw",
) -> pd.DataFrame:
    names = list(phase_names)
    labels = list(range(len(names)))
    cm = confusion_matrix(df[true_col].astype(int), df[pred_col].astype(int), labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"gt_{p}" for p in names], columns=[f"pred_{p}" for p in names])
    cm_df.insert(0, "mode", mode)
    return cm_df


def boundary_summary(
    df: pd.DataFrame,
    true_col: str = "gt_phase",
    pred_col: str = "pred_phase_raw",
    distance_col: str = "boundary_distance",
    mode: str = "raw",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"mode": mode}
    if distance_col not in df.columns or df.empty:
        out.update({"error_rate_near_3": float("nan"), "error_rate_far_10": float("nan"), "mean_boundary_distance_errors": float("nan")})
        return out
    work = df.copy()
    work["is_error"] = work[true_col].astype(int) != work[pred_col].astype(int)
    finite = work[np.isfinite(work[distance_col])]
    if finite.empty:
        out.update({"error_rate_near_3": float("nan"), "error_rate_far_10": float("nan"), "mean_boundary_distance_errors": float("nan")})
        return out
    near = finite[finite[distance_col] <= 3]
    far = finite[finite[distance_col] >= 10]
    out["error_rate_near_3"] = float(near["is_error"].mean()) if len(near) else float("nan")
    out["error_rate_far_10"] = float(far["is_error"].mean()) if len(far) else float("nan")
    out["mean_boundary_distance_errors"] = float(finite.loc[finite["is_error"], distance_col].mean()) if finite["is_error"].any() else float("nan")
    return out


def combined_metric_artifacts(
    df: pd.DataFrame,
    phase_names: Iterable[str],
    raw_col: str = "pred_phase_raw",
    smooth_col: str = "pred_phase_offline_smooth",
    true_col: str = "gt_phase",
) -> Dict[str, pd.DataFrame]:
    modes = [("raw", raw_col)]
    if smooth_col in df.columns:
        modes.append(("offline_smooth", smooth_col))
    summaries = []
    per_exercise = []
    confusion = []
    boundary = []
    for mode, pred_col in modes:
        summaries.append(classification_metric_block(df, phase_names, true_col=true_col, pred_col=pred_col, mode=mode))
        per_exercise.append(per_exercise_metric_block(df, phase_names, true_col=true_col, pred_col=pred_col, mode=mode))
        confusion.append(confusion_matrix_block(df, phase_names, true_col=true_col, pred_col=pred_col, mode=mode))
        boundary.append(boundary_summary(df, true_col=true_col, pred_col=pred_col, mode=mode))
    return {
        "summaries": pd.DataFrame(summaries),
        "per_exercise": pd.concat(per_exercise, ignore_index=True) if per_exercise else pd.DataFrame(),
        "confusion": pd.concat(confusion, ignore_index=True) if confusion else pd.DataFrame(),
        "boundary": pd.DataFrame(boundary),
    }


def write_metric_artifacts(
    df: pd.DataFrame,
    phase_names: Iterable[str],
    output_dir: Path | str,
    *,
    raw_predictions_name: str = "raw_phase_predictions.csv",
    metrics_name: str = "phase_diagnostics.csv",
    per_exercise_name: str = "phase_diagnostics_by_exercise.csv",
    confusion_name: str = "phase_confusion_matrix.csv",
    boundary_name: str = "phase_boundary_summary.csv",
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = combined_metric_artifacts(df, phase_names)
    paths = {
        "raw_predictions": output_dir / raw_predictions_name,
        "metrics": output_dir / metrics_name,
        "per_exercise": output_dir / per_exercise_name,
        "confusion_matrix": output_dir / confusion_name,
        "boundary_summary": output_dir / boundary_name,
    }
    df.to_csv(paths["raw_predictions"], index=False, encoding="utf-8-sig")
    artifacts["summaries"].to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    artifacts["per_exercise"].to_csv(paths["per_exercise"], index=False, encoding="utf-8-sig")
    artifacts["confusion"].to_csv(paths["confusion_matrix"], index=True, encoding="utf-8-sig")
    artifacts["boundary"].to_csv(paths["boundary_summary"], index=False, encoding="utf-8-sig")
    return paths
