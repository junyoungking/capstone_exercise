"""Exercise-label conditioned MediaPipe+barbell joint-fusion experiment runner.

This runner keeps the prior MediaPipe33+barbell34 input contract and adds an
exercise-label-conditioned phase path through ``phase_head_type=exercise_attn``.
It compares that variant against a same-run joint-fusion MLP baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

try:  # `import model.phase_barbell_joint_fusion_exercise_label_experiment`
    from . import phase_barbell_joint_fusion_experiment as joint_fusion
    from . import train_ablation as ta
except ImportError:  # `python model/phase_barbell_joint_fusion_exercise_label_experiment.py`
    import phase_barbell_joint_fusion_experiment as joint_fusion  # type: ignore
    import train_ablation as ta  # type: ignore


RUN_KIND = "full"
OUTPUT_ROOT = Path("phase_experiments") / "barbell_yolo_world" / "joint_fusion_exercise_label"
COMPARISON_FILENAME = "phase_barbell_joint_fusion_exercise_label_comparison.csv"
MLP_BASELINE_ARTIFACT_LABEL = "same_run_joint_fusion_mlp"
EXERCISE_LABEL_ARTIFACT_LABEL = "same_run_joint_fusion_exercise_label"
HISTORICAL_JOINT_FUSION_COMPARISON_CSV = (
    Path("phase_experiments")
    / "barbell_yolo_world"
    / "joint_fusion"
    / "comparison"
    / "phase_barbell_joint_fusion_comparison.csv"
)


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


def build_joint_fusion_cfg(
    *,
    phase_head_type: str,
    run_kind: str,
    output_root: Path,
    epochs: Optional[int],
    max_videos_per_split_type: Optional[int],
    barbell_edge_policy: str,
    force_retrain: bool,
    fresh_rerun: bool,
    run_tag: Optional[str],
) -> Dict[str, Any]:
    cfg = joint_fusion.build_train_cfg(
        pose_backend=ta.POSE_BACKEND_MEDIAPIPE_BARBELL,
        run_kind=run_kind,
        output_root=output_root,
        epochs=epochs,
        max_videos_per_split_type=max_videos_per_split_type,
        barbell_edge_policy=barbell_edge_policy,
        force_retrain=force_retrain,
        fresh_rerun=fresh_rerun,
        run_tag=run_tag,
    )
    cfg = copy.deepcopy(cfg)
    cfg.update(
        {
            "phase_head_type": ta.normalize_phase_head_type(phase_head_type),
            "phase_pooling": "temporal_avg",
            "phase_conditioning": (
                ta.PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL
                if ta.normalize_phase_head_type(phase_head_type) == ta.PHASE_HEAD_EXERCISE_ATTN
                else ta.PHASE_CONDITIONING_NONE
            ),
            "output_root": str(Path(output_root) / str(run_kind).lower() / "mediapipe_barbell"),
        }
    )
    return ta.normalize_cfg(cfg)


def comparison_row(result: Mapping[str, Any], *, artifact_label: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "artifact_label": artifact_label,
        "exp_name": result.get("exp_name", artifact_label),
        "run_kind": result.get("run_kind"),
        "completion_status": result.get("completion_status"),
        "pose_backend": result.get("pose_backend"),
        "pose_graph_id": result.get("pose_graph_id"),
        "model_graph_id": result.get("model_graph_id"),
        "barbell_edge_policy": result.get("barbell_edge_policy"),
        "phase_head_type": result.get("phase_head_type"),
        "phase_conditioning": result.get("phase_conditioning"),
        "exercise_id_source": result.get("exercise_id_source"),
        "phase_label_scheme": result.get("phase_label_scheme"),
        "checkpoint_path": result.get("checkpoint_path"),
        "best_epoch": result.get("best_epoch"),
        "best_val_score": result.get("best_val_score"),
        "elapsed_min": result.get("elapsed_min"),
        "fresh_training_run": not bool(result.get("skipped", False)),
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


def historical_joint_fusion_row(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    if "artifact_label" in frame.columns:
        filtered = frame[frame["artifact_label"].astype(str) == "same_run_mediapipe_barbell_fusion_bar_direction"]
        if not filtered.empty:
            frame = filtered
    row = frame.iloc[0].to_dict()
    row.update(
        {
            "artifact_label": "historical_joint_fusion_mlp",
            "phase_head_type": ta.PHASE_HEAD_MLP,
            "phase_conditioning": "none",
            "exercise_id_source": ta.EXERCISE_ID_SOURCE_NONE,
            "fresh_training_run": False,
        }
    )
    return row


def write_comparison(output_root: Path, rows: List[Dict[str, Any]]) -> Dict[str, Path]:
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = comparison_dir / COMPARISON_FILENAME
    comparison_json = comparison_dir / "phase_barbell_joint_fusion_exercise_label_comparison.json"
    frame = pd.DataFrame(rows)
    if not frame.empty:
        sort_cols = [c for c in ["run_kind", "phase_conditioning", "artifact_label"] if c in frame.columns]
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
    barbell_edge_policy: str = ta.BARBELL_EDGE_POLICY_WRISTS,
    skip_diagnostics: bool = False,
    smooth_window: int = joint_fusion.SMOOTH_WINDOW,
    diagnostic_max_videos: Optional[int] = None,
    force_retrain: bool = False,
    fresh_rerun: bool = False,
    run_tag: Optional[str] = None,
    dry_run: bool = False,
    skip_historical: bool = False,
    historical_joint_fusion_comparison_csv: Path = HISTORICAL_JOINT_FUSION_COMPARISON_CSV,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_cfg = build_joint_fusion_cfg(
        phase_head_type=ta.PHASE_HEAD_MLP,
        run_kind=run_kind,
        output_root=output_root,
        epochs=epochs,
        max_videos_per_split_type=max_videos_per_split_type,
        barbell_edge_policy=barbell_edge_policy,
        force_retrain=force_retrain,
        fresh_rerun=fresh_rerun,
        run_tag=run_tag,
    )
    exercise_label_cfg = build_joint_fusion_cfg(
        phase_head_type=ta.PHASE_HEAD_EXERCISE_ATTN,
        run_kind=run_kind,
        output_root=output_root,
        epochs=epochs,
        max_videos_per_split_type=max_videos_per_split_type,
        barbell_edge_policy=barbell_edge_policy,
        force_retrain=force_retrain,
        fresh_rerun=fresh_rerun,
        run_tag=run_tag,
    )

    if dry_run:
        payload = {"dry_run": True, "configs": {"baseline_mlp": baseline_cfg, "exercise_label": exercise_label_cfg}}
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))
        return payload

    comparison_rows: List[Dict[str, Any]] = []
    artifacts: Dict[str, Any] = {}
    if not skip_historical:
        historical = historical_joint_fusion_row(Path(historical_joint_fusion_comparison_csv))
        if historical is not None:
            comparison_rows.append(historical)

    baseline_payload = joint_fusion.run_training_variant(
        cfg=baseline_cfg,
        output_root=output_root,
        artifact_label=MLP_BASELINE_ARTIFACT_LABEL,
        skip_diagnostics=skip_diagnostics,
        smooth_window=smooth_window,
        diagnostic_max_videos=diagnostic_max_videos,
    )
    comparison_rows.append(comparison_row(baseline_payload["result"], artifact_label=MLP_BASELINE_ARTIFACT_LABEL))
    artifacts["baseline_mlp"] = baseline_payload["manifest"]

    exercise_payload = joint_fusion.run_training_variant(
        cfg=exercise_label_cfg,
        output_root=output_root,
        artifact_label=EXERCISE_LABEL_ARTIFACT_LABEL,
        skip_diagnostics=skip_diagnostics,
        smooth_window=smooth_window,
        diagnostic_max_videos=diagnostic_max_videos,
    )
    comparison_rows.append(comparison_row(exercise_payload["result"], artifact_label=EXERCISE_LABEL_ARTIFACT_LABEL))
    artifacts["exercise_label"] = exercise_payload["manifest"]

    comparison_artifacts = write_comparison(output_root, comparison_rows)
    manifest = {
        "run_id": f"barbell_joint_fusion_exercise_label_{run_kind}_{utc_stamp()}",
        "created_at_utc": utc_stamp(),
        "completion_status": "complete",
        "run_kind": run_kind,
        "phase_conditioning": "ground_truth_exercise_label",
        "exercise_id_source": ta.EXERCISE_ID_SOURCE_GROUND_TRUTH_LABEL,
        "barbell_edge_policy": ta.normalize_barbell_edge_policy(barbell_edge_policy, ta.POSE_BACKEND_MEDIAPIPE_BARBELL),
        "configs": {"baseline_mlp": baseline_cfg, "exercise_label": exercise_label_cfg},
        "artifacts": {**{k: str(v) for k, v in comparison_artifacts.items()}, "variants": artifacts},
        "alignment_records": {
            "baseline_mlp": baseline_payload["manifest"].get("alignment_records", []),
            "exercise_label": exercise_payload["manifest"].get("alignment_records", []),
        },
        "comparison_rows": comparison_rows,
        "note": "The exercise-label variant is an oracle/label-conditioned experiment: it uses phase_head_type=exercise_attn and passes ground-truth exercise ids to the phase path during training, video eval, and diagnostics. Do not read its phase metrics as deployment end-to-end performance unless the exercise label is known upstream.",
    }
    write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps({"run_manifest": str(output_root / "run_manifest.json"), **{k: str(v) for k, v in comparison_artifacts.items()}}, ensure_ascii=False, indent=2))
    return manifest


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate joint-fusion phase models with and without exercise-label conditioning.")
    parser.add_argument("--mode", choices=["smoke", "full"], default=RUN_KIND, help="smoke=1-epoch limited split, full=60 epochs")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count for the selected mode.")
    parser.add_argument("--max-videos-per-split-type", type=int, default=None, help="Limit videos per split/exercise type.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Stable experiment output root.")
    parser.add_argument("--barbell-edge-policy", default=ta.BARBELL_EDGE_POLICY_WRISTS, choices=list(ta.ALLOWED_BARBELL_EDGE_POLICIES), help="Hybrid graph edge policy.")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip phase diagnostic CSV/markdown exports.")
    parser.add_argument("--smooth-window", type=int, default=joint_fusion.SMOOTH_WINDOW, help="Offline smoothing window for diagnostics.")
    parser.add_argument("--diagnostic-max-videos", type=int, default=None, help="Limit diagnostic prediction export videos.")
    parser.add_argument("--force-retrain", action="store_true", help="Do not resume/skip existing checkpoints.")
    parser.add_argument("--fresh-rerun", action="store_true", help="When forcing retrain, add a fresh rerun tag.")
    parser.add_argument("--run-tag", default=None, help="Explicit fresh rerun tag when --force-retrain --fresh-rerun are used.")
    parser.add_argument("--dry-run", action="store_true", help="Print normalized configs without preparing data or training.")
    parser.add_argument("--skip-historical", action="store_true", help="Do not include the previous joint-fusion comparison row.")
    parser.add_argument("--historical-joint-fusion-comparison-csv", default=str(HISTORICAL_JOINT_FUSION_COMPARISON_CSV), help="Previous joint-fusion comparison CSV.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    return run(
        run_kind=args.mode,
        output_root=Path(args.output_root),
        epochs=args.epochs,
        max_videos_per_split_type=args.max_videos_per_split_type,
        barbell_edge_policy=args.barbell_edge_policy,
        skip_diagnostics=bool(args.skip_diagnostics),
        smooth_window=int(args.smooth_window),
        diagnostic_max_videos=args.diagnostic_max_videos,
        force_retrain=bool(args.force_retrain),
        fresh_rerun=bool(args.fresh_rerun),
        run_tag=args.run_tag,
        dry_run=bool(args.dry_run),
        skip_historical=bool(args.skip_historical),
        historical_joint_fusion_comparison_csv=Path(args.historical_joint_fusion_comparison_csv),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
