"""Run high-quality phase-head-only auxiliary-input ablations.

This runner keeps the ST-GCN backbone input on MediaPipe pose only, while
injecting extra signals only at the phase classifier feature vector.  It is for
the ablation requested after the current best high-quality setting:

- quality filter: high/상 only
- metadata source: MediaPipe + barbell caches, aligned by min frame count
- backbone graph/input: MediaPipe33 pose only (`joint_subset=pose_only`)
- model: LSTM temporal head by default, matching the best-score high-quality
  architecture comparison. Use `--model-types mlp` if you want the best
  video-phase-F1 MLP variant instead.
- phase label: bar_direction
- phase head: exercise_attn with ground-truth exercise conditioning
- tested phase-head aux inputs: wrist, barbell, acceleration

Examples
--------
Dry-run config check:

    python model/run_quality_high_phase_aux_inputs.py --dry-run --allow-cpu

Run the three requested variants:

    python model/run_quality_high_phase_aux_inputs.py

Run only the combined aux variant:

    python model/run_quality_high_phase_aux_inputs.py --variants wrist+barbell+acceleration
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import run_quality_filtered_ablation as quality_runner  # noqa: E402
from model import train_ablation as ta  # noqa: E402


VARIANT_ALIASES = {
    "wrist": "wrist",
    "wrists": "wrist",
    "wrist_only": "wrist",
    "bar": "barbell",
    "barbell": "barbell",
    "barbell_coord": "barbell",
    "barbell_coords": "barbell",
    "acc": "acceleration",
    "accel": "acceleration",
    "acceleration": "acceleration",
    "all": "wrist,barbell,acceleration",
    "combined": "wrist,barbell,acceleration",
}


def default_output_root() -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return PROJECT_ROOT / "phase_experiments" / "quality_high_phase_aux_inputs" / stamp


def parse_variants(raw: str) -> List[str]:
    variants: List[str] = []
    for part in raw.replace(";", ",").split(","):
        token = part.strip().lower().replace("-", "_").replace(" ", "_")
        if not token:
            continue
        token = VARIANT_ALIASES.get(token, token)
        if "+" in token:
            sub_tokens = [VARIANT_ALIASES.get(x.strip(), x.strip()) for x in token.split("+") if x.strip()]
            token = ",".join(sub_tokens)
        # Validate through train_ablation's normalizer.
        normalized = ",".join(ta.normalize_phase_aux_inputs(token))
        if not normalized:
            raise ValueError(f"empty phase aux variant from {part!r}")
        if normalized not in variants:
            variants.append(normalized)
    if not variants:
        raise ValueError("--variants must not be empty")
    return variants


def build_runner_argv(args: argparse.Namespace, variant: str, output_root: Path) -> List[str]:
    tag = args.fresh_run_tag or "phase_aux"
    variant_tag = variant.replace(",", "_")
    argv = [
        "--quality-xlsx",
        str(args.quality_xlsx),
        "--quality-levels",
        "high",
        "--quality-name",
        args.quality_name,
        "--output-root",
        str(output_root),
        "--model-types",
        args.model_types,
        "--epochs",
        str(args.epochs),
        "--pose-backend",
        ta.POSE_BACKEND_MEDIAPIPE_BARBELL,
        "--joint-subset",
        ta.JOINT_SUBSET_POSE_ONLY,
        "--phase-label-scheme",
        ta.PHASE_LABEL_SCHEME_BAR_DIRECTION,
        "--phase-head-type",
        args.phase_head_type,
        "--phase-conditioning",
        ta.PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL,
        "--barbell-edge-policy",
        ta.BARBELL_EDGE_POLICY_WRISTS,
        "--derivative-mode",
        "pose",
        "--phase-aux-inputs",
        variant,
        "--clip-len",
        str(args.clip_len),
        "--train-stride",
        str(args.train_stride),
        "--batch",
        str(args.batch),
        "--num-workers",
        str(args.num_workers),
        "--fresh-run-tag",
        f"{tag}_{variant_tag}",
    ]
    if args.force_retrain:
        argv.append("--force-retrain")
    if args.allow_cpu:
        argv.append("--allow-cpu")
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-xlsx", type=Path, default=PROJECT_ROOT / "dataset 분류.xlsx")
    parser.add_argument("--quality-name", default="high_phase_aux_inputs")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--variants",
        default="wrist,barbell,acceleration",
        help="Comma-separated variants. Use all or wrist+barbell+acceleration for the combined variant.",
    )
    parser.add_argument("--model-types", default="lstm")
    parser.add_argument("--phase-head-type", default=ta.PHASE_HEAD_EXERCISE_ATTN)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fresh-run-tag", default="phase_aux")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_root = (args.output_root or default_output_root()).resolve()
    variants = parse_variants(args.variants)
    print("[phase-aux] variants:", variants)
    print("[phase-aux] output_root:", output_root)
    for idx, variant in enumerate(variants, 1):
        runner_argv = build_runner_argv(args, variant, output_root)
        print(f"\n[phase-aux] {idx}/{len(variants)} launching variant={variant}")
        print("  " + " ".join(runner_argv))
        rc = quality_runner.main(runner_argv)
        if rc:
            return int(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
