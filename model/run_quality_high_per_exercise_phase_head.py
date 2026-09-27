"""Run the current best phase setting with separate phase heads per exercise.

Default setting mirrors the best-performing phase architecture family from the
saved summaries, then swaps the phase head to ``per_exercise_mlp``:

- quality filter: high/상 only
- pose input: MediaPipe33 + barbell fusion
- phase label: bar_direction
- barbell edge policy: wrists
- clip_len=16, train_stride=2, batch=64, epochs=30
- phase head: one MLP head each for squat / benchpress / deadlift

Examples
--------
Dry-run config check:

    python model/run_quality_high_per_exercise_phase_head.py --dry-run --allow-cpu

Full CUDA run:

    python model/run_quality_high_per_exercise_phase_head.py

Run only MLP or only LSTM:

    python model/run_quality_high_per_exercise_phase_head.py --model-types mlp
    python model/run_quality_high_per_exercise_phase_head.py --model-types lstm
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


def default_output_root() -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return PROJECT_ROOT / "phase_experiments" / "quality_high_per_exercise_head" / stamp


def build_runner_argv(args: argparse.Namespace) -> List[str]:
    output_root = args.output_root or default_output_root()
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
        ta.JOINT_SUBSET_ALL,
        "--phase-label-scheme",
        ta.PHASE_LABEL_SCHEME_BAR_DIRECTION,
        "--phase-head-type",
        ta.PHASE_HEAD_PER_EXERCISE_MLP,
        "--phase-conditioning",
        ta.PHASE_CONDITIONING_GROUND_TRUTH_EXERCISE_LABEL,
        "--barbell-edge-policy",
        ta.BARBELL_EDGE_POLICY_WRISTS,
        "--derivative-mode",
        "pose",
        "--clip-len",
        str(args.clip_len),
        "--train-stride",
        str(args.train_stride),
        "--batch",
        str(args.batch),
        "--num-workers",
        str(args.num_workers),
        "--fresh-run-tag",
        args.fresh_run_tag,
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
    parser.add_argument("--quality-name", default="high_per_exercise_head")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--model-types",
        default="mlp,lstm",
        help="Comma-separated model types to run. Default runs both for direct comparison.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fresh-run-tag", default="per_exercise_head")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    runner_argv = build_runner_argv(args)
    print("[per-exercise-head] launching quality runner with:")
    print("  " + " ".join(runner_argv))
    return quality_runner.main(runner_argv)


if __name__ == "__main__":
    raise SystemExit(main())
