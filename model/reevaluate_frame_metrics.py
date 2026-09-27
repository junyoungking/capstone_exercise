"""Re-evaluate existing checkpoints with frame-level validation metrics.

This does not retrain.  It loads each experiment's ``config.json`` and best
checkpoint, rebuilds the same validation split, then writes frame-level action
and phase metrics produced by ``train_ablation.eval_videos``.

Examples
--------
Re-evaluate the current high-quality best-architecture runs:

    python model/reevaluate_frame_metrics.py \
      phase_experiments/quality_filter_best_arch/20260615T004443Z \
      phase_experiments/quality_filter_best_arch_lstm/20260615T020450Z

Run on CPU so an active training process can keep the GPU:

    python model/reevaluate_frame_metrics.py --device cpu <experiment-root>
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import run_quality_filtered_ablation as quality_runner  # noqa: E402
from model import train_ablation as ta  # noqa: E402


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return ta.json_default(value)


def _checkpoint_epoch(path: Path) -> int:
    match = re.search(r"best_ep(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def find_config_paths(paths: Iterable[Path]) -> List[Path]:
    configs: List[Path] = []
    seen = set()
    for raw in paths:
        path = raw.resolve()
        candidates: List[Path]
        if path.is_file() and path.name == "config.json":
            candidates = [path]
        elif path.is_dir():
            if (path / "config.json").exists():
                candidates = [path / "config.json"]
            else:
                candidates = sorted(path.rglob("config.json"))
        else:
            candidates = []
        for cfg in candidates:
            key = str(cfg.resolve()).lower()
            if key not in seen:
                seen.add(key)
                configs.append(cfg.resolve())
    return configs


def select_checkpoint(exp_dir: Path, explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        ckpt = explicit.resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        return ckpt
    bests = sorted(exp_dir.glob("best_ep*.pt"), key=_checkpoint_epoch)
    if bests:
        return bests[-1]
    latest = exp_dir / "latest.pt"
    if latest.exists():
        return latest
    raise FileNotFoundError(f"no best_ep*.pt/latest.pt found under {exp_dir}")


def load_cfg(config_path: Path) -> Dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["extract_missing_pose"] = False
    cfg["write_meta_csv"] = False
    cfg["resume"] = True
    cfg["skip_completed"] = True
    return ta.normalize_cfg(cfg)


def prepare_eval_context(cfg: Mapping[str, Any]) -> ta.ExperimentContext:
    context = ta.prepare_context(cfg, verbose=False, update_globals=False)
    quality_xlsx = cfg.get("quality_xlsx")
    quality_levels = cfg.get("quality_levels")
    if quality_xlsx and quality_levels:
        xlsx = Path(str(quality_xlsx))
        if not xlsx.is_absolute():
            xlsx = PROJECT_ROOT / xlsx
        quality_map, conflicts = quality_runner.load_quality_map(xlsx)
        levels = tuple(str(level) for level in quality_levels)
        context = quality_runner.filter_context_by_quality(
            context,
            quality_map,
            levels,
            quality_name=str(cfg.get("quality_name") or quality_runner.quality_slug(levels)),
            quality_xlsx=xlsx,
            conflicts=conflicts,
        )
    return context


def reevaluate_one(
    config_path: Path,
    *,
    checkpoint: Optional[Path],
    device: str,
    eval_stride: int,
    write_per_experiment: bool,
) -> Dict[str, Any]:
    cfg = load_cfg(config_path)
    exp_dir = config_path.parent
    ckpt_path = select_checkpoint(exp_dir, checkpoint)
    context = prepare_eval_context(cfg)

    dev = torch.device(device if device != "auto" else ta.DEVICE)
    ckpt = torch.load(ckpt_path, map_location=dev)
    model = ta.build_model(cfg, device=dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    metadata = ta.model_param_metadata(model)
    metrics = ta.eval_videos(
        model,
        context.val_meta,
        int(cfg["clip_len"]),
        int(eval_stride),
        labels=context.labels,
        pose_backend=str(cfg["pose_backend"]),
        joint_subset=str(cfg["joint_subset"]),
        phase_label_scheme=str(cfg["phase_label_scheme"]),
        barbell_edge_policy=str(cfg["barbell_edge_policy"]),
        phase_conditioning=str(cfg["phase_conditioning"]),
    )
    result = {
        "exp_name": ta.make_run_exp_name(cfg),
        "config_path": str(config_path),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch", ckpt.get("best_epoch", -1))),
        "best_epoch": int(ckpt.get("best_epoch", -1)),
        "best_val_score": float(ckpt.get("best_score", float("nan"))),
        "eval_device": str(dev),
        "eval_stride": int(eval_stride),
        "val_videos": int(len(context.val_meta)),
        "quality_name": cfg.get("quality_name"),
        "quality_levels": cfg.get("quality_levels"),
        "pose_backend": cfg["pose_backend"],
        "joint_subset": cfg["joint_subset"],
        "model_type": cfg["model_type"],
        "phase_head_type": cfg["phase_head_type"],
        "phase_conditioning": cfg["phase_conditioning"],
        "derivative_mode": cfg["derivative_mode"],
        "phase_aux_inputs": cfg["phase_aux_inputs"],
        "phase_aux_dim": cfg["phase_aux_dim"],
        "phase_label_scheme": cfg["phase_label_scheme"],
        **metadata,
        **metrics,
    }
    if write_per_experiment:
        out_path = exp_dir / "frame_eval_metrics.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Experiment dirs, run roots, or config.json files.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Explicit checkpoint for a single config.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON summary path.")
    parser.add_argument("--csv", type=Path, default=None, help="Output CSV summary path.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--eval-stride", type=int, default=1, help="Frame eval stride. Use 1 for per-frame final validation.")
    parser.add_argument("--no-per-experiment-write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configs = find_config_paths(args.paths)
    if not configs:
        raise FileNotFoundError("no config.json files found in requested paths")
    if args.checkpoint is not None and len(configs) != 1:
        raise ValueError("--checkpoint can only be used with exactly one config")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    default_dir = PROJECT_ROOT / "phase_experiments" / "frame_metric_reeval"
    output_json = (args.output or (default_dir / f"frame_metrics_{stamp}.json")).resolve()
    output_csv = (args.csv or output_json.with_suffix(".csv")).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"[frame-reeval] configs={len(configs)} device={args.device} eval_stride={args.eval_stride}")
    results: List[Dict[str, Any]] = []
    for idx, config_path in enumerate(configs, 1):
        print(f"[frame-reeval] {idx}/{len(configs)} {config_path}", flush=True)
        try:
            result = reevaluate_one(
                config_path,
                checkpoint=args.checkpoint,
                device=args.device,
                eval_stride=int(args.eval_stride),
                write_per_experiment=not bool(args.no_per_experiment_write),
            )
            results.append(result)
            print(
                "  -> "
                f"frame_action_f1={result.get('frame_action_f1', float('nan')):.4f} "
                f"frame_phase_f1={result.get('frame_phase_f1', float('nan')):.4f} "
                f"vote_action_f1={result.get('video_vote_action_f1', float('nan')):.4f}",
                flush=True,
            )
        except Exception as exc:
            result = {
                "config_path": str(config_path),
                "status": "failed",
                "error": repr(exc),
            }
            results.append(result)
            print(f"  !! failed: {exc!r}", flush=True)

    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[frame-reeval] saved json: {output_json}")
    print(f"[frame-reeval] saved csv : {output_csv}")
    return 1 if any(row.get("status") == "failed" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
