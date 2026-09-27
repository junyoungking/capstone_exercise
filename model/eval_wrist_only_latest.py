"""
Evaluate the current wrist-only phase checkpoint without touching training.

The default mode auto-discovers the most recently updated ``latest.pt`` whose
neighboring ``config.json`` has ``joint_subset=wrist_only``.  This is intended
for checking an in-progress training run: the trainer keeps writing
``latest.pt``/``history.json`` every epoch, and this script can read a stable
snapshot with retries.

Examples
--------
Fast current training-log metrics only:

    python model/eval_wrist_only_latest.py --history-only

Full current validation from the latest wrist-only checkpoint:

    python model/eval_wrist_only_latest.py --device cpu

Specific experiment directory:

    python model/eval_wrist_only_latest.py --exp-dir data\\측면\\data\\STGCN_LSTM\\ablation\\mediapipe_mediapipe_wrist2_jwrist_only_mlp_h128_c16_pw_2.0_ts2_do03_aug1
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_ablation as ta  # noqa: E402


DEFAULT_SEARCH_ROOTS = (
    PROJECT_ROOT / "data" / "측면" / "data" / "STGCN_LSTM" / "ablation",
    PROJECT_ROOT / "phase_experiments",
)

PRINT_KEYS = (
    "epoch",
    "best_epoch",
    "best_val_score",
    "history_val_action_acc",
    "history_val_action_f1",
    "history_val_phase_acc",
    "history_val_phase_f1",
    "window_val_loss",
    "window_action_acc",
    "window_action_f1",
    "window_phase_acc",
    "window_phase_f1",
    "video_action_acc",
    "video_action_f1",
    "video_phase_acc",
    "video_phase_f1",
    "video_phase_macro_f1",
    "video_ready_f1",
    "video_down_f1",
    "video_up_f1",
    "video_count_mae",
    "video_count_obo",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return ta.json_default(value)


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if _is_nan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _last_history_values(history: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    max_len = 0
    for key, values in history.items():
        if not isinstance(values, list):
            continue
        max_len = max(max_len, len(values))
        if values:
            result[f"history_{key}"] = values[-1]
    if max_len:
        result["history_epochs"] = max_len
    return result


def _read_history(exp_dir: Path, checkpoint: Mapping[str, Any] | None = None) -> dict[str, Any]:
    ckpt_history = checkpoint.get("history") if checkpoint else None
    if isinstance(ckpt_history, dict) and ckpt_history:
        return dict(ckpt_history)
    history_path = exp_dir / "history.json"
    if history_path.exists():
        loaded = _read_json(history_path)
        if isinstance(loaded, dict):
            return loaded
    return {}


def _normalize_wrist_cfg(cfg: Mapping[str, Any], *, allow_non_wrist: bool) -> dict[str, Any]:
    norm = ta.normalize_cfg(cfg)
    if ta.normalize_joint_subset(norm.get("joint_subset")) != ta.JOINT_SUBSET_WRIST_ONLY and not allow_non_wrist:
        raise SystemExit(
            "Resolved checkpoint is not wrist-only. "
            f"joint_subset={norm.get('joint_subset')!r}; pass --allow-non-wrist to override."
        )
    # Keep validation read-mostly.  prepare_context() may create existing parent
    # dirs, but this prevents rewriting meta CSV or extracting poses.
    norm["write_meta_csv"] = False
    norm["extract_missing_pose"] = False
    return norm


def _merge_config(config_json: Mapping[str, Any], checkpoint: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = dict(config_json)
    ckpt_cfg = checkpoint.get("cfg") if checkpoint else None
    if isinstance(ckpt_cfg, Mapping):
        # The checkpoint cfg is the exact model contract used for this state
        # dict.  Keep config.json as fallback for path fields if needed.
        merged.update(dict(ckpt_cfg))
    return merged


def _checkpoint_load_errors() -> tuple[type[BaseException], ...]:
    return (RuntimeError, EOFError, OSError, pickle.UnpicklingError)


def load_checkpoint_with_retries(
    checkpoint_path: Path,
    *,
    map_location: str,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            loaded = torch.load(checkpoint_path, map_location=map_location)
            if not isinstance(loaded, dict):
                raise RuntimeError(f"checkpoint payload is not a dict: {type(loaded)!r}")
            return loaded
        except _checkpoint_load_errors() as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep)
    raise RuntimeError(f"failed to load stable checkpoint after {retries} attempts: {checkpoint_path}") from last_error


def candidate_from_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        return None
    try:
        cfg = _read_json(config_path)
        subset = ta.normalize_joint_subset(cfg.get("joint_subset"))
    except Exception:
        return None
    if subset != ta.JOINT_SUBSET_WRIST_ONLY:
        return None
    return {
        "checkpoint_path": checkpoint_path,
        "exp_dir": checkpoint_path.parent,
        "config_path": config_path,
        "mtime": checkpoint_path.stat().st_mtime,
        "cfg": cfg,
    }


def discover_wrist_checkpoint(search_roots: list[Path], checkpoint_name: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for root in search_roots:
        if not root.exists():
            continue
        for checkpoint_path in root.rglob(checkpoint_name):
            candidate = candidate_from_checkpoint(checkpoint_path)
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        searched = ", ".join(str(p) for p in search_roots)
        raise SystemExit(f"No wrist-only {checkpoint_name!r} found under: {searched}")
    candidates.sort(key=lambda item: item["mtime"], reverse=True)
    return candidates[0]


def list_wrist_candidates(search_roots: list[Path], checkpoint_name: str) -> int:
    candidates: list[dict[str, Any]] = []
    for root in search_roots:
        if not root.exists():
            continue
        for checkpoint_path in root.rglob(checkpoint_name):
            candidate = candidate_from_checkpoint(checkpoint_path)
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(key=lambda item: item["mtime"], reverse=True)
    if not candidates:
        print("No wrist-only candidates found.")
        return 1
    for i, item in enumerate(candidates, start=1):
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(item["mtime"])))
        print(f"[{i:02d}] {when}  {item['checkpoint_path']}")
    return 0


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        exp_dir = checkpoint_path.parent
        config_path = exp_dir / "config.json"
        config_json = _read_json(config_path) if config_path.exists() else {}
        return checkpoint_path, exp_dir, config_json

    if args.exp_dir:
        exp_dir = Path(args.exp_dir).expanduser().resolve()
        checkpoint_path = exp_dir / args.checkpoint_name
        config_path = exp_dir / "config.json"
        config_json = _read_json(config_path) if config_path.exists() else {}
        return checkpoint_path, exp_dir, config_json

    search_roots = [Path(p).expanduser().resolve() for p in args.search_root] if args.search_root else list(DEFAULT_SEARCH_ROOTS)
    candidate = discover_wrist_checkpoint(search_roots, args.checkpoint_name)
    return Path(candidate["checkpoint_path"]), Path(candidate["exp_dir"]), dict(candidate["cfg"])


def build_base_result(
    *,
    checkpoint_path: Path,
    exp_dir: Path,
    cfg: Mapping[str, Any],
    checkpoint: Mapping[str, Any] | None,
    history: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exp_name": exp_dir.name,
        "exp_dir": str(exp_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(checkpoint_path.stat().st_mtime)),
        "pose_backend": cfg.get("pose_backend"),
        "joint_subset": cfg.get("joint_subset"),
        "source_num_kpt": cfg.get("source_num_kpt"),
        "model_num_kpt": cfg.get("model_num_kpt"),
        "num_kpt": cfg.get("num_kpt"),
        "pose_graph_id": cfg.get("pose_graph_id"),
        "model_graph_id": cfg.get("model_graph_id"),
        "model_type": cfg.get("model_type"),
        "phase_head_type": cfg.get("phase_head_type"),
        "phase_pooling": cfg.get("phase_pooling"),
        "derivative_mode": cfg.get("derivative_mode"),
        "phase_label_scheme": cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED),
        "clip_len": cfg.get("clip_len"),
        "train_stride": cfg.get("train_stride"),
        "phase_loss_alpha": cfg.get("phase_loss_alpha"),
    }
    if checkpoint:
        result.update(
            {
                "epoch": int(checkpoint.get("epoch", -1)),
                "best_epoch": int(checkpoint.get("best_epoch", -1)),
                "best_val_score": float(checkpoint.get("best_score", float("nan"))),
            }
        )
    result.update(_last_history_values(history))
    result.setdefault("epoch", result.get("history_epochs", -1))
    return result


def evaluate_checkpoint(
    *,
    checkpoint: Mapping[str, Any],
    cfg: Mapping[str, Any],
    device: str,
    run_window_metrics: bool,
    run_video_metrics: bool,
    quiet_context: bool,
) -> dict[str, Any]:
    context = ta.prepare_context(cfg, verbose=not quiet_context, update_globals=False)
    model = ta.build_model(cfg, device=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metrics: dict[str, Any] = {}
    if run_window_metrics:
        train_ds = ta.CausalWindowDataset(
            context.train_meta,
            labels=context.labels,
            clip_len=int(cfg["clip_len"]),
            stride=int(cfg["train_stride"]),
            train=True,
            aug=False,
            derivative_mode=str(cfg["derivative_mode"]),
            pose_backend=str(cfg["pose_backend"]),
            joint_subset=str(cfg["joint_subset"]),
            phase_label_scheme=str(cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED)),
        )
        val_ds = ta.CausalWindowDataset(
            context.val_meta,
            labels=context.labels,
            clip_len=int(cfg["clip_len"]),
            stride=int(cfg["train_stride"]),
            train=False,
            aug=False,
            derivative_mode=str(cfg["derivative_mode"]),
            pose_backend=str(cfg["pose_backend"]),
            joint_subset=str(cfg["joint_subset"]),
            phase_label_scheme=str(cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED)),
        )
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise RuntimeError(f"empty dataset windows: train={len(train_ds)} val={len(val_ds)}")
        cw, pw = ta.compute_weights_from_ds(train_ds, device=device)
        val_dl = DataLoader(
            val_ds,
            batch_size=int(cfg["batch"]),
            shuffle=False,
            num_workers=int(cfg["num_workers"]),
            pin_memory=False,
        )
        window_metrics = ta.eval_windows(model, val_dl, cw, pw, float(cfg["phase_loss_alpha"]))
        metrics.update({f"window_{key}": value for key, value in window_metrics.items()})

    if run_video_metrics:
        video_metrics = ta.eval_videos(
            model,
            context.val_meta,
            int(cfg["clip_len"]),
            int(cfg["train_stride"]),
            labels=context.labels,
            pose_backend=str(cfg["pose_backend"]),
            joint_subset=str(cfg["joint_subset"]),
            phase_label_scheme=str(cfg.get("phase_label_scheme", ta.PHASE_LABEL_SCHEME_AS_LABELED)),
        )
        for key, value in video_metrics.items():
            out_key = key if key.startswith("video_") else f"video_{key}"
            metrics[out_key] = value

    return metrics


def write_outputs(result: Mapping[str, Any], *, exp_dir: Path, json_out: str | None, csv_out: str | None) -> None:
    if json_out:
        json_path = exp_dir / "wrist_only_validation_current.json" if json_out == "auto" else Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        print(f"saved json: {json_path}")
    if csv_out:
        csv_path = exp_dir / "wrist_only_validation_current.csv" if csv_out == "auto" else Path(csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(result)]).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"saved csv : {csv_path}")


def print_summary(result: Mapping[str, Any]) -> None:
    print("\nResolved wrist-only checkpoint")
    print(f"  exp       : {result.get('exp_name')}")
    print(f"  checkpoint: {result.get('checkpoint_path')}")
    print(f"  modified  : {result.get('checkpoint_mtime')}")
    print(
        "  config    : "
        f"backend={result.get('pose_backend')} subset={result.get('joint_subset')} "
        f"source_V={result.get('source_num_kpt')} model_V={result.get('model_num_kpt')} "
        f"head={result.get('phase_head_type')} input={result.get('derivative_mode')} "
        f"label_scheme={result.get('phase_label_scheme')}"
    )
    print("\nMetrics")
    for key in PRINT_KEYS:
        if key in result:
            print(f"  {key:24s} {_format_value(result[key])}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Get current validation metrics from the latest wrist-only checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp-dir", help="Experiment directory containing config.json and latest.pt.")
    parser.add_argument("--checkpoint", help="Explicit checkpoint path. Defaults to auto-discovered wrist-only latest.pt.")
    parser.add_argument("--checkpoint-name", default="latest.pt", help="Checkpoint filename used with --exp-dir or discovery.")
    parser.add_argument(
        "--search-root",
        action="append",
        help="Root to search for wrist-only checkpoints. Can be passed multiple times; overrides built-in roots.",
    )
    parser.add_argument("--list-candidates", action="store_true", help="List wrist-only candidates and exit.")
    parser.add_argument("--device", default="cpu", help="Evaluation device. Use cpu while another training process owns cuda.")
    parser.add_argument("--history-only", action="store_true", help="Only read history.json/config.json; do not load/evaluate checkpoint.")
    parser.add_argument("--no-window-metrics", action="store_true", help="Skip validation-window metrics from eval_windows().")
    parser.add_argument("--no-video-metrics", action="store_true", help="Skip video-level eval_videos() metrics.")
    parser.add_argument("--quiet-context", action="store_true", help="Suppress prepare_context() dataset summary output.")
    parser.add_argument("--retries", type=int, default=8, help="Checkpoint load attempts for in-progress writer races.")
    parser.add_argument("--retry-sleep", type=float, default=1.0, help="Seconds between checkpoint load attempts.")
    parser.add_argument("--allow-non-wrist", action="store_true", help="Allow evaluating a non-wrist checkpoint explicitly.")
    parser.add_argument("--json-out", help="Optional JSON output path; use 'auto' for exp_dir/wrist_only_validation_current.json.")
    parser.add_argument("--csv-out", help="Optional CSV output path; use 'auto' for exp_dir/wrist_only_validation_current.csv.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    search_roots = [Path(p).expanduser().resolve() for p in args.search_root] if args.search_root else list(DEFAULT_SEARCH_ROOTS)
    if args.list_candidates:
        return list_wrist_candidates(search_roots, args.checkpoint_name)

    checkpoint_path, exp_dir, config_json = resolve_paths(args)
    if not checkpoint_path.exists() and not args.history_only:
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    checkpoint: dict[str, Any] | None = None
    if not args.history_only:
        checkpoint = load_checkpoint_with_retries(
            checkpoint_path,
            map_location=args.device,
            retries=max(1, int(args.retries)),
            retry_sleep=max(0.0, float(args.retry_sleep)),
        )

    cfg = _normalize_wrist_cfg(_merge_config(config_json, checkpoint), allow_non_wrist=bool(args.allow_non_wrist))
    history = _read_history(exp_dir, checkpoint)
    result = build_base_result(checkpoint_path=checkpoint_path, exp_dir=exp_dir, cfg=cfg, checkpoint=checkpoint, history=history)

    if not args.history_only and checkpoint is not None:
        eval_metrics = evaluate_checkpoint(
            checkpoint=checkpoint,
            cfg=cfg,
            device=args.device,
            run_window_metrics=not args.no_window_metrics,
            run_video_metrics=not args.no_video_metrics,
            quiet_context=bool(args.quiet_context),
        )
        result.update(eval_metrics)

    print_summary(result)
    write_outputs(result, exp_dir=exp_dir, json_out=args.json_out, csv_out=args.csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
