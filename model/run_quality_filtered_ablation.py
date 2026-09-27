"""Run quality-filtered phase ablations from the dataset quality workbook.

This is a thin CLI around ``train_ablation.py``.  It keeps the existing
training/evaluation path intact, but narrows the prepared metadata to videos
whose workbook quality labels match the requested levels.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import train_ablation as ta  # noqa: E402


XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"a": XLSX_MAIN_NS, "r": XLSX_REL_NS}

COL_TYPE = "\uc885\ubaa9"
COL_NAME = "\uc601\uc0c1 \ud30c\uc77c\uba85"
COL_QUALITY = "\uae30\uc900(\uc0c1/\uc911/\ud558)"
COL_CHANGED_QUALITY = "\ubcc0\uacbd \ud6c4 \uae30\uc900"

QUALITY_HIGH = "\uc0c1"
QUALITY_MEDIUM = "\uc911"
QUALITY_LOW = "\ud558"
VALID_QUALITIES = {QUALITY_HIGH, QUALITY_MEDIUM, QUALITY_LOW}
QUALITY_ALIASES = {
    "high": QUALITY_HIGH,
    "top": QUALITY_HIGH,
    "best": QUALITY_HIGH,
    "sang": QUALITY_HIGH,
    QUALITY_HIGH: QUALITY_HIGH,
    "medium": QUALITY_MEDIUM,
    "mid": QUALITY_MEDIUM,
    "middle": QUALITY_MEDIUM,
    "jung": QUALITY_MEDIUM,
    QUALITY_MEDIUM: QUALITY_MEDIUM,
    "low": QUALITY_LOW,
    "ha": QUALITY_LOW,
    QUALITY_LOW: QUALITY_LOW,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return ta.json_default(value)


def _col_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref or "A")
    if not match:
        return 0
    idx = 0
    for char in match.group(1):
        idx = idx * 26 + ord(char) - 64
    return idx - 1


def _read_xlsx_rows(path: Path) -> List[Tuple[str, List[List[str]]]]:
    """Read XLSX rows without requiring openpyxl."""

    sheets: List[Tuple[str, List[List[str]]]] = []
    with zipfile.ZipFile(path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", NS):
                shared_strings.append("".join(text.text or "" for text in item.iter(f"{{{XLSX_MAIN_NS}}}t")))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        def cell_value(cell: ET.Element) -> str:
            cell_type = cell.attrib.get("t")
            value = cell.find("a:v", NS)
            if cell_type == "s" and value is not None:
                return shared_strings[int(value.text or "0")]
            if cell_type == "inlineStr":
                return "".join(text.text or "" for text in cell.iter(f"{{{XLSX_MAIN_NS}}}t"))
            return value.text if value is not None and value.text is not None else ""

        for sheet in workbook.findall("a:sheets/a:sheet", NS):
            name = str(sheet.attrib["name"])
            rel_id = sheet.attrib[f"{{{XLSX_REL_NS}}}id"]
            target = relmap[rel_id]
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            root = ET.fromstring(zf.read(target))
            rows: List[List[str]] = []
            for row in root.findall("a:sheetData/a:row", NS):
                values: List[str] = []
                for cell in row.findall("a:c", NS):
                    idx = _col_index(cell.attrib.get("r", "A"))
                    while len(values) <= idx:
                        values.append("")
                    values[idx] = str(cell_value(cell)).strip()
                rows.append(values)
            sheets.append((name, rows))
    return sheets


def _header_index(headers: Sequence[str], expected: str) -> Optional[int]:
    for idx, header in enumerate(headers):
        if str(header).strip() == expected:
            return idx
    return None


def load_quality_map(path: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, str]], List[Dict[str, Any]]]:
    quality: Dict[Tuple[str, str], Dict[str, str]] = {}
    conflicts: List[Dict[str, Any]] = []
    for sheet_name, rows in _read_xlsx_rows(path):
        if not rows:
            continue
        headers = [str(cell).strip() for cell in rows[0]]
        type_idx = _header_index(headers, COL_TYPE)
        name_idx = _header_index(headers, COL_NAME)
        quality_idx = _header_index(headers, COL_QUALITY)
        changed_idx = _header_index(headers, COL_CHANGED_QUALITY)
        if type_idx is None or name_idx is None or quality_idx is None:
            continue

        for row in rows[1:]:
            def get(idx: Optional[int]) -> str:
                if idx is None or idx >= len(row):
                    return ""
                return str(row[idx]).strip()

            exercise = get(type_idx)
            name = get(name_idx)
            level = get(changed_idx) or get(quality_idx)
            if level not in VALID_QUALITIES or not exercise or not name or name.upper() == "X":
                continue
            key = (exercise, name)
            previous = quality.get(key)
            if previous and previous["quality"] != level:
                conflicts.append(
                    {
                        "type": exercise,
                        "name": name,
                        "previous_quality": previous["quality"],
                        "previous_sheet": previous["sheet"],
                        "new_quality": level,
                        "new_sheet": sheet_name,
                    }
                )
            quality[key] = {"quality": level, "sheet": sheet_name}
    return quality, conflicts


def parse_quality_levels(raw: str) -> Tuple[str, ...]:
    levels: List[str] = []
    for part in re.split(r"[,;\s]+", raw.strip()):
        if not part:
            continue
        token = part.strip().lower()
        if token in {
            "high_mid",
            "highmid",
            "high+medium",
            "high+mid",
            "high-medium",
            "high_medium",
            "top_mid",
            "sangjung",
            "sang_jung",
            f"{QUALITY_HIGH}{QUALITY_MEDIUM}",
            f"{QUALITY_HIGH}+{QUALITY_MEDIUM}",
            f"{QUALITY_HIGH}_{QUALITY_MEDIUM}",
            f"{QUALITY_HIGH}-{QUALITY_MEDIUM}",
        }:
            candidates = [QUALITY_HIGH, QUALITY_MEDIUM]
        else:
            if token not in QUALITY_ALIASES:
                raise ValueError(f"Unsupported quality level token: {token!r}")
            candidates = [QUALITY_ALIASES[token]]
        for level in candidates:
            if level not in levels:
                levels.append(level)
    if not levels:
        raise ValueError("At least one quality level is required")
    return tuple(levels)


def quality_slug(levels: Sequence[str]) -> str:
    if tuple(levels) == (QUALITY_HIGH,):
        return "high"
    if set(levels) == {QUALITY_HIGH, QUALITY_MEDIUM}:
        return "high_mid"
    tokens = {QUALITY_HIGH: "high", QUALITY_MEDIUM: "mid", QUALITY_LOW: "low"}
    return "_".join(tokens[level] for level in levels)


def filter_context_by_quality(
    context: ta.ExperimentContext,
    quality_map: Mapping[Tuple[str, str], Mapping[str, str]],
    levels: Sequence[str],
    *,
    quality_name: str,
    quality_xlsx: Path,
    conflicts: Sequence[Mapping[str, Any]],
) -> ta.ExperimentContext:
    allowed = {key for key, value in quality_map.items() if value["quality"] in levels}

    meta = context.meta_df.copy()
    key_series = list(zip(meta["type"].astype(str), meta["name"].astype(str)))
    mask = [key in allowed for key in key_series]
    filtered = meta.loc[mask].reset_index(drop=True)
    if filtered.empty:
        raise RuntimeError(f"Quality filter {levels!r} produced no metadata rows")

    filtered["quality"] = [quality_map[(str(row["type"]), str(row["name"]))]["quality"] for _, row in filtered.iterrows()]
    filtered["quality_source_sheet"] = [
        quality_map[(str(row["type"]), str(row["name"]))]["sheet"] for _, row in filtered.iterrows()
    ]

    train = filtered[filtered["split"] == "train"].reset_index(drop=True)
    val = filtered[filtered["split"] == "val"].reset_index(drop=True)
    if train.empty or val.empty:
        raise RuntimeError(
            f"Quality filter {levels!r} produced an empty split: train={len(train)} val={len(val)}"
        )

    filtered_keys = {(str(row["type"]), str(row["name"])) for _, row in filtered.iterrows()}
    labels = {key: value for key, value in context.labels.items() if key in filtered_keys}

    cfg = copy.deepcopy(context.cfg)
    cfg.update(
        {
            "quality_xlsx": str(quality_xlsx),
            "quality_name": quality_name,
            "quality_levels": list(levels),
            "quality_conflicts": list(conflicts),
        }
    )
    return dataclasses.replace(context, cfg=cfg, labels=labels, meta_df=filtered, train_meta=train, val_meta=val)


def summarize_context(context: ta.ExperimentContext) -> Dict[str, Any]:
    def table(df: Any) -> Dict[str, Dict[str, int]]:
        grouped = df.groupby(["split", "type"]).size().unstack(fill_value=0)
        return {
            str(split): {str(col): int(value) for col, value in row.items()}
            for split, row in grouped.iterrows()
        }

    return {
        "videos": int(len(context.meta_df)),
        "train_videos": int(len(context.train_meta)),
        "val_videos": int(len(context.val_meta)),
        "split_type_counts": table(context.meta_df),
        "quality_counts": {str(k): int(v) for k, v in Counter(context.meta_df["quality"]).items()},
    }


def build_cfgs(args: argparse.Namespace, levels: Sequence[str], qname: str) -> List[Dict[str, Any]]:
    model_types = [part.strip().lower() for part in args.model_types.split(",") if part.strip()]
    if not model_types:
        raise ValueError("--model-types must not be empty")
    cfgs: List[Dict[str, Any]] = []
    for model_type in model_types:
        cfg = copy.deepcopy(ta.DEFAULT_EXPERIMENT_CONFIG)
        cfg.update(
            {
                "output_root": str(args.output_root),
                "pose_backend": args.pose_backend,
                "joint_subset": args.joint_subset,
                "phase_label_scheme": args.phase_label_scheme,
                "phase_head_type": args.phase_head_type,
                "phase_pooling": args.phase_pooling,
                "derivative_mode": args.derivative_mode,
                "phase_aux_inputs": args.phase_aux_inputs,
                "phase_conditioning": args.phase_conditioning,
                "exercise_id_source": args.exercise_id_source,
                "model_type": model_type,
                "epochs": int(args.epochs),
                "clip_len": int(args.clip_len),
                "train_stride": int(args.train_stride),
                "batch": int(args.batch),
                "num_workers": int(args.num_workers),
                "write_meta_csv": False,
                "extract_missing_pose": False,
                "resume": not bool(args.force_retrain),
                "skip_completed": not bool(args.force_retrain),
                "force_retrain": bool(args.force_retrain),
                "fresh_run_tag": args.fresh_run_tag,
                "run_kind": f"quality_{qname}",
                "quality_xlsx": str(args.quality_xlsx),
                "quality_name": qname,
                "quality_levels": list(levels),
            }
        )
        if args.barbell_edge_policy is not None:
            cfg["barbell_edge_policy"] = args.barbell_edge_policy

        normalized = ta.normalize_cfg(cfg)
        if args.exercise_id_source not in {None, "", "auto"}:
            requested_source = str(args.exercise_id_source).strip().lower()
            if normalized.get("exercise_id_source") != requested_source:
                raise ValueError(
                    "--exercise-id-source is derived from --phase-conditioning in train_ablation; "
                    f"requested {requested_source!r}, normalized to {normalized.get('exercise_id_source')!r}"
                )
        cfgs.append(normalized)
    return cfgs


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-xlsx", type=Path, default=PROJECT_ROOT / "dataset \ubd84\ub958.xlsx")
    parser.add_argument("--quality-levels", required=True, help="Comma-separated: high, medium, low or high_mid")
    parser.add_argument("--quality-name", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-types", default="mlp,lstm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pose-backend", default=ta.POSE_BACKEND_MEDIAPIPE)
    parser.add_argument("--joint-subset", default=ta.JOINT_SUBSET_ALL)
    parser.add_argument("--phase-label-scheme", default=ta.PHASE_LABEL_SCHEME_AS_LABELED)
    parser.add_argument("--phase-head-type", default=ta.PHASE_HEAD_MLP)
    parser.add_argument("--phase-pooling", default="temporal_avg")
    parser.add_argument("--derivative-mode", default="pose")
    parser.add_argument(
        "--phase-aux-inputs",
        default="",
        help="Comma-separated phase-head-only auxiliary inputs: wrist,barbell,acceleration.",
    )
    parser.add_argument("--barbell-edge-policy", default=None)
    parser.add_argument("--phase-conditioning", default=None)
    parser.add_argument("--exercise-id-source", default=None)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fresh-run-tag", default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Disable the default CUDA-required guard.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.quality_xlsx = args.quality_xlsx.resolve()
    args.output_root = args.output_root.resolve()
    levels = parse_quality_levels(args.quality_levels)
    qname = args.quality_name or quality_slug(levels)

    if not args.allow_cpu and ta.DEVICE != "cuda":
        raise RuntimeError(
            f"CUDA is required, but train_ablation.DEVICE={ta.DEVICE!r}. "
            "Run with a CUDA-enabled PyTorch interpreter."
        )
    if not args.quality_xlsx.exists():
        raise FileNotFoundError(f"Quality workbook not found: {args.quality_xlsx}")

    print(
        f"[quality-ablation] device={ta.DEVICE} torch={ta.torch.__version__} "
        f"cuda_available={ta.torch.cuda.is_available()} gpu="
        f"{ta.torch.cuda.get_device_name(0) if ta.torch.cuda.is_available() else 'NONE'}",
        flush=True,
    )
    quality_map, conflicts = load_quality_map(args.quality_xlsx)
    print(
        f"[quality-ablation] workbook={args.quality_xlsx} unique_quality_rows={len(quality_map)} "
        f"conflicts={len(conflicts)} selected={list(levels)} name={qname}",
        flush=True,
    )

    cfgs = build_cfgs(args, levels, qname)
    context = ta.prepare_context(cfgs[0], verbose=True, update_globals=True)
    context = filter_context_by_quality(
        context,
        quality_map,
        levels,
        quality_name=qname,
        quality_xlsx=args.quality_xlsx,
        conflicts=conflicts,
    )
    summary = summarize_context(context)
    print("[quality-ablation] filtered_summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    filtered_meta_csv = args.output_root / "quality_filtered_meta.csv"
    context.meta_df.to_csv(filtered_meta_csv, index=False, encoding="utf-8-sig")

    manifest_base = {
        "created_at_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "quality_xlsx": str(args.quality_xlsx),
        "quality_name": qname,
        "quality_levels": list(levels),
        "quality_conflicts": conflicts,
        "filtered_meta_csv": str(filtered_meta_csv),
        "summary": summary,
        "configs": cfgs,
        "device": ta.DEVICE,
        "torch": ta.torch.__version__,
        "cuda_available": bool(ta.torch.cuda.is_available()),
        "gpu_name": ta.torch.cuda.get_device_name(0) if ta.torch.cuda.is_available() else None,
    }
    write_manifest(args.output_root / "quality_run_manifest.json", {**manifest_base, "status": "dry_run" if args.dry_run else "started"})

    if args.dry_run:
        print("[quality-ablation] dry-run complete; no training launched")
        return 0

    all_results: List[Dict[str, Any]] = []
    if context.ablation_log_json.exists():
        all_results = json.loads(context.ablation_log_json.read_text(encoding="utf-8"))
    for idx, cfg in enumerate(cfgs, start=1):
        print(f"[quality-ablation] run {idx}/{len(cfgs)}: {ta.make_run_exp_name(cfg)}", flush=True)
        result = ta.run_one_experiment(cfg, context=context)
        result.update({"quality_name": qname, "quality_levels": list(levels), "quality_summary": summary})
        all_results = [row for row in all_results if row.get("exp_name") != result["exp_name"]]
        all_results.append(result)
        ta.append_result_logs(all_results, context)
    write_manifest(args.output_root / "quality_run_manifest.json", {**manifest_base, "status": "complete", "results": all_results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
