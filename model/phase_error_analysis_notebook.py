"""Phase error analysis script-backed notebook.

PyCharm/Jupyter usage:
    Open this file and run the ``# %%`` cells top-to-bottom.

Plain Python usage:
    python model/phase_error_analysis_notebook.py

Default behavior is artifact-only:
    - no training
    - no fresh checkpoint inference
    - no dataset/label mutation
    - joint masking ablation is marked optional/skipped unless existing outputs
      are already present.
"""

# %% Setup / imports / configuration
from __future__ import annotations

import json
import math
import os
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

try:  # Optional but normally available in the capstone environment.
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    SKLEARN_CLUSTER_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for lean notebooks.
    KMeans = PCA = StandardScaler = None  # type: ignore[assignment]
    SKLEARN_CLUSTER_AVAILABLE = False


def find_project_root(start: Optional[Path] = None) -> Path:
    """Resolve project root in script and PyCharm/Jupyter cell contexts."""
    if start is None:
        file_name = globals().get("__file__")
        start = Path(str(file_name)).resolve() if file_name else Path.cwd().resolve()
    start = start if start.is_dir() else start.parent
    for candidate in [start, *start.parents]:
        if (candidate / "model").exists() and ((candidate / "phase_experiments").exists() or (candidate / ".omx").exists()):
            return candidate
    return start


def resolve_script_path_and_root() -> Tuple[Optional[Path], str, Path]:
    file_name = globals().get("__file__")
    if file_name:
        script_path = Path(str(file_name)).resolve()
        project_root = find_project_root(script_path)
        try:
            script_label = str(script_path.relative_to(project_root))
        except Exception:
            script_label = str(script_path)
        return script_path, script_label, project_root
    project_root = find_project_root(Path.cwd())
    return None, "interactive_cell:phase_error_analysis_notebook.py", project_root


SCRIPT_PATH, SCRIPT_LABEL, PROJECT_ROOT = resolve_script_path_and_root()

# PyCharm-friendly top-level controls.  Set RUN_DIR to a Path to pin a run.
RUN_DIR: Optional[Path] = None
RUN_DIR_ENV = os.environ.get("RUN_DIR")
if RUN_DIR_ENV:
    RUN_DIR = Path(RUN_DIR_ENV)

SEARCH_PATTERNS = [
    "phase_experiments/exercise_embedding/full/**/raw_phase_predictions.csv",
    "phase_experiments/pooling_ablation/full/**/raw_phase_predictions.csv",
    "phase_experiments/**/raw_phase_predictions.csv",
]

RUN_JOINT_MASKING = os.environ.get("RUN_JOINT_MASKING", "0").strip().lower() in {"1", "true", "yes", "on"}
BOUNDARY_WINDOW = int(os.environ.get("BOUNDARY_WINDOW", "5"))
N_TIMELINE_VIDEOS = int(os.environ.get("N_TIMELINE_VIDEOS", "3"))
TRANSITION_MATCH_TOLERANCE_FRAMES = int(os.environ.get("TRANSITION_MATCH_TOLERANCE_FRAMES", "45"))
RANDOM_SEED = int(os.environ.get("PHASE_ERROR_ANALYSIS_SEED", "42"))

PHASE_NAMES = ["ready", "down", "up"]
PHASE_IDS = list(range(len(PHASE_NAMES)))
PHASE_NAME_BY_ID = dict(enumerate(PHASE_NAMES))
PHASE_ID_BY_NAME = {name: idx for idx, name in PHASE_NAME_BY_ID.items()}

RAW_COL = "pred_phase_raw"
SMOOTH_COL = "pred_phase_offline_smooth"
PREDICTION_MODES = [("raw", RAW_COL), ("offline_smooth", SMOOTH_COL)]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="cp949"))


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if pd.isna(value):
        return None
    return str(value)


def safe_name(value: str, max_len: int = 120) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return out[:max_len] or "item"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_fig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Small dependency-free markdown table writer.

    pandas.DataFrame.to_markdown requires the optional ``tabulate`` package,
    which is not guaranteed in the capstone environment.
    """
    if df.empty:
        return "_empty_"
    work = df.head(max_rows).copy()
    columns = [str(c) for c in work.columns]

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}" if np.isfinite(value) else ""
        text = "" if pd.isna(value) else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in work.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_... {len(df) - max_rows} more rows omitted_")
    return "\n".join(lines)


def phase_name(value: Any) -> str:
    try:
        return PHASE_NAME_BY_ID[int(value)]
    except Exception:
        return str(value)


def parse_manifest_time(value: Any) -> float:
    if not value:
        return 0.0
    text = str(value)
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    return 0.0


def manifest_completion_score(manifest: Mapping[str, Any]) -> int:
    status = str(manifest.get("completion_status", manifest.get("status", ""))).lower()
    return 1 if status in {"complete", "completed", "success", "succeeded", "finished"} else 0


def find_run_candidates() -> List[Dict[str, Any]]:
    candidates: Dict[Path, Dict[str, Any]] = {}
    for pattern in SEARCH_PATTERNS:
        for raw_path in PROJECT_ROOT.glob(pattern):
            run_dir = raw_path.parent
            manifest_path = run_dir / "run_manifest.json"
            manifest: Dict[str, Any] = {}
            if manifest_path.exists():
                try:
                    manifest = read_json(manifest_path)
                except Exception:
                    manifest = {}
            timestamp = (
                parse_manifest_time(manifest.get("created_at_utc"))
                or parse_manifest_time(manifest.get("created_at"))
                or raw_path.stat().st_mtime
            )
            candidates[run_dir.resolve()] = {
                "run_dir": run_dir,
                "raw_predictions": raw_path,
                "manifest": manifest_path if manifest_path.exists() else None,
                "completion_score": manifest_completion_score(manifest),
                "timestamp": timestamp,
                "raw_mtime": raw_path.stat().st_mtime,
                "pattern": pattern,
            }
    return sorted(
        candidates.values(),
        key=lambda row: (row["completion_score"], row["timestamp"], row["raw_mtime"], str(row["run_dir"]).lower()),
        reverse=True,
    )


def resolve_run_dir(explicit: Optional[Path]) -> Tuple[Path, List[Dict[str, Any]]]:
    if explicit is not None:
        run_dir = explicit if explicit.is_absolute() else PROJECT_ROOT / explicit
        run_dir = run_dir.resolve()
        if not (run_dir / "raw_phase_predictions.csv").exists():
            raise FileNotFoundError(f"RUN_DIR does not contain raw_phase_predictions.csv: {run_dir}")
        return run_dir, []
    candidates = find_run_candidates()
    if not candidates:
        raise FileNotFoundError("No raw_phase_predictions.csv found under phase_experiments/**")
    return Path(candidates[0]["run_dir"]).resolve(), candidates


def artifact_path(run_dir: Path, name: str) -> Optional[Path]:
    path = run_dir / name
    return path if path.exists() else None


def find_existing_file(config: Mapping[str, Any], config_keys: Sequence[str], fallback_names: Sequence[str]) -> Optional[Path]:
    for key in config_keys:
        value = config.get(key)
        if value:
            p = Path(str(value))
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                return p
    for name in fallback_names:
        hits = sorted((PROJECT_ROOT / "data").rglob(name)) if (PROJECT_ROOT / "data").exists() else []
        if hits:
            return hits[0]
    return None


def normalize_phase_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["gt_phase", RAW_COL, SMOOTH_COL, "frame_idx", "clip_len", "stride"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["gt_phase", RAW_COL, SMOOTH_COL, "frame_idx", "clip_len", "stride"]:
        if col in out.columns:
            out[col] = out[col].astype("Int64")
    if "gt_phase_name" not in out.columns and "gt_phase" in out.columns:
        out["gt_phase_name"] = out["gt_phase"].map(lambda x: phase_name(x) if pd.notna(x) else None)
    for mode, col in PREDICTION_MODES:
        name_col = f"{col}_name"
        if name_col not in out.columns and col in out.columns:
            out[name_col] = out[col].map(lambda x: phase_name(x) if pd.notna(x) else None)
    if "split" not in out.columns:
        out["split"] = "unknown"
    if "type" not in out.columns:
        out["type"] = "unknown"
    if "name" not in out.columns:
        out["name"] = "unknown"
    out["video_key"] = out[["type", "name", "split"]].astype(str).agg(" | ".join, axis=1)
    return out.sort_values(["type", "name", "split", "frame_idx"]).reset_index(drop=True)


def usable_modes(df: pd.DataFrame) -> List[Tuple[str, str]]:
    return [(mode, col) for mode, col in PREDICTION_MODES if col in df.columns]


def required_columns(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[bool, List[str]]:
    missing = [c for c in columns if c not in df.columns]
    return not missing, missing


def phase_metric_rows(df: pd.DataFrame, pred_col: str, mode: str, group: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    work = df.dropna(subset=["gt_phase", pred_col])
    if work.empty:
        return []
    y_true = work["gt_phase"].astype(int).to_numpy()
    y_pred = work[pred_col].astype(int).to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=PHASE_IDS, zero_division=0
    )
    rows: List[Dict[str, Any]] = []
    base = {"mode": mode, "n": int(len(work)), **(dict(group) if group else {})}
    for idx, name in enumerate(PHASE_NAMES):
        rows.append(
            {
                **base,
                "phase_id": idx,
                "phase": name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
                "accuracy_overall": float(accuracy_score(y_true, y_pred)),
                "macro_f1_overall": float(f1_score(y_true, y_pred, average="macro", labels=PHASE_IDS, zero_division=0)),
            }
        )
    return rows


def phase_summary_metrics(df: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    work = df.dropna(subset=["gt_phase", pred_col])
    if work.empty:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    y_true = work["gt_phase"].astype(int)
    y_pred = work[pred_col].astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=PHASE_IDS, zero_division=0)),
    }


def phase_confusion(df: pd.DataFrame, pred_col: str, mode: str) -> pd.DataFrame:
    work = df.dropna(subset=["gt_phase", pred_col])
    cm = confusion_matrix(work["gt_phase"].astype(int), work[pred_col].astype(int), labels=PHASE_IDS)
    rows = []
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.maximum(row_sums, 1), out=np.zeros_like(cm, dtype=float), where=np.maximum(row_sums, 1) > 0)
    for i, true_name in enumerate(PHASE_NAMES):
        for j, pred_name in enumerate(PHASE_NAMES):
            rows.append(
                {
                    "mode": mode,
                    "gt_phase_id": i,
                    "gt_phase": true_name,
                    "pred_phase_id": j,
                    "pred_phase": pred_name,
                    "count": int(cm[i, j]),
                    "normalized_by_gt": float(norm[i, j]),
                }
            )
    return pd.DataFrame(rows)


def plot_grouped_bar(df: pd.DataFrame, x: str, y: str, hue: str, title: str, ylabel: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    piv = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
    piv.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x)
    ax.legend(title=hue)
    ax.grid(axis="y", alpha=0.25)
    return save_fig(fig, path)


def sequence_transitions(sub: pd.DataFrame, col: str) -> List[Dict[str, Any]]:
    clean = sub.dropna(subset=[col, "frame_idx"]).sort_values("frame_idx")
    vals = clean[col].astype(int).to_numpy()
    frames = clean["frame_idx"].astype(int).to_numpy()
    out: List[Dict[str, Any]] = []
    for idx in range(1, len(vals)):
        if vals[idx] != vals[idx - 1]:
            out.append(
                {
                    "frame_idx": int(frames[idx]),
                    "from_phase": int(vals[idx - 1]),
                    "to_phase": int(vals[idx]),
                    "from_phase_name": phase_name(vals[idx - 1]),
                    "to_phase_name": phase_name(vals[idx]),
                    "transition_type": f"{phase_name(vals[idx - 1])}->{phase_name(vals[idx])}",
                }
            )
    return out


def nearest_boundary_distances(sub: pd.DataFrame) -> pd.Series:
    frames = sub["frame_idx"].astype(int).to_numpy()
    transitions = [item["frame_idx"] for item in sequence_transitions(sub, "gt_phase")]
    if not transitions:
        return pd.Series(np.nan, index=sub.index)
    trans = np.asarray(transitions, dtype=int)
    distances = [float(np.min(np.abs(trans - frame))) for frame in frames]
    return pd.Series(distances, index=sub.index)


def contiguous_segments(sub: pd.DataFrame, col: str, phase_id: int) -> List[Dict[str, Any]]:
    clean = sub.dropna(subset=[col, "frame_idx"]).sort_values("frame_idx")
    vals = clean[col].astype(int).to_numpy()
    frames = clean["frame_idx"].astype(int).to_numpy()
    segments: List[Dict[str, Any]] = []
    start_idx: Optional[int] = None
    for i, value in enumerate(vals):
        if value == phase_id and start_idx is None:
            start_idx = i
        if start_idx is not None and (value != phase_id or i == len(vals) - 1):
            end_idx = i - 1 if value != phase_id else i
            if end_idx >= start_idx:
                segments.append(
                    {
                        "segment_order": len(segments) + 1,
                        "start_frame": int(frames[start_idx]),
                        "end_frame": int(frames[end_idx]),
                        "sample_count": int(end_idx - start_idx + 1),
                        "frame_span": int(frames[end_idx] - frames[start_idx] + 1),
                    }
                )
            start_idx = None
    return segments


def parse_label_row(row: pd.Series, max_reps: int = 80) -> List[Tuple[int, int, int]]:
    reps: List[Tuple[int, int, int]] = []
    for i in range(max_reps):
        s_col, b_col, f_col = f"L{3 * i + 1}", f"L{3 * i + 2}", f"L{3 * i + 3}"
        if s_col not in row.index:
            break
        s, b, f = row.get(s_col), row.get(b_col), row.get(f_col)
        if pd.isna(s) and pd.isna(b) and pd.isna(f):
            break
        try:
            s_i, b_i, f_i = int(float(s)), int(float(b)), int(float(f))
        except Exception:
            break
        if f_i > s_i:
            reps.append((s_i, b_i, f_i))
    return reps


def load_label_reps(labels_path: Optional[Path]) -> Dict[Tuple[str, str], List[Tuple[int, int, int]]]:
    if labels_path is None or not labels_path.exists():
        return {}
    labels_df = read_csv(labels_path)
    labels_df.columns = [str(c).strip() for c in labels_df.columns]
    labels_df = labels_df.loc[:, ~labels_df.columns.str.startswith("Unnamed")]
    reps: Dict[Tuple[str, str], List[Tuple[int, int, int]]] = {}
    if {"type", "name"}.issubset(labels_df.columns):
        for _, row in labels_df.iterrows():
            reps[(str(row["type"]).strip(), str(row["name"]).strip())] = parse_label_row(row)
    return reps


def assign_rep_order(df: pd.DataFrame, reps_by_video: Mapping[Tuple[str, str], List[Tuple[int, int, int]]]) -> pd.Series:
    rep_labels = pd.Series(pd.NA, index=df.index, dtype="object")
    for (typ, name, split), sub in df.groupby(["type", "name", "split"], dropna=False):
        reps = reps_by_video.get((str(typ), str(name)), [])
        if not reps:
            continue
        for rep_idx, (start, _bottom, finish) in enumerate(reps, start=1):
            mask = sub["frame_idx"].astype(int).between(int(start), int(finish))
            if mask.any():
                total = len(reps)
                if rep_idx == 1:
                    bucket = "first"
                elif rep_idx == total:
                    bucket = "last"
                else:
                    bucket = "middle"
                rep_labels.loc[sub.index[mask]] = f"{rep_idx}:{bucket}"
    return rep_labels


def add_status_text(lines: List[str], title: str, status: str, reason: str) -> None:
    lines.append(f"### {title}\n\n- status: `{status}`\n- reason/caveat: {reason}\n")


@dataclass
class AnalysisRecorder:
    output_dir: Path
    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    narrative_lines: List[str] = field(default_factory=list)

    def record(
        self,
        diag_id: str,
        name: str,
        status_or_cell_name: str,
        reason_or_status: str,
        reason: Optional[str] = None,
        *,
        cell_name: Optional[str] = None,
        inputs: Optional[Sequence[Path | str]] = None,
        outputs: Optional[Sequence[Path | str]] = None,
        caveat: str = "",
    ) -> None:
        # Most calls use the explicit form:
        #   record(id, name, status, reason, cell_name="...")
        # Some notebook-section calls are easier to read as:
        #   record(id, name, cell_name, status, reason)
        # Support both to keep D01-D13 sections concise and robust.
        if reason is None:
            status = status_or_cell_name
            resolved_reason = reason_or_status
            resolved_cell_name = cell_name or ""
        else:
            resolved_cell_name = status_or_cell_name
            status = reason_or_status
            resolved_reason = reason
        inputs_s = [str(x) for x in (inputs or [])]
        outputs_s = [str(x) for x in (outputs or [])]
        self.entries[diag_id] = {
            "diagnostic_id": diag_id,
            "name": name,
            "cell_name": resolved_cell_name,
            "status": status,
            "reason": resolved_reason,
            "caveat": caveat,
            "inputs": inputs_s,
            "outputs": outputs_s,
        }
        if status != "immediate" or caveat:
            add_status_text(self.narrative_lines, f"{diag_id} {name}", status, caveat or resolved_reason)

    def write_status_csv(self) -> Path:
        rows = []
        for diag_id in [f"D{i:02d}" for i in range(1, 14)]:
            row = self.entries.get(
                diag_id,
                {
                    "diagnostic_id": diag_id,
                    "name": "",
                    "cell_name": "",
                    "status": "unavailable",
                    "reason": "diagnostic did not run",
                    "caveat": "",
                    "inputs": [],
                    "outputs": [],
                },
            )
            rows.append({**row, "inputs": json.dumps(row.get("inputs", []), ensure_ascii=False), "outputs": json.dumps(row.get("outputs", []), ensure_ascii=False)})
        return write_csv(pd.DataFrame(rows), self.output_dir / "diagnostic_status.csv")

    def write_narrative(self) -> Path:
        text = "# Phase Error Analysis Notes\n\n"
        if self.narrative_lines:
            text += "\n".join(self.narrative_lines)
        else:
            text += "All diagnostics completed without degraded status caveats.\n"
        path = self.output_dir / "phase_error_analysis_notes.md"
        path.write_text(text, encoding="utf-8")
        return path


# %% Run discovery / artifact inventory / schema validation
RUN_DIR_RESOLVED, RUN_CANDIDATES = resolve_run_dir(RUN_DIR)
OUTPUT_DIR = RUN_DIR_RESOLVED / "phase_error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for stale_error in OUTPUT_DIR.glob("D*_error.txt"):
    stale_error.unlink(missing_ok=True)
RECORDER = AnalysisRecorder(OUTPUT_DIR)

RAW_PATH = RUN_DIR_RESOLVED / "raw_phase_predictions.csv"
CONFIG_PATH = artifact_path(RUN_DIR_RESOLVED, "config.json")
RUN_MANIFEST_PATH = artifact_path(RUN_DIR_RESOLVED, "run_manifest.json")
HISTORY_PATH = artifact_path(RUN_DIR_RESOLVED, "history.json")

RUN_MANIFEST = read_json(RUN_MANIFEST_PATH) if RUN_MANIFEST_PATH else {}
CONFIG = read_json(CONFIG_PATH) if CONFIG_PATH else RUN_MANIFEST.get("config", {})

META_PATH = find_existing_file(CONFIG, ["meta_csv", "team_meta_csv"], ["meta_stgcn_lstm.csv", "meta_v4.csv"])
LABELS_PATH = find_existing_file(CONFIG, ["labels_csv"], ["labels.csv"])

RAW_DF = normalize_phase_columns(read_csv(RAW_PATH))
if RAW_DF.empty:
    raise RuntimeError(f"raw predictions are empty: {RAW_PATH}")

MODE_COLUMNS = usable_modes(RAW_DF)
FRESH_INFERENCE_ENABLED = False

ARTIFACT_INVENTORY = {
    "run_dir": rel(RUN_DIR_RESOLVED),
    "raw_predictions": rel(RAW_PATH),
    "config": rel(CONFIG_PATH) if CONFIG_PATH else None,
    "run_manifest": rel(RUN_MANIFEST_PATH) if RUN_MANIFEST_PATH else None,
    "history": rel(HISTORY_PATH) if HISTORY_PATH else None,
    "meta": rel(META_PATH) if META_PATH else None,
    "labels": rel(LABELS_PATH) if LABELS_PATH else None,
    "mode_columns": MODE_COLUMNS,
}

schema_rows = []
for column in [
    "type",
    "name",
    "split",
    "frame_idx",
    "clip_len",
    "stride",
    "gt_phase",
    RAW_COL,
    SMOOTH_COL,
    "boundary_distance",
    "action_pred",
]:
    schema_rows.append({"column": column, "present": column in RAW_DF.columns})
SCHEMA_PATH = write_csv(pd.DataFrame(schema_rows), OUTPUT_DIR / "schema_validation.csv")

print("Phase error analysis")
print("RUN_DIR:", RUN_DIR_RESOLVED)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("Rows:", len(RAW_DF), "| Modes:", MODE_COLUMNS)


# %% Valid-clip coverage caveat
coverage_rows = []
for (typ, name, split), sub in RAW_DF.groupby(["type", "name", "split"], dropna=False):
    clip_len = int(sub["clip_len"].dropna().iloc[0]) if "clip_len" in sub.columns and sub["clip_len"].notna().any() else None
    stride = int(sub["stride"].dropna().iloc[0]) if "stride" in sub.columns and sub["stride"].notna().any() else None
    coverage_rows.append(
        {
            "type": typ,
            "name": name,
            "split": split,
            "n_rows": int(len(sub)),
            "min_frame_idx": int(sub["frame_idx"].min()),
            "max_frame_idx": int(sub["frame_idx"].max()),
            "clip_len": clip_len,
            "stride": stride,
            "expected_first_valid_frame": clip_len - 1 if clip_len is not None else None,
            "valid_clip_frame_caveat": "Rows are model output/evaluation frames, not necessarily every original video frame.",
        }
    )
COVERAGE_DF = pd.DataFrame(coverage_rows)
COVERAGE_PATH = write_csv(COVERAGE_DF, OUTPUT_DIR / "valid_clip_frame_coverage.csv")
RECORDER.narrative_lines.append(
    "## Valid-frame caveat\n\n"
    "The raw prediction artifact contains evaluated/valid model-output frames. "
    "For causal window models this usually starts at `clip_len - 1` and may be sampled by stride; "
    "timeline, transition, rep-order, and segment-length diagnostics should be read over these diagnostic frames, not as full original-frame coverage.\n"
)


def run_diagnostic(diag_id: str, name: str, cell_name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception as exc:  # keep notebook progressing with structured status.
        error_path = OUTPUT_DIR / f"{diag_id}_error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        RECORDER.record(
            diag_id,
            name,
            "unavailable",
            f"diagnostic raised {type(exc).__name__}: {exc}",
            cell_name=cell_name,
            inputs=[RAW_PATH],
            outputs=[error_path],
            caveat="The diagnostic failed but the notebook continued; inspect the error artifact.",
        )


# %% D01 Phase distribution
def d01_phase_distribution() -> None:
    ok, missing = required_columns(RAW_DF, ["gt_phase", RAW_COL])
    if not ok:
        RECORDER.record("D01", "phase distribution", "D01 Phase distribution", "unavailable", f"missing columns: {missing}", inputs=[RAW_PATH])
        return

    rows: List[Dict[str, Any]] = []
    sources = [("gt", "gt_phase")] + [(mode, col) for mode, col in MODE_COLUMNS]
    for scope_name, scope_df in [("overall", RAW_DF)] + [(f"type={typ}", sub) for typ, sub in RAW_DF.groupby("type")]:
        for source, col in sources:
            counts = scope_df[col].dropna().astype(int).value_counts().reindex(PHASE_IDS, fill_value=0)
            total = int(counts.sum())
            for pid, count in counts.items():
                rows.append(
                    {
                        "scope": scope_name,
                        "source": source,
                        "phase_id": int(pid),
                        "phase": phase_name(pid),
                        "count": int(count),
                        "ratio": float(count / total) if total else float("nan"),
                    }
                )
    dist_df = pd.DataFrame(rows)
    csv_path = write_csv(dist_df, OUTPUT_DIR / "D01_phase_distribution.csv")

    plot_df = dist_df[dist_df["scope"] == "overall"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    piv = plot_df.pivot_table(index="phase", columns="source", values="ratio", aggfunc="sum").reindex(PHASE_NAMES)
    piv.plot(kind="bar", ax=ax)
    ax.set_title("D01 Phase distribution ratio")
    ax.set_ylabel("ratio")
    ax.grid(axis="y", alpha=0.25)
    png_path = save_fig(fig, OUTPUT_DIR / "D01_phase_distribution.png")

    RECORDER.record(
        "D01",
        "phase distribution",
        "D01 Phase distribution",
        "immediate",
        "computed from existing raw prediction rows",
        inputs=[RAW_PATH],
        outputs=[csv_path, png_path],
    )


run_diagnostic("D01", "phase distribution", "D01 Phase distribution", d01_phase_distribution)


# %% D02 Per-phase F1
def d02_per_phase_f1() -> None:
    ok, missing = required_columns(RAW_DF, ["gt_phase", RAW_COL])
    if not ok:
        RECORDER.record("D02", "per-phase F1", "D02 Per-phase F1", "unavailable", f"missing columns: {missing}", inputs=[RAW_PATH])
        return

    rows: List[Dict[str, Any]] = []
    for mode, col in MODE_COLUMNS:
        rows.extend(phase_metric_rows(RAW_DF, col, mode))
    f1_df = pd.DataFrame(rows)
    csv_path = write_csv(f1_df, OUTPUT_DIR / "D02_per_phase_f1.csv")
    png_path = plot_grouped_bar(
        f1_df,
        x="phase",
        y="f1",
        hue="mode",
        title="D02 Per-phase F1 (average=None)",
        ylabel="F1",
        path=OUTPUT_DIR / "D02_per_phase_f1_raw_vs_smooth.png",
    )
    RECORDER.record(
        "D02",
        "phase별 F1 (average=None)",
        "D02 Per-phase F1",
        "immediate",
        "computed per phase for raw and offline_smooth where available",
        inputs=[RAW_PATH],
        outputs=[csv_path, png_path],
        caveat="offline_smooth is post-hoc/non-causal and should not be read as real-time performance.",
    )


run_diagnostic("D02", "phase별 F1 (average=None)", "D02 Per-phase F1", d02_per_phase_f1)


# %% D03 Confusion matrix
def d03_confusion_matrix() -> None:
    ok, missing = required_columns(RAW_DF, ["gt_phase", RAW_COL])
    if not ok:
        RECORDER.record("D03", "Confusion Matrix", "D03 Confusion matrix", "unavailable", f"missing columns: {missing}", inputs=[RAW_PATH])
        return

    rows = [phase_confusion(RAW_DF, col, mode) for mode, col in MODE_COLUMNS]
    cm_df = pd.concat(rows, ignore_index=True)
    csv_path = write_csv(cm_df, OUTPUT_DIR / "D03_confusion_matrix.csv")

    modes = cm_df["mode"].unique().tolist()
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4.5), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        sub = cm_df[cm_df["mode"] == mode]
        mat = sub.pivot_table(index="gt_phase", columns="pred_phase", values="normalized_by_gt", aggfunc="sum").reindex(index=PHASE_NAMES, columns=PHASE_NAMES)
        im = ax.imshow(mat.fillna(0).to_numpy(), vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"{mode} normalized by GT")
        ax.set_xticks(range(len(PHASE_NAMES)), PHASE_NAMES, rotation=30)
        ax.set_yticks(range(len(PHASE_NAMES)), PHASE_NAMES)
        for i in range(len(PHASE_NAMES)):
            for j in range(len(PHASE_NAMES)):
                ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    png_path = save_fig(fig, OUTPUT_DIR / "D03_confusion_matrix.png")

    RECORDER.record(
        "D03",
        "Confusion Matrix",
        "D03 Confusion matrix",
        "immediate",
        "computed confusion matrices from existing prediction rows",
        inputs=[RAW_PATH],
        outputs=[csv_path, png_path],
    )


run_diagnostic("D03", "Confusion Matrix", "D03 Confusion matrix", d03_confusion_matrix)


# %% D04 Timeline visualization
def d04_timeline_visualization() -> None:
    ok, missing = required_columns(RAW_DF, ["type", "name", "split", "frame_idx", "gt_phase", RAW_COL])
    if not ok:
        RECORDER.record("D04", "시간축 시각화", "D04 Timeline visualization", "unavailable", f"missing columns: {missing}", inputs=[RAW_PATH])
        return

    work = RAW_DF.dropna(subset=["gt_phase", RAW_COL, "frame_idx"]).copy()
    work["is_error_raw"] = work["gt_phase"].astype(int) != work[RAW_COL].astype(int)
    video_summary = (
        work.groupby(["type", "name", "split"], dropna=False)
        .agg(n_rows=("frame_idx", "size"), raw_error_rate=("is_error_raw", "mean"), min_frame=("frame_idx", "min"), max_frame=("frame_idx", "max"))
        .reset_index()
        .sort_values(["raw_error_rate", "n_rows"], ascending=[False, False])
    )
    selection = video_summary.head(N_TIMELINE_VIDEOS)
    selection_path = write_csv(selection, OUTPUT_DIR / "D04_timeline_examples.csv")

    timeline_dir = OUTPUT_DIR / "D04_timeline_examples"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: List[Path] = []
    fig, axes = plt.subplots(max(len(selection), 1), 1, figsize=(12, 3.2 * max(len(selection), 1)), squeeze=False)
    if selection.empty:
        axes[0, 0].text(0.5, 0.5, "No videos available", ha="center", va="center")
    for ax_idx, (_, row) in enumerate(selection.iterrows()):
        sub = work[
            (work["type"] == row["type"]) & (work["name"] == row["name"]) & (work["split"] == row["split"])
        ].sort_values("frame_idx")
        frames = sub["frame_idx"].astype(int)
        ax = axes[ax_idx, 0]
        ax.step(frames, sub["gt_phase"].astype(int), where="post", label="GT", linewidth=2.0)
        ax.step(frames, sub[RAW_COL].astype(int) + 0.06, where="post", label="raw", alpha=0.8)
        if SMOOTH_COL in sub.columns:
            ax.step(frames, sub[SMOOTH_COL].astype(int) - 0.06, where="post", label="offline_smooth", alpha=0.8)
        err = sub[sub["is_error_raw"]]
        ax.scatter(err["frame_idx"], err["gt_phase"].astype(int), s=12, c="red", label="raw error", alpha=0.55)
        ax.set_yticks(PHASE_IDS, PHASE_NAMES)
        ax.set_title(f"{row['type']} / {row['name']} / {row['split']} | raw_error={row['raw_error_rate']:.3f}")
        ax.set_xlabel("frame_idx (valid diagnostic frames)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", ncol=4)

        indiv_fig, indiv_ax = plt.subplots(figsize=(12, 3.2))
        indiv_ax.step(frames, sub["gt_phase"].astype(int), where="post", label="GT", linewidth=2.0)
        indiv_ax.step(frames, sub[RAW_COL].astype(int) + 0.06, where="post", label="raw", alpha=0.8)
        if SMOOTH_COL in sub.columns:
            indiv_ax.step(frames, sub[SMOOTH_COL].astype(int) - 0.06, where="post", label="offline_smooth", alpha=0.8)
        indiv_ax.scatter(err["frame_idx"], err["gt_phase"].astype(int), s=12, c="red", label="raw error", alpha=0.55)
        indiv_ax.set_yticks(PHASE_IDS, PHASE_NAMES)
        indiv_ax.set_title(ax.get_title())
        indiv_ax.set_xlabel("frame_idx (valid diagnostic frames)")
        indiv_ax.grid(alpha=0.25)
        indiv_ax.legend(loc="upper right", ncol=4)
        plot_paths.append(save_fig(indiv_fig, timeline_dir / f"{safe_name(row['type'])}_{safe_name(row['name'])}_{safe_name(row['split'])}.png"))
    combined_path = save_fig(fig, OUTPUT_DIR / "D04_timeline_examples.png")
    RECORDER.record(
        "D04",
        "시간축 시각화",
        "D04 Timeline visualization",
        "immediate",
        "selected worst videos by raw error rate and plotted GT/raw/offline_smooth over valid diagnostic frames",
        inputs=[RAW_PATH],
        outputs=[selection_path, combined_path, *plot_paths],
        caveat="Timeline rows are valid model-output frames; this is not guaranteed to be every original frame.",
    )


run_diagnostic("D04", "시간축 시각화", "D04 Timeline visualization", d04_timeline_visualization)


# %% D05 Transition timing error
def d05_transition_timing_error() -> None:
    ok, missing = required_columns(RAW_DF, ["type", "name", "split", "frame_idx", "gt_phase", RAW_COL])
    if not ok:
        RECORDER.record("D05", "전환 타이밍 오차", "D05 Transition timing error", "unavailable", f"missing columns: {missing}", inputs=[RAW_PATH])
        return

    detail_rows: List[Dict[str, Any]] = []
    for (typ, name, split), sub in RAW_DF.groupby(["type", "name", "split"], dropna=False):
        sub = sub.sort_values("frame_idx")
        gt_transitions = sequence_transitions(sub, "gt_phase")
        for mode, col in MODE_COLUMNS:
            pred_transitions = sequence_transitions(sub, col)
            used: set[int] = set()
            for gt_idx, gt in enumerate(gt_transitions):
                candidates = [
                    (idx, pred)
                    for idx, pred in enumerate(pred_transitions)
                    if idx not in used and pred["transition_type"] == gt["transition_type"]
                ]
                match_kind = "same_transition_type"
                if not candidates:
                    candidates = [
                        (idx, pred)
                        for idx, pred in enumerate(pred_transitions)
                        if idx not in used and pred["to_phase"] == gt["to_phase"]
                    ]
                    match_kind = "same_target_phase"
                if not candidates:
                    detail_rows.append(
                        {
                            "type": typ,
                            "name": name,
                            "split": split,
                            "mode": mode,
                            "gt_transition_order": gt_idx + 1,
                            "gt_frame": gt["frame_idx"],
                            "gt_transition_type": gt["transition_type"],
                            "pred_frame": np.nan,
                            "pred_transition_type": None,
                            "signed_error_frames": np.nan,
                            "abs_error_frames": np.nan,
                            "match_kind": "missed",
                            "within_tolerance": False,
                        }
                    )
                    continue
                pred_idx, pred = min(candidates, key=lambda item: abs(item[1]["frame_idx"] - gt["frame_idx"]))
                used.add(pred_idx)
                signed = int(pred["frame_idx"] - gt["frame_idx"])
                detail_rows.append(
                    {
                        "type": typ,
                        "name": name,
                        "split": split,
                        "mode": mode,
                        "gt_transition_order": gt_idx + 1,
                        "gt_frame": gt["frame_idx"],
                        "gt_transition_type": gt["transition_type"],
                        "pred_frame": pred["frame_idx"],
                        "pred_transition_type": pred["transition_type"],
                        "signed_error_frames": signed,
                        "abs_error_frames": abs(signed),
                        "match_kind": match_kind,
                        "within_tolerance": abs(signed) <= TRANSITION_MATCH_TOLERANCE_FRAMES,
                    }
                )

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        RECORDER.record("D05", "전환 타이밍 오차", "D05 Transition timing error", "unavailable", "no GT transitions found", inputs=[RAW_PATH])
        return
    detail_path = write_csv(detail, OUTPUT_DIR / "D05_transition_timing_error.csv")
    summary = (
        detail.groupby("mode")
        .agg(
            n_gt_transitions=("gt_frame", "size"),
            n_matched=("pred_frame", lambda s: int(s.notna().sum())),
            missed_rate=("pred_frame", lambda s: float(s.isna().mean())),
            mean_signed_error_frames=("signed_error_frames", "mean"),
            median_signed_error_frames=("signed_error_frames", "median"),
            mean_abs_error_frames=("abs_error_frames", "mean"),
            late_rate=("signed_error_frames", lambda s: float((s.dropna() > 0).mean()) if s.notna().any() else float("nan")),
            early_rate=("signed_error_frames", lambda s: float((s.dropna() < 0).mean()) if s.notna().any() else float("nan")),
            within_tolerance_rate=("within_tolerance", "mean"),
        )
        .reset_index()
    )
    summary_path = write_csv(summary, OUTPUT_DIR / "D05_transition_timing_error_summary.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for mode, sub in detail.dropna(subset=["signed_error_frames"]).groupby("mode"):
        ax.hist(sub["signed_error_frames"], bins=30, alpha=0.55, label=mode)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("D05 Transition timing signed error (pred_frame - gt_frame)")
    ax.set_xlabel("signed error in frames; positive = prediction late")
    ax.set_ylabel("count")
    ax.legend()
    png_path = save_fig(fig, OUTPUT_DIR / "D05_transition_timing_error.png")
    RECORDER.record(
        "D05",
        "전환 타이밍 오차",
        "D05 Transition timing error",
        "approximate",
        "derived by matching GT and predicted transitions on existing valid diagnostic frames",
        inputs=[RAW_PATH],
        outputs=[detail_path, summary_path, png_path],
        caveat="Approximate because transitions are computed over sampled valid model-output frames. Positive signed error means prediction is late.",
    )


run_diagnostic("D05", "전환 타이밍 오차", "D05 Transition timing error", d05_transition_timing_error)


# %% D06 Boundary-vs-middle accuracy
def d06_boundary_vs_middle_accuracy() -> None:
    ok, missing = required_columns(RAW_DF, ["frame_idx", "gt_phase", RAW_COL])
    if not ok:
        RECORDER.record("D06", "경계 vs 중간 정확도 비교", "D06 Boundary-vs-middle accuracy", "unavailable", f"missing columns: {missing}", inputs=[RAW_PATH])
        return

    work = RAW_DF.copy()
    distance_source = "boundary_distance"
    if "boundary_distance" not in work.columns or work["boundary_distance"].isna().all():
        work["_boundary_distance_derived"] = np.nan
        for _key, sub in work.groupby(["type", "name", "split"], dropna=False):
            work.loc[sub.index, "_boundary_distance_derived"] = nearest_boundary_distances(sub)
        distance_col = "_boundary_distance_derived"
        distance_source = "derived_from_gt_transitions"
    else:
        distance_col = "boundary_distance"

    finite = work[np.isfinite(pd.to_numeric(work[distance_col], errors="coerce"))].copy()
    if finite.empty:
        RECORDER.record("D06", "경계 vs 중간 정확도 비교", "D06 Boundary-vs-middle accuracy", "unavailable", "no finite boundary distance", inputs=[RAW_PATH])
        return
    finite["boundary_bin"] = np.where(finite[distance_col].astype(float) <= BOUNDARY_WINDOW, "boundary", "middle")
    rows = []
    for mode, col in MODE_COLUMNS:
        for boundary_bin, sub in finite.groupby("boundary_bin"):
            metrics = phase_summary_metrics(sub, col)
            rows.append(
                {
                    "mode": mode,
                    "boundary_bin": boundary_bin,
                    "boundary_window_frames": BOUNDARY_WINDOW,
                    "distance_source": distance_source,
                    "n": int(len(sub)),
                    **metrics,
                }
            )
    out = pd.DataFrame(rows)
    csv_path = write_csv(out, OUTPUT_DIR / "D06_boundary_vs_middle_accuracy.csv")
    png_path = plot_grouped_bar(out, "boundary_bin", "accuracy", "mode", "D06 Boundary vs middle accuracy", "accuracy", OUTPUT_DIR / "D06_boundary_vs_middle_accuracy.png")
    RECORDER.record(
        "D06",
        "경계 vs 중간 정확도 비교",
        "D06 Boundary-vs-middle accuracy",
        "immediate",
        f"computed using {distance_source}",
        inputs=[RAW_PATH],
        outputs=[csv_path, png_path],
        caveat=f"boundary bin is distance <= {BOUNDARY_WINDOW} valid/evaluated frames.",
    )


run_diagnostic("D06", "경계 vs 중간 정확도 비교", "D06 Boundary-vs-middle accuracy", d06_boundary_vs_middle_accuracy)


# %% D07 Per-exercise phase F1
def d07_per_exercise_phase_f1() -> None:
    ok, missing = required_columns(RAW_DF, ["type", "gt_phase", RAW_COL])
    if not ok:
        RECORDER.record("D07", "종목별 phase F1", "D07 Per-exercise phase F1", "unavailable", f"missing columns: {missing}", inputs=[RAW_PATH])
        return

    rows: List[Dict[str, Any]] = []
    for typ, sub in RAW_DF.groupby("type", dropna=False):
        for mode, col in MODE_COLUMNS:
            rows.extend(phase_metric_rows(sub, col, mode, group={"type": typ}))
    per_ex = pd.DataFrame(rows)
    csv_path = write_csv(per_ex, OUTPUT_DIR / "D07_per_exercise_phase_f1.csv")
    macro_rows = []
    for typ, sub in RAW_DF.groupby("type", dropna=False):
        for mode, col in MODE_COLUMNS:
            metrics = phase_summary_metrics(sub, col)
            macro_rows.append({"type": typ, "mode": mode, "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"], "n": int(len(sub))})
    macro_df = pd.DataFrame(macro_rows)
    png_path = plot_grouped_bar(macro_df, "type", "macro_f1", "mode", "D07 Per-exercise phase macro F1", "macro F1", OUTPUT_DIR / "D07_per_exercise_phase_f1.png")

    outputs: List[Path] = [csv_path, png_path]
    if "action_pred" in RAW_DF.columns:
        action = RAW_DF.copy()
        action["action_matches_gt_type"] = action["action_pred"].astype(str) == action["type"].astype(str)
        action_rows = []
        for (typ, match), sub in action.groupby(["type", "action_matches_gt_type"], dropna=False):
            for mode, col in MODE_COLUMNS:
                metrics = phase_summary_metrics(sub, col)
                action_rows.append({"type": typ, "action_matches_gt_type": bool(match), "mode": mode, "n": int(len(sub)), **metrics})
        action_path = write_csv(pd.DataFrame(action_rows), OUTPUT_DIR / "D07_action_mismatch_slice.csv")
        outputs.append(action_path)
    RECORDER.record(
        "D07",
        "종목별 phase F1",
        "D07 Per-exercise phase F1",
        "immediate",
        "computed by GT type; action mismatch is a separate slice when action_pred exists",
        inputs=[RAW_PATH],
        outputs=outputs,
    )


run_diagnostic("D07", "종목별 phase F1", "D07 Per-exercise phase F1", d07_per_exercise_phase_f1)


# %% D08 Per-rep-order performance
def d08_per_rep_order_performance() -> None:
    ok, missing = required_columns(RAW_DF, ["type", "name", "split", "frame_idx", "gt_phase", RAW_COL])
    if not ok:
        RECORDER.record("D08", "영상 내 rep 순서별 성능", "D08 Per-rep-order performance", "unavailable", f"missing columns: {missing}", inputs=[RAW_PATH])
        return

    reps_by_video = load_label_reps(LABELS_PATH)
    if not reps_by_video:
        RECORDER.record(
            "D08",
            "영상 내 rep 순서별 성능",
            "D08 Per-rep-order performance",
            "unavailable",
            "label rep intervals were not found; refusing prediction-only rep inference",
            inputs=[RAW_PATH],
            caveat="This diagnostic intentionally does not infer reps from predictions.",
        )
        return

    work = RAW_DF.copy()
    work["rep_label"] = assign_rep_order(work, reps_by_video)
    assigned = work[work["rep_label"].notna()].copy()
    if assigned.empty:
        RECORDER.record("D08", "영상 내 rep 순서별 성능", "D08 Per-rep-order performance", "unavailable", "no frames matched GT rep intervals", inputs=[RAW_PATH, LABELS_PATH or ""])
        return
    assigned["rep_order"] = assigned["rep_label"].astype(str).str.extract(r":(.+)$")[0]
    rows = []
    for (typ, rep_order), sub in assigned.groupby(["type", "rep_order"], dropna=False):
        for mode, col in MODE_COLUMNS:
            metrics = phase_summary_metrics(sub, col)
            rows.append({"type": typ, "rep_order": rep_order, "mode": mode, "n": int(len(sub)), **metrics})
    out = pd.DataFrame(rows)
    csv_path = write_csv(out, OUTPUT_DIR / "D08_per_rep_order_performance.csv")
    png_path = plot_grouped_bar(out, "rep_order", "accuracy", "mode", "D08 Rep-order accuracy", "accuracy", OUTPUT_DIR / "D08_per_rep_order_performance.png")
    RECORDER.record(
        "D08",
        "영상 내 rep 순서별 성능",
        "D08 Per-rep-order performance",
        "approximate",
        "frames assigned to GT label intervals, then grouped into first/middle/last reps",
        inputs=[RAW_PATH, LABELS_PATH or ""],
        outputs=[csv_path, png_path],
        caveat="Approximate because raw predictions cover valid diagnostic frames only; rep assignment uses GT label intervals, not predictions.",
    )


run_diagnostic("D08", "영상 내 rep 순서별 성능", "D08 Per-rep-order performance", d08_per_rep_order_performance)


# %% D09 Up-segment length error
def d09_up_segment_length_error() -> None:
    ok, missing = required_columns(RAW_DF, ["type", "name", "split", "frame_idx", "gt_phase", RAW_COL])
    if not ok:
        RECORDER.record("D09", "up 구간 길이 오차", "D09 Up-segment length error", "unavailable", f"missing columns: {missing}", inputs=[RAW_PATH])
        return

    rows = []
    for (typ, name, split), sub in RAW_DF.groupby(["type", "name", "split"], dropna=False):
        sub = sub.sort_values("frame_idx")
        gt_segments = contiguous_segments(sub, "gt_phase", PHASE_ID_BY_NAME["up"])
        for mode, col in MODE_COLUMNS:
            pred_segments = contiguous_segments(sub, col, PHASE_ID_BY_NAME["up"])
            max_len = max(len(gt_segments), len(pred_segments))
            for idx in range(max_len):
                gt = gt_segments[idx] if idx < len(gt_segments) else None
                pred = pred_segments[idx] if idx < len(pred_segments) else None
                gt_span = gt["frame_span"] if gt else np.nan
                pred_span = pred["frame_span"] if pred else np.nan
                rows.append(
                    {
                        "type": typ,
                        "name": name,
                        "split": split,
                        "mode": mode,
                        "segment_order": idx + 1,
                        "gt_start_frame": gt["start_frame"] if gt else np.nan,
                        "gt_end_frame": gt["end_frame"] if gt else np.nan,
                        "pred_start_frame": pred["start_frame"] if pred else np.nan,
                        "pred_end_frame": pred["end_frame"] if pred else np.nan,
                        "gt_frame_span": gt_span,
                        "pred_frame_span": pred_span,
                        "length_error_frames": pred_span - gt_span if gt and pred else np.nan,
                        "abs_length_error_frames": abs(pred_span - gt_span) if gt and pred else np.nan,
                        "missing_pred_up_segment": bool(gt is not None and pred is None),
                        "extra_pred_up_segment": bool(gt is None and pred is not None),
                    }
                )
    detail = pd.DataFrame(rows)
    if detail.empty:
        RECORDER.record("D09", "up 구간 길이 오차", "D09 Up-segment length error", "unavailable", "no up segments found", inputs=[RAW_PATH])
        return
    detail_path = write_csv(detail, OUTPUT_DIR / "D09_up_segment_length_error.csv")
    summary = (
        detail.groupby("mode")
        .agg(
            n_segment_rows=("segment_order", "size"),
            mean_length_error_frames=("length_error_frames", "mean"),
            mean_abs_length_error_frames=("abs_length_error_frames", "mean"),
            missing_pred_up_segments=("missing_pred_up_segment", "sum"),
            extra_pred_up_segments=("extra_pred_up_segment", "sum"),
        )
        .reset_index()
    )
    summary_path = write_csv(summary, OUTPUT_DIR / "D09_up_segment_length_error_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode, sub in detail.dropna(subset=["length_error_frames"]).groupby("mode"):
        ax.hist(sub["length_error_frames"], bins=30, alpha=0.55, label=mode)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("D09 Up-segment length error")
    ax.set_xlabel("predicted span - GT span (frames)")
    ax.legend()
    png_path = save_fig(fig, OUTPUT_DIR / "D09_up_segment_length_error.png")
    RECORDER.record(
        "D09",
        "up 구간 길이 오차",
        "D09 Up-segment length error",
        "approximate",
        "matched GT and predicted up segments by order using existing valid-frame sequences",
        inputs=[RAW_PATH],
        outputs=[detail_path, summary_path, png_path],
        caveat="Frame-span units come from diagnostic frame indices; no FPS conversion is applied silently.",
    )


run_diagnostic("D09", "up 구간 길이 오차", "D09 Up-segment length error", d09_up_segment_length_error)


# %% D10 FPS distribution
def d10_fps_distribution() -> None:
    if META_PATH is None or not META_PATH.exists():
        RECORDER.record(
            "D10",
            "FPS 분포 확인",
            "D10 FPS distribution",
            "unavailable",
            "no metadata CSV containing FPS was found",
            inputs=[RAW_PATH],
            caveat="No silent 30 FPS fallback is used.",
        )
        return
    meta = read_csv(META_PATH)
    if "fps" not in meta.columns:
        RECORDER.record("D10", "FPS 분포 확인", "D10 FPS distribution", "unavailable", f"metadata has no fps column: {META_PATH}", inputs=[META_PATH])
        return
    join_cols = [c for c in ["type", "name", "split"] if c in meta.columns and c in RAW_DF.columns]
    if not {"type", "name"}.issubset(join_cols):
        RECORDER.record("D10", "FPS 분포 확인", "D10 FPS distribution", "unavailable", "metadata cannot join on type/name", inputs=[RAW_PATH, META_PATH])
        return
    videos = RAW_DF[["type", "name", "split"]].drop_duplicates()
    joined = videos.merge(meta[[*join_cols, "fps"]].drop_duplicates(), on=join_cols, how="left")
    joined["fps_source"] = rel(META_PATH)
    if joined["fps"].notna().sum() == 0:
        RECORDER.record("D10", "FPS 분포 확인", "D10 FPS distribution", "unavailable", "join found no FPS values", inputs=[RAW_PATH, META_PATH], caveat="No silent 30 FPS fallback is used.")
        return
    csv_path = write_csv(joined, OUTPUT_DIR / "D10_fps_distribution.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(joined["fps"].dropna(), bins=20, alpha=0.75)
    ax.set_title("D10 FPS distribution from existing metadata")
    ax.set_xlabel("FPS")
    ax.set_ylabel("video count")
    ax.grid(axis="y", alpha=0.25)
    png_path = save_fig(fig, OUTPUT_DIR / "D10_fps_distribution.png")
    RECORDER.record(
        "D10",
        "FPS 분포 확인",
        "D10 FPS distribution",
        "approximate",
        "joined existing metadata FPS by available video identifiers",
        inputs=[RAW_PATH, META_PATH],
        outputs=[csv_path, png_path],
        caveat="FPS values are read from existing metadata; missing values are not filled with an assumed 30 FPS.",
    )


run_diagnostic("D10", "FPS 분포 확인", "D10 FPS distribution", d10_fps_distribution)


# %% D11 Learning curves
def d11_learning_curves() -> None:
    if HISTORY_PATH is None or not HISTORY_PATH.exists():
        RECORDER.record("D11", "학습 곡선 패턴", "D11 Learning curves", "unavailable", "history.json not found", inputs=[RUN_DIR_RESOLVED])
        return
    history = read_json(HISTORY_PATH)
    max_len = max((len(v) for v in history.values() if isinstance(v, list)), default=0)
    if max_len == 0:
        RECORDER.record("D11", "학습 곡선 패턴", "D11 Learning curves", "unavailable", "history contains no list-like curves", inputs=[HISTORY_PATH])
        return
    rows = []
    for epoch in range(max_len):
        row = {"epoch": epoch + 1}
        for key, values in history.items():
            if isinstance(values, list) and epoch < len(values):
                row[key] = values[epoch]
        rows.append(row)
    hist_df = pd.DataFrame(rows)
    csv_path = write_csv(hist_df, OUTPUT_DIR / "D11_learning_curves.csv")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    metric_cols = [c for c in hist_df.columns if c != "epoch"]
    loss_cols = [c for c in metric_cols if "loss" in c.lower()]
    other_cols = [c for c in metric_cols if c not in loss_cols]
    for col in loss_cols:
        axes[0].plot(hist_df["epoch"], hist_df[col], label=col)
    axes[0].set_title("Loss curves")
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)
    for col in other_cols:
        axes[1].plot(hist_df["epoch"], hist_df[col], label=col)
    if "val_phase_f1" in hist_df.columns:
        best_idx = int(hist_df["val_phase_f1"].idxmax())
        axes[1].axvline(hist_df.loc[best_idx, "epoch"], color="red", linestyle="--", alpha=0.6, label="best val_phase_f1")
    axes[1].set_title("Validation metric curves")
    axes[1].set_xlabel("epoch")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)
    png_path = save_fig(fig, OUTPUT_DIR / "D11_learning_curves.png")
    RECORDER.record(
        "D11",
        "학습 곡선 패턴",
        "D11 Learning curves",
        "immediate",
        "plotted existing history.json only",
        inputs=[HISTORY_PATH],
        outputs=[csv_path, png_path],
    )


run_diagnostic("D11", "학습 곡선 패턴", "D11 Learning curves", d11_learning_curves)


# %% D12 Misprediction sample clustering
def d12_misprediction_clustering() -> None:
    ok, missing = required_columns(RAW_DF, ["gt_phase", RAW_COL, "type", "frame_idx"])
    prob_cols = [c for c in ["phase_prob_ready", "phase_prob_down", "phase_prob_up"] if c in RAW_DF.columns]
    if not ok or len(prob_cols) < 3:
        RECORDER.record(
            "D12",
            "오예측 샘플 클러스터링",
            "D12 Misprediction sample clustering",
            "unavailable",
            f"missing columns: {missing}; probability columns present: {prob_cols}",
            inputs=[RAW_PATH],
        )
        return
    errors = RAW_DF[RAW_DF["gt_phase"].astype(int) != RAW_DF[RAW_COL].astype(int)].copy()
    if len(errors) < 2:
        RECORDER.record("D12", "오예측 샘플 클러스터링", "D12 Misprediction sample clustering", "unavailable", "fewer than two raw prediction errors", inputs=[RAW_PATH])
        return

    probs = errors[prob_cols].astype(float).to_numpy()
    probs_safe = np.clip(probs, 1e-9, 1.0)
    errors["confidence"] = probs_safe.max(axis=1)
    sorted_probs = np.sort(probs_safe, axis=1)
    errors["confidence_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    errors["entropy"] = -(probs_safe * np.log(probs_safe)).sum(axis=1)
    if "boundary_distance" not in errors.columns:
        errors["boundary_distance"] = np.nan

    feature_cols = ["confidence", "confidence_margin", "entropy", "boundary_distance", *prob_cols]
    feature_df = errors[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(errors[feature_cols].median(numeric_only=True))
    status_reason = "clustered raw mispredictions using existing row-level probabilities and metadata"
    outputs: List[Path] = []

    if SKLEARN_CLUSTER_AVAILABLE and len(errors) >= 3:
        k = min(5, max(2, int(math.sqrt(len(errors) / 2))))
        x = StandardScaler().fit_transform(feature_df.to_numpy())  # type: ignore[union-attr]
        labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(x)  # type: ignore[union-attr]
        errors["cluster"] = labels
        pca = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(x)  # type: ignore[union-attr]
        errors["pca1"] = pca[:, 0]
        errors["pca2"] = pca[:, 1]
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(errors["pca1"], errors["pca2"], c=errors["cluster"], s=12, cmap="tab10", alpha=0.75)
        ax.set_title("D12 Misprediction clusters (PCA of existing features)")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        fig.colorbar(sc, ax=ax, label="cluster")
        outputs.append(save_fig(fig, OUTPUT_DIR / "D12_misprediction_clusters.png"))
    else:
        errors["cluster"] = (
            errors["type"].astype(str)
            + "|gt="
            + errors["gt_phase"].map(phase_name).astype(str)
            + "|pred="
            + errors[RAW_COL].map(phase_name).astype(str)
        )
        status_reason = "grouped raw mispredictions descriptively because sklearn clustering was unavailable or sample size was too small"

    cluster_path = write_csv(errors, OUTPUT_DIR / "D12_misprediction_clusters.csv")
    summary = (
        errors.groupby("cluster")
        .agg(
            n=("frame_idx", "size"),
            mean_confidence=("confidence", "mean"),
            mean_margin=("confidence_margin", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_boundary_distance=("boundary_distance", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    summary_path = write_csv(summary, OUTPUT_DIR / "D12_cluster_examples.csv")
    outputs = [cluster_path, summary_path, *outputs]
    RECORDER.record(
        "D12",
        "오예측 샘플 클러스터링",
        "D12 Misprediction sample clustering",
        "approximate",
        status_reason,
        inputs=[RAW_PATH],
        outputs=outputs,
        caveat=f"Uses existing columns only; no new embeddings/inference. Features: {', '.join(feature_cols)}.",
    )


run_diagnostic("D12", "오예측 샘플 클러스터링", "D12 Misprediction sample clustering", d12_misprediction_clustering)


# %% D13 Joint masking ablation
def d13_joint_masking_ablation() -> None:
    d13_dir = OUTPUT_DIR / "D13_joint_masking_ablation"
    existing_csv = d13_dir / "D13_joint_masking_ablation.csv"
    if existing_csv.exists():
        RECORDER.record(
            "D13",
            "관절 제거 실험",
            "D13 Joint masking ablation",
            "immediate",
            "found existing joint masking ablation output",
            inputs=[existing_csv],
            outputs=[existing_csv],
        )
        return

    d13_dir.mkdir(parents=True, exist_ok=True)
    d13_status = "unavailable" if RUN_JOINT_MASKING else "optional_skipped"
    d13_reason = (
        "RUN_JOINT_MASKING was requested, but this artifact-only notebook does not execute fresh inference; "
        "provide existing D13 outputs or implement a separate isolated ablation runner."
        if RUN_JOINT_MASKING
        else "Joint masking requires fresh checkpoint inference/experiment and is off by default."
    )
    skip_manifest = {
        "diagnostic_id": "D13",
        "status": d13_status,
        "reason": d13_reason,
        "run_joint_masking_flag": RUN_JOINT_MASKING,
        "fresh_inference_enabled": FRESH_INFERENCE_ENABLED,
        "created_at_utc": utc_now(),
        "expected_inputs_if_enabled": ["checkpoint", "pose npz files", "config", "labels"],
    }
    manifest_path = d13_dir / "manifest.json"
    write_json(manifest_path, skip_manifest)
    RECORDER.record(
        "D13",
        "관절 제거 실험",
        "D13 Joint masking ablation",
        d13_status,
        d13_reason,
        inputs=[RAW_PATH],
        outputs=[manifest_path],
        caveat="Cell is intentionally present but skipped by default. Outputs are isolated under D13_joint_masking_ablation/.",
    )


run_diagnostic("D13", "관절 제거 실험", "D13 Joint masking ablation", d13_joint_masking_ablation)


# %% Summary dashboard / manifest writing
def build_summary_dashboard() -> Path:
    lines = [
        "# Phase Error Analysis Summary",
        "",
        f"- created_at_utc: `{utc_now()}`",
        f"- run_dir: `{rel(RUN_DIR_RESOLVED)}`",
        f"- raw_predictions: `{rel(RAW_PATH)}`",
        "- default behavior: existing artifacts only; no training or fresh inference",
        "",
        "## Diagnostic statuses",
        "",
    ]
    status_df = pd.DataFrame(RECORDER.entries.values()).sort_values("diagnostic_id")
    if not status_df.empty:
        lines.append(markdown_table(status_df[["diagnostic_id", "name", "status", "reason"]]))
    else:
        lines.append("No diagnostics were recorded.")

    f1_path = OUTPUT_DIR / "D02_per_phase_f1.csv"
    if f1_path.exists():
        f1_df = read_csv(f1_path)
        raw = f1_df[f1_df["mode"] == "raw"]
        if not raw.empty:
            weakest = raw.sort_values("f1").iloc[0]
            lines.extend(["", "## Weakest raw phase", "", f"- `{weakest['phase']}` F1 = `{weakest['f1']:.4f}`"])

    cm_path = OUTPUT_DIR / "D03_confusion_matrix.csv"
    if cm_path.exists():
        cm_df = read_csv(cm_path)
        off_diag = cm_df[(cm_df["mode"] == "raw") & (cm_df["gt_phase"] != cm_df["pred_phase"])].sort_values("count", ascending=False)
        if not off_diag.empty:
            top = off_diag.iloc[0]
            lines.extend(["", "## Most common raw confusion", "", f"- GT `{top['gt_phase']}` → Pred `{top['pred_phase']}`: `{int(top['count'])}` rows"])

    d05_path = OUTPUT_DIR / "D05_transition_timing_error_summary.csv"
    if d05_path.exists():
        d05 = read_csv(d05_path)
        if not d05.empty:
            lines.extend(["", "## Transition lag summary", "", markdown_table(d05)])

    d07_path = OUTPUT_DIR / "D07_per_exercise_phase_f1.csv"
    if d07_path.exists():
        d07 = read_csv(d07_path)
        if not d07.empty:
            raw = d07[d07["mode"] == "raw"]
            if not raw.empty:
                macro = raw.groupby("type")["macro_f1_overall"].mean().sort_values()
                lines.extend(["", "## Worst exercise by raw macro F1", "", f"- `{macro.index[0]}` macro F1 = `{macro.iloc[0]:.4f}`"])

    lines.extend(["", "## Caveats", "", "- Offline smoothing is post-hoc/non-causal.", "- Transition, rep-order, up-segment, FPS, and clustering diagnostics may be approximate depending on available artifacts.", "- D13 joint masking is skipped by default because it requires fresh inference/experiment."])
    path = OUTPUT_DIR / "phase_error_analysis_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


STATUS_CSV_PATH = RECORDER.write_status_csv()
NOTES_PATH = RECORDER.write_narrative()
SUMMARY_PATH = build_summary_dashboard()

candidate_manifest_rows = []
for row in RUN_CANDIDATES[:20]:
    candidate_manifest_rows.append(
        {
            "run_dir": rel(Path(row["run_dir"])),
            "raw_predictions": rel(Path(row["raw_predictions"])),
            "manifest": rel(Path(row["manifest"])) if row.get("manifest") else None,
            "completion_score": row["completion_score"],
            "timestamp": row["timestamp"],
            "raw_mtime": row["raw_mtime"],
            "pattern": row["pattern"],
        }
    )

MANIFEST = {
    "created_at_utc": utc_now(),
    "script": SCRIPT_LABEL,
    "script_path": rel(SCRIPT_PATH) if SCRIPT_PATH else None,
    "project_root": str(PROJECT_ROOT),
    "selected_run_dir": rel(RUN_DIR_RESOLVED),
    "output_dir": rel(OUTPUT_DIR),
    "artifact_inventory": ARTIFACT_INVENTORY,
    "run_discovery": {
        "explicit_run_dir": str(RUN_DIR) if RUN_DIR else None,
        "search_patterns": SEARCH_PATTERNS,
        "selection_rule": "explicit RUN_DIR else complete manifest score, manifest timestamp, raw CSV mtime, path tie-breaker",
        "candidates": candidate_manifest_rows,
    },
    "fresh_inference_enabled": FRESH_INFERENCE_ENABLED,
    "run_joint_masking_requested": RUN_JOINT_MASKING,
    "smoothing_policy": {
        "raw_column": RAW_COL if RAW_COL in RAW_DF.columns else None,
        "offline_smooth_column": SMOOTH_COL if SMOOTH_COL in RAW_DF.columns else None,
        "offline_smooth_caveat": "post-hoc/non-causal; not a real-time deployment metric",
    },
    "coverage_caveat": "Raw prediction rows are valid model-output frames, not necessarily every original frame.",
    "diagnostics": [RECORDER.entries.get(f"D{i:02d}", {"diagnostic_id": f"D{i:02d}", "status": "unavailable", "reason": "not recorded"}) for i in range(1, 14)],
    "status_csv": rel(STATUS_CSV_PATH),
    "notes": rel(NOTES_PATH),
    "summary": rel(SUMMARY_PATH),
    "schema_validation": rel(SCHEMA_PATH),
    "coverage": rel(COVERAGE_PATH),
}
MANIFEST_PATH = OUTPUT_DIR / "phase_error_analysis_manifest.json"
write_json(MANIFEST_PATH, MANIFEST)

print(json.dumps({"manifest": rel(MANIFEST_PATH), "status_csv": rel(STATUS_CSV_PATH), "summary": rel(SUMMARY_PATH)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # All work is executed at cell level above so the file behaves like a
    # PyCharm/Jupyter script-backed notebook and a plain Python script.
    pass
