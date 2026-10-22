"""사용자 rep의 포지션 값을 선택한 ref rep과 비교한다."""

from __future__ import annotations

import math

from feedback import config


def _finite(value) -> bool:
    return value is not None and math.isfinite(value)


def compare_values(user, ref, threshold: float, unit: str) -> dict:
    if not _finite(user) or not _finite(ref):
        return {"user": None, "ref": None, "diff": None, "unit": unit,
                "threshold": None, "k": None, "dir": None, "color": "none"}
    diff = float(user - ref)
    k = abs(diff) / threshold if threshold > 0 else None
    color = "none"
    if k is not None and k > config.K_YELLOW_MAX:
        color = "red"
    elif k is not None and k > config.K_NONE_MAX:
        color = "yellow"
    return {"user": float(user), "ref": float(ref), "diff": diff, "unit": unit,
            "threshold": float(threshold), "k": k,
            "dir": "+" if diff > 0 else "-" if diff < 0 else "0", "color": color}


def _cells(user_rep: dict, ref_rep: dict, exercise: str) -> list[dict]:
    cells = []
    for position in config.POSITIONS:
        for metric in config.DISPLAY_METRICS[exercise]:
            values = compare_values(user_rep["positions"][position]["values"].get(metric),
                                    ref_rep["positions"][position]["values"].get(metric),
                                    config.DELTA_THRESHOLDS[exercise][metric],
                                    "ratio" if metric in config.RATIO_METRICS else "deg")
            available = values["diff"] is not None
            status = "unavailable" if not available else (
                "advisory" if metric in config.ADVISORY_METRICS.get(exercise, set()) else "ok")
            cells.append({"where": position, "metric": metric, "status": status, **values,
                          "range": [values["ref"] - values["threshold"],
                                    values["ref"] + values["threshold"]] if available else None})
    return cells


def _reference_available_ratio(rep: dict, exercise: str) -> float:
    values = [rep["positions"][position]["values"].get(metric)
              for position in config.POSITIONS for metric in config.DISPLAY_METRICS[exercise]]
    return sum(_finite(value) for value in values) / len(values)


def compare_reps(user: dict, ref: dict, ref_rep: int = 1) -> dict:
    exercise = user["exercise"]
    if exercise != ref["exercise"]:
        raise ValueError({"user": exercise, "ref": ref["exercise"]})
    matches = [rep for rep in ref["reps"] if rep["rep"] == ref_rep]
    if len(matches) != 1:
        raise ValueError({"ref_rep": ref_rep, "matches": len(matches)})
    reference = matches[0]
    if (reference["excluded"]
            or any(value < config.MIN_VALID_RATIO for value in reference["detect"].values())
            or _reference_available_ratio(reference, exercise) < config.MIN_REP_METRIC_RATIO):
        raise ValueError(config.REF_EXCLUDE_REASON)

    reps = []
    for user_rep in user["reps"]:
        cells = _cells(user_rep, reference, exercise)
        available_ratio = sum(cell["status"] != "unavailable" for cell in cells) / len(cells)
        reason = user_rep["exclude_reason"]
        excluded = user_rep["excluded"]
        if not excluded and available_ratio < config.MIN_REP_METRIC_RATIO:
            excluded = True
            reason = config.METRIC_REASON_TEMPLATE.format(ratio=available_ratio,
                                                        minimum=config.MIN_REP_METRIC_RATIO)
        tempo = [{"metric": metric,
                  **compare_values(user_rep["tempo"].get(metric), reference["tempo"].get(metric),
                                   threshold, "s")}
                 for metric, threshold in config.TEMPO_THRESHOLDS.items()]
        reps.append({"rep": user_rep["rep"], "excluded": excluded, "exclude_reason": reason,
                     "cells": cells, "tempo": tempo,
                     "tempo_ratio": {"user": user_rep["tempo"]["ratio"],
                                     "ref": reference["tempo"]["ratio"]},
                     "goal": None, "tempo_goal": None})
    return {"exercise": exercise, "ref_rep": ref_rep,
            "thresholds": dict(config.DELTA_THRESHOLDS[exercise]),
            "tempo_thresholds": dict(config.TEMPO_THRESHOLDS), "reps": reps}
