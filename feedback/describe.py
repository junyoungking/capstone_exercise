"""설계 5.5의 고정 템플릿으로 선택 결과를 서술한다."""

from __future__ import annotations

from feedback import config


def _amount(cell: dict) -> int:
    scale = config.RATIO_PERCENT_SCALE if cell["unit"] == "ratio" else 1
    return round(abs(cell["diff"]) * scale)


def _tempo_sentence(rep: dict, exercise: str = "squat") -> str | None:
    goal = rep["tempo_goal"]
    if goal is None:
        return None
    metric = goal["metric"]
    if metric == "bottom_s":
        template = config.TEMPO_BOTTOM_TEMPLATE.replace("바닥", config.phase_labels(exercise)["bottom"])
        return template.format(
            n=abs(goal["diff"]), direction=config.TEMPO_BOTTOM_DIRECTIONS[goal["dir"]])
    phase = metric.removesuffix("_s")
    return (config.TEMPO_MOVEMENT_TEMPLATE.replace("하강:상승", "상승:하강")
            if exercise == "deadlift" else config.TEMPO_MOVEMENT_TEMPLATE).format(
        phase=config.phase_labels(exercise)[phase], n=abs(goal["diff"]),
        direction=config.TEMPO_MOVEMENT_DIRECTIONS[goal["dir"]],
        user_ratio=rep["tempo_ratio"]["user"], ref_ratio=rep["tempo_ratio"]["ref"])


def describe_reps(findings: dict) -> list[dict]:
    described = []
    for rep in findings["reps"]:
        if rep["excluded"]:
            label = config.TAB_EXCLUDED_LABEL
            sentence = config.EXCLUDE_TEMPLATE.format(reason=rep["exclude_reason"])
            tempo_sentence = None
        else:
            goal = rep["goal"]
            cells = {(cell["where"], cell["metric"]): cell for cell in rep["cells"]}
            if goal["type"] == "none":
                label, sentence = config.TAB_NONE_LABEL, config.NONE_SENTENCE
            elif goal["type"] == "group":
                knee = _amount(cells[goal["where"], "knee_angle"])
                hip = _amount(cells[goal["where"], "hip_angle"])
                sentence = config.DEPTH_TEMPLATES[goal["dir"]].format(knee=knee, hip=hip)
                label = config.TAB_DEPTH_LABELS[goal["dir"]]
            else:
                metric = goal["metric"]
                cell = cells[goal["where"], metric]
                amount = _amount(cell)
                where = config.position_labels(findings["exercise"])[goal["where"]]
                phrase = config.METRIC_TEMPLATES[metric][goal["dir"]].format(n=amount)
                sentence = config.SINGLE_TEMPLATE.format(where=where, phrase=phrase)
                label = config.TAB_SINGLE_TEMPLATE.format(
                    where=where, metric=config.METRIC_LABELS_KO[metric],
                    sign="+" if goal["dir"] == "+" else "−", n=amount,
                    unit="%" if cell["unit"] == "ratio" else "°")
            tempo_sentence = _tempo_sentence(rep, findings["exercise"])
        described.append({"rep": rep["rep"], "tab_label": label,
                          "goal_sentence": sentence, "tempo_sentence": tempo_sentence})
    return described
