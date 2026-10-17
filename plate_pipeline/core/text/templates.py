"""Template sentences (구현 설계 tab, "텍스트 피드백 엔진").

Findings (result == "different") become one sentence each: vs-reference first, then in-set; inside a
kind the larger |delta| / (k * SE) first. max_shown sentences are shown, the rest folded. Then, per kind,
one "none" sentence if that kind judged something and found nothing, then the skip sentences. "none" and
"skip" sentences do not count towards max_shown. Times are written with time_digits decimals, distances
in whole cm, and the difference is the difference of the two written numbers.
No judgement words (config text.banned), polite 합니다체.
"""
from dataclasses import dataclass
from typing import List, Optional

from ..config import DEFAULT, Config

EX_NAME = {"squat": "스쿼트", "bench": "벤치프레스", "deadlift": "데드리프트"}
FAR = {"squat": "최저점", "bench": "가슴", "deadlift": "락아웃"}
KIND_NAME = {"inset": "세트 안 비교", "vs_ref": "기준 영상 비교"}


@dataclass
class Sentence:
    key: str
    text: str
    kind: str                  # "finding" | "none" | "skip" | "summary"
    shown: bool = True
    compare: Optional[object] = None


def value_name(value, ex):
    if value == "down":
        return "내리는 시간"
    if value == "far_pause":
        return f"{FAR[ex]}에서 멈춘 시간"
    if value == "far_dist":
        return "올라간 높이" if ex == "deadlift" else "내려간 깊이"
    raise KeyError(value)


def _has_final(word):
    c = word.strip()[-1]
    return 0xAC00 <= ord(c) <= 0xD7A3 and (ord(c) - 0xAC00) % 28 != 0


def join_names(names):
    """A과 B / A와 B by A's final consonant; three or more: A, B과 C."""
    if len(names) == 1:
        return names[0]
    head = ", ".join(names[:-2] + [names[-2]]) if len(names) > 2 else names[0]
    last_left = names[-2]
    return f"{head}{'과' if _has_final(last_left) else '와'} {names[-1]}"


def _t(x, dg):
    return f"{x:.{dg}f}"


def finding_sentence(c, ex, cfg: Config = DEFAULT):
    dg = cfg.text.time_digits
    g = cfg.cmp.set_group_n
    if c.kind == "vs_ref":
        a, b = round(c.a, dg), round(c.b, dg)
        d = abs(a - b)
        longer = a > b
        word = "길었" if longer else "짧았"
        name = value_name(c.value, ex)
        subj = "이" if _has_final(name) else "가"
        text = (f"{name}{subj} 기준 영상보다 {_t(d, dg)}초 {word}습니다 "
                f"(내 세트 {_t(a, dg)}초, 기준 {_t(b, dg)}초).")
        return Sentence(f"vs_ref.{c.value}.{'+' if longer else '-'}", text, "finding", compare=c)
    if c.value == "far_dist":
        a, b = round(c.a * 100), round(c.b * 100)
        d = abs(b - a)
        more = b > a
        if ex == "deadlift":
            text = (f"마지막 {g} rep은 처음 {g} rep보다 바가 {d} cm {'더 높이' if more else '덜'} 올라갔습니다 "
                    f"({a} cm → {b} cm).")
        else:
            text = (f"마지막 {g} rep은 처음 {g} rep보다 {d} cm {'더' if more else '덜'} 내려갔습니다 "
                    f"({a} cm → {b} cm).")
        return Sentence(f"inset.far_dist.{'+' if more else '-'}", text, "finding", compare=c)
    a, b = round(c.a, dg), round(c.b, dg)
    d = abs(b - a)
    longer = b > a
    text = (f"마지막 {g} rep의 내리는 시간이 처음 {g} rep보다 {_t(d, dg)}초 {'길었' if longer else '짧았'}습니다 "
            f"({_t(a, dg)}초 → {_t(b, dg)}초).")
    return Sentence(f"inset.down.{'+' if longer else '-'}", text, "finding", compare=c)


def none_sentence(kind, names, cfg: Config = DEFAULT):
    g = cfg.cmp.set_group_n
    joined = join_names(names)
    if kind == "vs_ref":
        return Sentence("none.vs_ref", f"기준 영상과 비교한 {joined}의 차이가 판정 기준을 넘지 않았습니다.", "none")
    return Sentence("none.inset", f"처음 {g} rep과 마지막 {g} rep 사이에 {joined}의 변화가 판정 기준을 넘지 않았습니다.",
                    "none")


def skip_sentence(c, ex, ref_ex=None):
    r = c.reason
    if r == "exercise_mismatch":
        return Sentence("skip.exercise_mismatch",
                        f"기준 영상의 종목({EX_NAME.get(ref_ex, ref_ex)})이 이 세트({EX_NAME[ex]})와 달라 비교하지 않았습니다.",
                        "skip", compare=c)
    if r == "no_ref_values":
        return Sentence("skip.no_ref_values", "기준 영상에서 원판을 찾지 못해 비교하지 않았습니다.", "skip", compare=c)
    name = value_name(c.value, ex)
    if r == "set_too_short":
        return Sentence("skip.set_too_short",
                        f"이 세트에서 {name} 값이 있는 rep이 {c.n_a}개라 {KIND_NAME[c.kind]}에서 이 값은 판정하지 않았습니다 "
                        f"({c.need}개부터 판정).", "skip", compare=c)
    if r == "ref_too_short":
        return Sentence("skip.ref_too_short",
                        f"기준 영상에서 {name} 값이 있는 rep이 {c.n_b}개라 이 값은 판정하지 않았습니다 ({c.need}개부터 판정).",
                        "skip", compare=c)
    if r == "no_scatter":
        side = "이 세트" if c.side == "mine" else "기준 영상"
        return Sentence("skip.no_scatter",
                        f"{side}에서 {name} 값이 있는 rep이 세 개 연달아 이어지지 않아 이 값은 판정하지 않았습니다.",
                        "skip", compare=c)
    raise ValueError(r)


def render(compares, exercise, ref_exercise=None, cfg: Config = DEFAULT) -> List[Sentence]:
    out = []
    finds = [c for c in compares if c.result == "different"]
    finds.sort(key=lambda c: (0 if c.kind == "vs_ref" else 1, -(c.ratio or 0)))
    for i, c in enumerate(finds):
        s = finding_sentence(c, exercise, cfg)
        s.shown = i < cfg.text.max_shown
        out.append(s)
    for kind in ("vs_ref", "inset"):
        same = [c for c in compares if c.kind == kind and c.result == "same"]
        if same and not any(c.kind == kind for c in finds):
            out.append(none_sentence(kind, [value_name(c.value, exercise) for c in same], cfg))
    for kind in ("vs_ref", "inset"):
        for c in compares:
            if c.kind == kind and c.result == "skip":
                out.append(skip_sentence(c, exercise, ref_exercise))
    return out


def summary_sentence(exercise, n_reps, speeds, loss_last_pct=None):
    """One measurement sentence at the top of the report. Empty values are left out."""
    parts = [f"{EX_NAME[exercise]} {n_reps}회를 기록했습니다."]
    v = [x for x in speeds if x is not None]
    if v:
        lo, hi = min(v), max(v)
        rng = f"{lo:.2f} m/s" if round(lo, 2) == round(hi, 2) else f"{lo:.2f}~{hi:.2f} m/s"
        parts.append(f"올리는 평균 속도는 {rng}였습니다.")
    if loss_last_pct is not None and n_reps > 1:
        parts.append(f"마지막 rep은 가장 빠른 rep보다 {loss_last_pct:.0f}% 느렸습니다.")
    return Sentence("summary", " ".join(parts), "summary")
