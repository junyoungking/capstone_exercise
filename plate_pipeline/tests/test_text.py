from dataclasses import replace

import numpy as np

from core.config import DEFAULT
from core.compare.judge import Compare, run_compare
from core.compare.summary import from_values
from core.text.templates import render, join_names, summary_sentence, finding_sentence
from core.text.check import banned_words, same_facts, is_polite
from core.text.polish import polish


def _ref(value, a, b, ratio=2.0):
    return Compare("vs_ref", value, "different", a=a, b=b, delta=a - b, se=abs(a - b) / (2 * ratio), k=2.0)


def _inset(value, a, b, ratio=2.0):
    return Compare("inset", value, "different", a=a, b=b, delta=b - a, se=abs(b - a) / (2 * ratio), k=2.0)


def test_vs_ref_sentence_uses_printed_numbers():
    s = finding_sentence(_ref("down", 2.104, 1.596), "squat")
    assert s.text == "내리는 시간이 기준 영상보다 0.50초 길었습니다 (내 세트 2.10초, 기준 1.60초)."
    s = finding_sentence(_ref("far_pause", 0.2, 0.9), "deadlift")
    assert s.text == "락아웃에서 멈춘 시간이 기준 영상보다 0.70초 짧았습니다 (내 세트 0.20초, 기준 0.90초)."


def test_inset_sentences():
    assert finding_sentence(_inset("far_dist", 0.524, 0.486), "squat").text == \
        "마지막 2 rep은 처음 2 rep보다 3 cm 덜 내려갔습니다 (52 cm → 49 cm)."
    assert finding_sentence(_inset("far_dist", 0.50, 0.55), "deadlift").text == \
        "마지막 2 rep은 처음 2 rep보다 바가 5 cm 더 높이 올라갔습니다 (50 cm → 55 cm)."
    assert finding_sentence(_inset("down", 2.5, 1.5), "bench").text == \
        "마지막 2 rep의 내리는 시간이 처음 2 rep보다 1.00초 짧았습니다 (2.50초 → 1.50초)."


def test_join_names():
    assert join_names(["내리는 시간", "가슴에서 멈춘 시간"]) == "내리는 시간과 가슴에서 멈춘 시간"
    assert join_names(["내려간 깊이", "내리는 시간"]) == "내려간 깊이와 내리는 시간"
    assert join_names(["a"]) == "a"


def test_order_fold_none_skip():
    cs = [_inset("down", 2.5, 1.5, ratio=9), _ref("down", 2.6, 1.6, ratio=1.5), _ref("far_pause", 0.9, 0.1, ratio=3),
          _inset("far_dist", 0.50, 0.45, ratio=2),
          Compare("inset", "far_dist", "skip", reason="set_too_short", n_a=3, need=4)]
    out = render(cs, "squat")
    keys = [s.key for s in out]
    assert keys[:4] == ["vs_ref.far_pause.+", "vs_ref.down.+", "inset.down.-", "inset.far_dist.-"]
    assert [s.shown for s in out[:4]] == [True, True, True, False]
    assert keys[4] == "skip.set_too_short"
    assert "세트 안 비교에서" in out[4].text


def test_none_sentence_per_kind():
    cs = [Compare("inset", "far_dist", "same"), Compare("inset", "down", "same"), _ref("down", 2.6, 1.6)]
    out = render(cs, "bench")
    assert [s.key for s in out] == ["vs_ref.down.+", "none.inset"]
    assert out[1].text == "처음 2 rep과 마지막 2 rep 사이에 내려간 깊이와 내리는 시간의 변화가 판정 기준을 넘지 않았습니다."


def test_skip_sentences():
    out = render([Compare("vs_ref", "*", "skip", reason="exercise_mismatch")], "squat", ref_exercise="bench")
    assert out[0].text == "기준 영상의 종목(벤치프레스)이 이 세트(스쿼트)와 달라 비교하지 않았습니다."
    out = render([Compare("vs_ref", "down", "skip", reason="no_scatter", side="ref")], "squat")
    assert out[0].text.startswith("기준 영상에서 내리는 시간 값이 있는 rep이 세 개 연달아")


def test_summary_sentence():
    s = summary_sentence("deadlift", 5, [0.61, 0.55, 0.48, None, 0.42], 31.2)
    assert s.text == "데드리프트 5회를 기록했습니다. 올리는 평균 속도는 0.42~0.61 m/s였습니다. 마지막 rep은 가장 빠른 rep보다 31% 느렸습니다."
    assert is_polite(s.text) and not banned_words(s.text)


def test_all_sentences_neutral_and_polite():
    """Random sets through compare + render: no judgement words, every sentence 합니다체."""
    rng = np.random.default_rng(3)
    seen = set()
    for i in range(1500):
        ex = ["squat", "bench", "deadlift"][i % 3]
        n, nr = int(rng.integers(1, 10)), int(rng.integers(0, 8))

        def vals(m, shift=0.0):
            out = list(rng.normal(1.8 + shift, rng.uniform(0.02, 0.4), m))
            return [None if rng.random() < 0.1 else v for v in out]
        mine = from_values(ex, 30, down=vals(n, rng.choice([0, 0.8])), far_pause=vals(n, -1.5),
                           far_dist=[None if v is None else v / 4 for v in vals(n)])
        rex = ex if rng.random() < 0.9 else "squat"
        ref = from_values(rex, 30, down=vals(nr), far_pause=vals(nr, -1.5), far_dist=vals(nr))
        out = render(run_compare(mine, ref), ex, rex)
        out.append(summary_sentence(ex, n, list(rng.uniform(0.3, 1.0, n)), float(rng.uniform(0, 40))))
        for s in out:
            seen.add(s.key)
            assert not banned_words(s.text), s.text
            assert is_polite(s.text), s.text
    assert {"vs_ref.down.+", "vs_ref.down.-", "inset.down.+", "none.inset", "skip.set_too_short",
            "skip.ref_too_short", "skip.no_scatter", "skip.exercise_mismatch"} <= seen


def test_same_facts():
    t = "내리는 시간이 기준 영상보다 0.50초 길었습니다 (내 세트 2.10초, 기준 1.60초)."
    assert same_facts(t, "기준 영상과 견주면 내리는 데 0.5초 더 걸려 길었습니다 (내 세트 2.10초, 기준 1.60초).") == ["direction"]
    assert same_facts(t, "내리는 시간이 기준 영상보다 0.50초 길었습니다. 내 세트는 2.10초, 기준은 1.60초였습니다.") == []
    assert same_facts(t, "내리는 시간이 기준 영상보다 0.60초 길었습니다 (내 세트 2.10초, 기준 1.60초).") == ["numbers"]
    assert same_facts(t, "내리는 시간이 기준 영상보다 0.50초 짧았습니다 (내 세트 2.10초, 기준 1.60초).") == ["direction"]
    assert "banned" in same_facts(t, t[:-1] + " 좋습니다.")


def test_polish_off_by_default_and_fallback():
    out = render([_ref("down", 2.6, 1.6)], "squat")
    same, log = polish(out, client=lambda p: "아무 문장")
    assert same == out and log == ["template"]
    cfg = replace(DEFAULT, text=replace(DEFAULT.text, llm_polish=True))
    new, log = polish(out, client=lambda p: "내리는 시간이 1.00초 짧았습니다.", cfg=cfg)
    assert log == ["rejected"] and new[0].text == out[0].text
    good = "기준 영상보다 내리는 시간이 1.00초 길었습니다 (내 세트 2.60초, 기준 1.60초)."
    new, log = polish(out, client=lambda p: good, cfg=cfg)
    assert log == ["polished"] and new[0].text == good

    def boom(p):
        raise RuntimeError
    new, log = polish(out, client=boom, cfg=cfg)
    assert log == ["error"] and new[0].text == out[0].text
