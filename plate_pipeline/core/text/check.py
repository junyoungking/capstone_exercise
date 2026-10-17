"""Checks for sentences: judgement words, and that a rewritten sentence keeps the numbers, units and
direction words of the template (구현 설계 tab, LLM 다듬기 2단계)."""
import re
from collections import Counter

from ..config import DEFAULT

NUM_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(초|cm|m/s|%|회|개|rep)")
DIRECTION = ("길었", "짧았", "더 높이", "덜", "더", "느렸")


def banned_words(text, banned=DEFAULT.text.banned):
    return [w for w in banned if w in text]


def numbers(text):
    return Counter((float(n), u) for n, u in NUM_UNIT.findall(text))


def directions(text):
    found = []
    rest = text
    for w in DIRECTION:            # longest first matters for "더 높이" vs "더"
        k = rest.count(w)
        found += [w] * k
        rest = rest.replace(w, " ")
    return Counter(found)


def same_facts(template, rewritten, banned=DEFAULT.text.banned):
    """Returns a list of problems; empty means the rewrite may be used."""
    probs = []
    if numbers(template) != numbers(rewritten):
        probs.append("numbers")
    if directions(template) != directions(rewritten):
        probs.append("direction")
    if banned_words(rewritten, banned):
        probs.append("banned")
    return probs


def is_polite(text):
    """Every sentence ends in the 합니다체 (…니다.)."""
    parts = [p.strip() for p in re.split(r"(?<=[.])\s+", text.strip()) if p.strip()]
    return bool(parts) and all(re.search(r"니다( \([^()]*\))?\.$", p) for p in parts)
