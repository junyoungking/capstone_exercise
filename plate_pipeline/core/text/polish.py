"""Optional LLM rewrite of the template sentences (off by default: config text.llm_polish = False).

polish(sentences, client) asks `client(prompt) -> str` (one rewritten sentence per line) to rewrite the
finding sentences without changing numbers, units, direction or rep numbers and without judgement words.
Each rewrite is checked against its template; any mismatch keeps the template sentence.
Returns (sentences, log) where log lists what happened per sentence.
"""
from dataclasses import replace

from ..config import DEFAULT, Config
from .check import same_facts

PROMPT = ("다음 운동 피드백 문장을 더 자연스럽게 다듬어 주세요. 숫자, 단위, 방향(길었/짧았, 더/덜), rep 번호를 바꾸지 말고, "
          "좋다·나쁘다·해야 한다 같은 판정어를 쓰지 말고, 문장 수를 늘리지 말고, 합니다체로 한 줄에 한 문장씩 돌려주세요.\n\n")


def polish(sentences, client=None, cfg: Config = DEFAULT):
    if not cfg.text.llm_polish or client is None:
        return sentences, ["template"] * len(sentences)
    finds = [s for s in sentences if s.kind == "finding"]
    if not finds:
        return sentences, ["template"] * len(sentences)
    try:
        reply = client(PROMPT + "\n".join(s.text for s in finds))
        lines = [l.strip() for l in str(reply).splitlines() if l.strip()]
    except Exception:
        return sentences, ["error"] * len(sentences)
    out, log = [], []
    it = iter(lines)
    for s in sentences:
        if s.kind != "finding":
            out.append(s); log.append("template")
            continue
        new = next(it, None)
        if new is None or same_facts(s.text, new, cfg.text.banned):
            out.append(s); log.append("rejected")
        else:
            out.append(replace(s, text=new)); log.append("polished")
    return out, log
