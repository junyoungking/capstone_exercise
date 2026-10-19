"""Set report as one static HTML file (first version of web/report; no external files).

Per set: the sentences, the bar height over time with each rep shaded, a per-rep table, tempo bars
(lowering / pause at the far point / raising) and mean raise speed per rep.
"""
import html
import json

CSS = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--mute:#6b6b66;--line:#e2e1dc;--card:#fff;--a:#2f6fb3;--b:#c9822b;--c:#5a9a5a;--rep:rgba(47,111,179,.08)}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mute:#9a9992;--line:#33332f;--card:#1f1f1d;--a:#6ea3dc;--b:#e0a457;--c:#86bf86;--rep:rgba(110,163,220,.12)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,"Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 10px}
.meta{color:var(--mute);font-size:13px}.warn{border-left:3px solid var(--b);padding:6px 10px;margin:12px 0;font-size:13px;color:var(--mute)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.sent p{margin:6px 0}.sent .skip{color:var(--mute);font-size:13px}.sent .fold{display:none}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--mute);font-weight:500}
svg text{fill:var(--mute);font-size:11px}.legend span{display:inline-block;margin-right:14px;font-size:12px;color:var(--mute)}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.wrap{overflow-x:auto}
"""


def _f(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


def _height_svg(sig, reps, W=820, H=230, pad=40):
    t, y = sig["t"], sig["y"]
    if not t:
        return ""
    t0, t1 = t[0], t[-1]
    lo, hi = min(y), max(y)
    rng = max(hi - lo, 1e-6)
    X = lambda v: pad + (v - t0) / max(t1 - t0, 1e-6) * (W - pad - 8)
    Y = lambda v: 22 + (hi - v) / rng * (H - 22 - 24)
    step = max(1, len(t) // 1500)
    pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(t[::step], y[::step]))
    shade = "".join(f'<rect x="{X(r["start"]):.1f}" y="18" width="{max(1, X(r["back"]) - X(r["start"])):.1f}" '
                    f'height="{H - 42}" fill="var(--rep)"/>'
                    f'<text x="{(X(r["start"]) + X(r["back"])) / 2:.1f}" y="13" text-anchor="middle">rep {r["n"]}</text>'
                    for r in reps if r["start"] is not None and r["back"] is not None)
    ticks = ""
    s = 0
    span = t1 - t0
    stp = 1 if span <= 15 else 2 if span <= 30 else 5 if span <= 90 else 10
    v = (int(t0 // stp) + 1) * stp
    while v < t1:
        ticks += f'<text x="{X(v):.1f}" y="{H - 6}" text-anchor="middle">{v:g}s</text>'
        v += stp
    yl = (f'<text x="{pad - 6}" y="{Y(hi) + 4:.1f}" text-anchor="end">{hi * 100:.0f}</text>'
          f'<text x="{pad - 6}" y="{Y(lo) + 4:.1f}" text-anchor="end">{lo * 100:.0f}</text>'
          f'<text x="{pad - 6}" y="{Y(0) + 4:.1f}" text-anchor="end">0</text>'
          f'<line x1="{pad}" x2="{W - 8}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--line)"/>')
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="바 높이">{shade}{yl}'
            f'<polyline points="{pts}" fill="none" stroke="var(--a)" stroke-width="1.6"/>{ticks}</svg>')


def _tempo_svg(reps, W=820, row=26):
    ok = [r for r in reps if r["down"] is not None and r["up"] is not None]
    if not ok:
        return ""
    tot = max((r["down"] or 0) + (r["far_pause"] or 0) + (r["up"] or 0) for r in ok)
    H = row * len(ok) + 6
    sc = (W - 40 - 150) / max(tot, 1e-6)
    out = []
    for i, r in enumerate(ok):
        y0 = 4 + i * row
        x = 40
        for key, col in (("down", "var(--a)"), ("far_pause", "var(--b)"), ("up", "var(--c)")):
            w = (r[key] or 0) * sc
            out.append(f'<rect x="{x:.1f}" y="{y0}" width="{w:.1f}" height="{row - 8}" rx="2" fill="{col}"/>')
            x += w
        out.append(f'<text x="4" y="{y0 + row - 12}">rep {r["n"]}</text>')
        out.append(f'<text x="{x + 6:.1f}" y="{y0 + row - 12}">{r["down"]:.2f} / {r["far_pause"] or 0:.2f} / {r["up"]:.2f} s</text>')
    return f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="템포">{"".join(out)}</svg>'


def _speed_svg(reps, W=820, H=140):
    v = [(r["n"], r["v_mean"]) for r in reps if r["v_mean"] is not None]
    if not v:
        return ""
    vmax = max(b for _, b in v) * 1.15
    bw = min(60, (W - 40) / len(v) * 0.6)
    gap = (W - 40) / len(v)
    out = []
    for i, (n, b) in enumerate(v):
        h = b / vmax * (H - 36)
        x = 30 + i * gap + (gap - bw) / 2
        out.append(f'<rect x="{x:.1f}" y="{H - 20 - h:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3" fill="var(--a)"/>')
        out.append(f'<text x="{x + bw / 2:.1f}" y="{H - 24 - h:.1f}" text-anchor="middle">{b:.2f}</text>')
        out.append(f'<text x="{x + bw / 2:.1f}" y="{H - 6}" text-anchor="middle">{n}</text>')
    return f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="평균 속도">{"".join(out)}</svg>'


def render_report(res, sig):
    ex = {"squat": "스쿼트", "bench": "벤치프레스", "deadlift": "데드리프트"}[res["exercise"]]
    far = {"squat": "최저점", "bench": "가슴", "deadlift": "락아웃"}[res["exercise"]]
    dist = "올라간 높이" if res["exercise"] == "deadlift" else "내려간 깊이"
    parts = [f"<h1>{ex} 세트 리포트</h1>",
             f'<div class="meta">{html.escape(res["video"])} · {res["width"]}×{res["height"]} · {res["fps"]:.1f} fps · '
             f'원판 {res["plate_mm"]:.0f} mm · 원판이 잡힌 프레임 {res["plate_frames"]}/{res["frames"]}</div>']
    if res["plate_source"] != "model":
        parts.append(f'<div class="warn">원판 위치: {html.escape(res["plate_source"])}. 학습한 검출기 없이 만든 결과라 '
                     f"절대값(cm, m/s)은 검증되지 않았습니다.</div>")
    if not res["sets"]:
        parts.append('<div class="card">세트를 찾지 못했습니다.</div>')
    for s in res["sets"]:
        reps = s["rep_values"]
        sents = "".join(f'<p class="{"skip" if x["kind"] in ("skip", "none") else ""}{"" if x["shown"] else " fold"}">'
                        f'{html.escape(x["text"])}</p>' for x in s["sentences"])
        rows = "".join(f'<tr><td>{r["n"]}</td><td>{_f(r["far_dist_m"] and r["far_dist_m"] * 100, 0)}</td>'
                       f'<td>{_f(r["down"])}</td><td>{_f(r["far_pause"])}</td><td>{_f(r["up"])}</td>'
                       f'<td>{_f(r["v_mean"])}</td></tr>' for r in reps)
        set_sig = sig
        parts.append(
            f'<h2>세트 {s["set"]} · {s["reps"]}회</h2><div class="card sent">{sents}</div>'
            f'<div class="card"><div class="legend"><span>바 높이 (cm, 시작 위치 0)</span>'
            f'<span><i class="sw" style="background:var(--rep)"></i>rep</span></div>{_height_svg(set_sig, reps)}</div>'
            f'<div class="card wrap"><table><tr><th>rep</th><th>{dist} cm</th><th>내리는 시간 s</th>'
            f'<th>{far}에서 멈춤 s</th><th>올리는 시간 s</th><th>평균 속도 m/s</th></tr>{rows}</table></div>'
            f'<div class="card"><div class="legend"><span><i class="sw" style="background:var(--a)"></i>내림</span>'
            f'<span><i class="sw" style="background:var(--b)"></i>{far}에서 멈춤</span>'
            f'<span><i class="sw" style="background:var(--c)"></i>올림</span></div>{_tempo_svg(reps)}</div>'
            f'<div class="card"><div class="legend"><span>rep별 올리는 평균 속도 (m/s)</span></div>{_speed_svg(reps)}</div>')
    body = "".join(parts)
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>세트 리포트</title><style>{CSS}</style></head>'
            f"<body><main>{body}</main></body></html>")


def write_report(path, res, sig):
    open(path, "w", encoding="utf-8").write(render_report(res, sig))
