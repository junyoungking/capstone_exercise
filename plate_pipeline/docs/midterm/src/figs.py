"""Slide figures: tempo alignment and rep-level vs set-level judgement (real data from bundle3)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
for w in ("Regular", "Bold"):
    fm.fontManager.addfont(f"/usr/share/fonts/opentype/noto/NotoSansCJK-{w}.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
B = json.load(open('../newvid/bundle3.json'))
NAVY, TEAL, ORANGE, GRAY = "#1F2A5C", "#17A08F", "#E2862E", "#9AA3AE"
U, R = B['user'], B['ref']


def style(ax):
    ax.grid(color="#E6E8EC"); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#C9CED6")
    ax.tick_params(colors="#555", labelsize=12)


# ---------------- alignment
def rep_curve(V, r):
    t = np.array(V['t']); d = np.array(V['depth']); m = (t >= r['start']) & (t <= r['back'])
    return t[m] - r['start'], d[m]


def prog(V, r):
    t = np.array(V['t']); d = np.array(V['depth'])
    i0, i1 = np.searchsorted(t, r['start']), np.searchsorted(t, r['back'])
    fi = np.searchsorted(t, r['far_in']); dmax = d[i0:i1 + 1].max()
    ts, ps = [], []; run = 0
    for i in range(i0, fi + 1):
        run = max(run, d[i] / dmax); ts.append(t[i]); ps.append(.49 * min(1, run))
    ts += [r['far_in'], max(r['far_out'], r['far_in'] + 1e-3)]; ps += [.49, .51]; run = 0
    for i in range(np.searchsorted(t, r['far_out']) + 1, i1 + 1):
        run = max(run, 1 - d[i] / dmax); ts.append(t[i]); ps.append(.51 + .49 * min(1, run))
    ts.append(r['back']); ps.append(1)
    o = np.argsort(ts); return np.array(ts)[o], np.maximum.accumulate(np.array(ps)[o])


def t_at(ts, ps, p):
    i = min(max(np.searchsorted(ps, p), 1), len(ps) - 1)
    f = (p - ps[i - 1]) / max(1e-9, ps[i] - ps[i - 1]); return ts[i - 1] + f * (ts[i] - ts[i - 1])


dep = lambda V, tt: np.interp(tt, V['t'], V['depth'])
ur, rr = U['reps'][0], R['reps'][0]
ut, ud = rep_curve(U, ur); rt, rd = rep_curve(R, rr)
uts, ups = prog(U, ur); rts, rps = prog(R, rr)
fig, axs = plt.subplots(1, 2, figsize=(16, 7.2), sharey=True)
for k, ax in enumerate(axs):
    ax.plot(ut, ud, color=TEAL, lw=3.5, label=f"내 rep 1 (무거운 세트, {ur['back'] - ur['start']:.1f}초)")
    ax.plot(rt, rd, '--', color=NAVY, lw=3, label=f"기준 rep 1 ({rr['back'] - rr['start']:.1f}초)")
    if k == 0:
        for s in np.arange(.4, rt[-1], .5):
            ax.plot([s, s], [dep(U, ur['start'] + s), dep(R, rr['start'] + s)], color=ORANGE, lw=1.8, alpha=.9)
        ax.set_title("시각으로 맞추면: 같은 시각, 다른 동작", loc="left", fontsize=17, color=NAVY, fontweight="bold", pad=12)
    else:
        for p in [.06, .13, .2, .27, .34, .41, .49, .59, .67, .75, .83, .91, .97]:
            tu = t_at(uts, ups, p); tr = t_at(rts, rps, p)
            ax.plot([tu - ur['start'], tr - rr['start']], [dep(U, tu), dep(R, tr)], color=ORANGE, lw=1.6, alpha=.85)
        ax.set_title("바 높이 비율로 맞추면: 같은 동작끼리 (우리 방식)", loc="left", fontsize=17, color=NAVY, fontweight="bold", pad=12)
    ax.set_xlabel("rep 시작부터 시간 (초)", fontsize=13, color="#444"); style(ax)
axs[0].invert_yaxis()
axs[0].set_ylabel("바가 내려간 깊이 (cm, 아래가 깊음)", fontsize=13, color="#444")
h, l = axs[0].get_legend_handles_labels(); h.append(Line2D([0], [0], color=ORANGE, lw=2)); l.append("비교하는 프레임 짝")
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(.5, .905))
fig.suptitle("템포가 다른 두 영상 정렬: 원판 높이로 rep 진행도를 만든다", x=.06, y=.985, ha="left", fontsize=21, color=NAVY, fontweight="bold")
fig.text(.06, .02, "rep 이벤트(시작·바닥 도착·바닥 출발·락아웃)는 원판 상태기계가 찾는다. 내려가는 구간은 최대 깊이의 몇 %까지 왔는지, 올라오는 구간은 몇 %를 회복했는지가 진행도다. 템포 차이는 따로 잰다.", fontsize=11.5, color="#666")
plt.tight_layout(rect=(.03, .05, 1, .86)); plt.savefig("alignment.png", dpi=200)

# ---------------- rep level vs set level
ix = {n: i for i, n in enumerate(U['j3names'])}


def ang(a, b, c):
    u = a - b; v = c - b; return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


def knees(V):
    t = np.array(V['t']); out = []
    for r in V['reps']:
        i = int(np.argmin(abs(t - r['far'])))
        qs = [np.array(V['j3'][k]) for k in range(i - 2, i + 3) if V['j3'][k]]
        out.append(np.mean([(ang(q[ix[23]], q[ix[25]], q[ix[27]]) + ang(q[ix[24]], q[ix[26]], q[ix[28]])) / 2 for q in qs]))
    return np.array(out)


ku, kr = knees(U), knees(R)
fig, axs = plt.subplots(1, 2, figsize=(16, 7.2))
ax = axs[0]
for xs, ks, col, lab in [(np.arange(1, len(ku) + 1), ku, TEAL, "내 세트"), (np.arange(1, len(kr) + 1) + .15, kr, NAVY, "기준 영상")]:
    ax.plot(xs, ks, 'o', color=col, ms=11, label=lab)
    m, se = ks.mean(), ks.std(ddof=1) / np.sqrt(len(ks))
    ax.axhspan(m - 3 * se, m + 3 * se, color=col, alpha=.12)
    ax.axhline(m, color=col, lw=2)
    ax.text(6.55, m, f"평균 {m:.0f}°\n±3·SE {3 * se:.0f}°", color=col, va="center", fontsize=12, fontweight="bold")
ax.set_xlim(.5, 7.6); ax.set_xticks(range(1, 7)); ax.set_xticklabels([f"rep {i}" for i in range(1, 7)])
ax.set_ylabel("바닥에서 무릎각 (°, 3D 관절)", fontsize=13, color="#444"); style(ax)
ax.set_title(f"포즈 값: rep마다 ±{np.std(ku, ddof=1):.0f}° 흔들림 → 세트 평균으로 판정", loc="left", fontsize=16, color=NAVY, fontweight="bold", pad=12)
ax.legend(frameon=False, fontsize=13, loc="lower left")
ax = axs[1]
ru = np.array([r['rom_cm'] for r in U['reps']]); vu = np.array([r['v'] for r in U['reps']])
med = np.median(ru)
cols = [ORANGE if abs(x - med) > 3 else TEAL for x in ru]
ax.bar(range(1, len(ru) + 1), ru, color=cols, width=.6)
for i, x in enumerate(ru):
    ax.text(i + 1, x + .6, f"{x:.1f}", ha="center", fontsize=12, color="#333")
ax.axhline(med, color=GRAY, lw=1.5, ls="--"); ax.text(6.45, med + .4, f"중앙값 {med:.1f} cm", color="#666", fontsize=12)
ax.set_ylim(40, 64); ax.set_xticks(range(1, 7)); ax.set_xticklabels([f"rep {i}" for i in range(1, 7)])
ax.set_ylabel("바 이동 깊이 (cm, 원판)", fontsize=13, color="#444"); style(ax)
ok = np.abs(ru - med) <= 3
ax.set_title(f"원판 값: 나머지 rep 차이 ±{np.std(ru[ok], ddof=1):.1f} cm → rep 하나도 집어낼 수 있음", loc="left", fontsize=16, color=NAVY, fontweight="bold", pad=12)
ax.annotate("rep 2: −8.9 cm,\n가장 느림 0.18 m/s", (2, ru[1]), xytext=(2.6, 44), fontsize=12, color=ORANGE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ORANGE))
fig.suptitle("rep 단위 vs 세트 단위: 값의 잡음에 따라 나눈다", x=.06, y=.985, ha="left", fontsize=21, color=NAVY, fontweight="bold")
fig.text(.06, .02, "자세 문장(포즈)은 세트 단위로 판정하고 (|차이| > 3·표준오차, 최소 8°), 원판 값(깊이·템포·속도)은 rep마다 보여 주며 세트 안에서 튀는 rep을 표시한다.", fontsize=11.5, color="#666")
plt.tight_layout(rect=(.03, .05, 1, .9)); plt.savefig("rep_vs_set.png", dpi=200)
print("ok", ku.round(), kr.round())
