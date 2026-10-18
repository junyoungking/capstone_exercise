"""Break-it scenarios (근거·출처 tab, 깨뜨리기 시험: 41 situations + 2 run separately).

Each scenario returns (exercises, factory) where factory(ex, rng) -> Opts for one set, plus run flags.
Doc numbers (iteration 16, 30 fps, noise 1 cm, share of sets with the right count, final / live) are kept
next to each scenario in DOC so the runner can print the comparison.
"""
import numpy as np

from sim.gen import GEN, Opts, TrueRep, profile

G = 9.81
TOP = ("squat", "bench")
ALL = ("squat", "bench", "deadlift")


def _u(rng, ab):
    return float(rng.uniform(*ab))


def std_rep(tl, ex, rest_h, rom, d_dur, u_dur, shape, sway, fp_s, x_end=None, lower_to=None):
    """One normal rep from the rest height. Returns TrueRep."""
    x0 = tl.cx
    x_end = x0 if x_end is None else x_end
    if ex == "deadlift":
        r0 = tl.now
        tl.move(u_dur, y1=rest_h + rom, shape=shape, sway=sway)
        r1 = fin = tl.now
        tl.hold(fp_s)
        fout = tl.now
        tl.move(d_dur, y1=rest_h if lower_to is None else lower_to, x1=x_end, shape="cos")
    else:
        tl.move(d_dur, y1=rest_h - rom, shape="cos", sway=sway)
        fin = tl.now
        tl.hold(fp_s)
        fout = r0 = tl.now
        tl.move(u_dur, y1=rest_h, x1=x_end, shape=shape)
        r1 = tl.now
    return TrueRep(fin, fout, r0, r1, rom, rom / (r1 - r0))


# ---- hooks ---------------------------------------------------------------------------------

def far_pause(lo, hi):
    return lambda ex, rng: Opts(far_pause=(lo, hi))


def slow_down(lo, hi):
    return lambda ex, rng: Opts(down=(lo, hi))


def dead_touch_and_go(ex, rng):
    def after(tl, rng, k, n, rest_h):
        b = _u(rng, (0.03, 0.06))
        tf = np.sqrt(2 * b / G)
        tl.move(tf, y1=rest_h + b, shape="cos")
        tl.move(tf, y1=rest_h, shape="cos")
        if k == n - 1:
            tl.hold(_u(rng, (0.5, 2.0)))
    return Opts(hooks={"after_rep": after})


def regrip(depth=None, rom_range=None):
    """Bench: between reps the bar is lowered 8-12 cm and brought back (re-grip)."""
    def f(ex, rng):
        def after(tl, rng, k, n, rest_h):
            if k == n - 1:
                return
            dd = depth if depth is not None else _u(rng, (0.08, 0.12))
            tl.hold(_u(rng, (0.3, 0.8)))
            tl.move(_u(rng, (0.4, 0.8)), y1=rest_h - dd)
            tl.hold(_u(rng, (0.3, 1.0)))
            tl.move(_u(rng, (0.4, 0.8)), y1=rest_h)
            tl.hold(_u(rng, (0.3, 0.8)))
        return Opts(hooks={"after_rep": after}, rom_range=rom_range)
    return f


def dip_pause(pause, first_only):
    """Descent stops 8-12 cm down for `pause` s, then continues (first rep only or every rep)."""
    def f(ex, rng):
        def rep(tl, rng, k, n, rest_h, rom, d_dur, u_dur, shape, sway, fp):
            if first_only and k > 0:
                return std_rep(tl, ex, rest_h, rom, d_dur, u_dur, shape, sway, _u(rng, fp))
            dd = _u(rng, (0.08, 0.12))
            x0 = tl.cx
            tl.move(d_dur * dd / rom + 0.2, y1=rest_h - dd, shape="cos")
            tl.hold(_u(rng, pause))
            tl.move(d_dur, y1=rest_h - rom, shape="cos", sway=sway)
            fin = tl.now
            tl.hold(_u(rng, fp))
            fout = r0 = tl.now
            tl.move(u_dur, y1=rest_h, x1=x0, shape=shape)
            return TrueRep(fin, fout, r0, tl.now, rom, rom / (tl.now - r0))
        return Opts(hooks={"rep": rep})
    return f


def walk_bob(ex, rng):
    def walkout(tl, rng, rest_h, walk):
        tl.move(_u(rng, GEN["walk_s"][ex]), x1=walk, bob=_u(rng, (0.03, 0.06)), rng=rng)
    return Opts(hooks={"walkout": walkout})


def one_rep(ex, rng):
    return Opts(reps=1)


def occlude(lo, hi):
    return lambda ex, rng: Opts(occlude=(lo, hi))


def quick_start(ex, rng):
    return Opts(lead=(0.0, 1.0))


def bench_move_in_first(lo, hi):
    """Bench: lo-hi of the move over the chest is finished during the first descent."""
    def f(ex, rng):
        st = {}

        def unrack(tl, rng, rack, rest_h, walk, rom0):
            tl.move(_u(rng, (0.6, 1.2)), y1=rest_h)
            frac = _u(rng, (lo, hi))
            st["walk"] = walk
            tl.move(_u(rng, GEN["walk_s"]["bench"]) * (1 - frac), x1=walk * (1 - frac), bob=GEN["walk_bob"], rng=rng)
            tl.hold(_u(rng, (0.0, 0.3)))

        def rep(tl, rng, k, n, rest_h, rom, d_dur, u_dur, shape, sway, fp):
            if k == 0:
                tl.move(d_dur, y1=rest_h - rom, x1=st["walk"], shape="cos")
                fin = tl.now
                tl.hold(_u(rng, fp))
                fout = r0 = tl.now
                tl.move(u_dur, y1=rest_h, shape=shape)
                return TrueRep(fin, fout, r0, tl.now, rom, rom / (tl.now - r0))
            return std_rep(tl, "bench", rest_h, rom, d_dur, u_dur, shape, sway, _u(rng, fp))
        return Opts(hooks={"unrack": unrack, "rep": rep})
    return f


def bench_rack_last(lo, hi):
    """Bench: the last rep is put on the hooks before lockout, at lo-hi of the hook->lockout height."""
    def f(ex, rng):
        st = {}

        def unrack(tl, rng, rack, rest_h, walk, rom0):
            st["hook"] = rack
            tl.move(_u(rng, (0.6, 1.2)), y1=rest_h)
            tl.move(_u(rng, GEN["walk_s"]["bench"]), x1=walk, bob=GEN["walk_bob"], rng=rng)
            tl.hold(_u(rng, GEN["settle"]))

        def rep(tl, rng, k, n, rest_h, rom, d_dur, u_dur, shape, sway, fp):
            if k < n - 1:
                return std_rep(tl, "bench", rest_h, rom, d_dur, u_dur, shape, sway, _u(rng, fp))
            hook = st["hook"]
            h = hook + _u(rng, (lo, hi)) * (rest_h - hook)
            tl.move(d_dur, y1=rest_h - rom, shape="cos", sway=sway)
            fin = tl.now
            tl.hold(_u(rng, fp))
            fout = r0 = tl.now
            dist = h - (rest_h - rom)
            tl.move(u_dur * dist / rom, y1=h, shape=shape)
            r1 = tl.now
            tl.move(_u(rng, (0.6, 1.2)), x1=0.0, y1=hook + 0.01)
            tl.move(0.3, y1=hook)
            return TrueRep(fin, fout, r0, r1, dist, dist / (r1 - r0))

        def post(tl, rng, rest_h, rom0):
            tl.hold(_u(rng, (0.5, 2.0)))
        return Opts(hooks={"unrack": unrack, "rep": rep, "post": post})
    return f


def tail(lo, hi):
    return lambda ex, rng: Opts(tail=(lo, hi))


def double_move(ex, rng):
    """One rep rises, sinks back more than 14 cm, then rises to the end (sets with ROM >= 0.32 m only)."""
    st = {}

    def rep(tl, rng, k, n, rest_h, rom, d_dur, u_dur, shape, sway, fp):
        if "k" not in st:
            st["k"] = int(rng.integers(n))
        if k != st["k"]:
            return std_rep(tl, ex, rest_h, rom, d_dur, u_dur, shape, sway, _u(rng, fp))
        x0 = tl.cx
        up_frac = _u(rng, (0.5, 0.8))
        sink = _u(rng, (0.14, max(0.141, min(0.18, up_frac * rom - 0.01))))
        if ex == "deadlift":
            base, top = rest_h, rest_h + rom
            r0 = tl.now
            tl.move(u_dur * up_frac, y1=base + up_frac * rom)
            tl.move(_u(rng, (0.4, 0.8)), y1=base + up_frac * rom - sink)
            tl.move(u_dur, y1=top)
            r1 = fin = tl.now
            tl.hold(_u(rng, fp))
            fout = tl.now
            tl.move(d_dur, y1=base, x1=x0)
            return TrueRep(fin, fout, r0, r1, rom, rom / (r1 - r0))
        bot = rest_h - rom
        tl.move(d_dur, y1=bot, shape="cos", sway=sway)
        fin = tl.now
        tl.hold(_u(rng, fp))
        fout = r0 = tl.now
        tl.move(u_dur * up_frac, y1=bot + up_frac * rom)
        tl.move(_u(rng, (0.4, 0.8)), y1=bot + up_frac * rom - sink)
        tl.move(u_dur, y1=rest_h, x1=x0)
        return TrueRep(fin, fout, r0, tl.now, rom, rom / (tl.now - r0))

    rom = None
    m, s, lo, hi = GEN["rom"][ex]
    for _ in range(100):
        r = float(np.clip(rng.normal(m, s), lo, hi))
        if r >= 0.32:
            rom = r
            break
    return Opts(rom=rom or 0.32, hooks={"rep": rep})


def first_low(lo, hi):
    """The first rep starts lo-hi lower than the rest height."""
    def f(ex, rng):
        def pre(tl, rng, rest_h, rom0):
            tl.move(_u(rng, (0.5, 1.0)), y1=rest_h - _u(rng, (lo, hi)))
            tl.hold(_u(rng, (0.5, 1.5)))
        return Opts(hooks={"pre": pre})
    return f


def dead_first_side(ex, rng):
    def rep(tl, rng, k, n, rest_h, rom, d_dur, u_dur, shape, sway, fp):
        x_end = tl.cx + _u(rng, (0.09, 0.15)) * rng.choice([-1, 1]) if k == 0 else None
        return std_rep(tl, "deadlift", rest_h, rom, d_dur, u_dur, shape, sway, _u(rng, fp), x_end=x_end)
    return Opts(hooks={"rep": rep})


def dead_float(pause, first_only):
    """Deadlift: lift 5-12 cm off the floor, hold `pause` s, then pull."""
    def f(ex, rng):
        def rep(tl, rng, k, n, rest_h, rom, d_dur, u_dur, shape, sway, fp):
            if first_only and k > 0:
                return std_rep(tl, "deadlift", rest_h, rom, d_dur, u_dur, shape, sway, _u(rng, fp))
            fl = _u(rng, (0.05, 0.12))
            x0 = tl.cx
            tl.move(_u(rng, (0.3, 0.6)), y1=rest_h + fl)
            tl.hold(_u(rng, pause))
            r0 = tl.now
            tl.move(u_dur * (rom - fl) / rom, y1=rest_h + rom, shape=shape, sway=sway)
            r1 = fin = tl.now
            tl.hold(_u(rng, fp))
            fout = tl.now
            tl.move(d_dur, y1=rest_h, x1=x0)
            return TrueRep(fin, fout, r0, r1, rom - fl, (rom - fl) / (r1 - r0))
        return Opts(hooks={"rep": rep})
    return f


def x_outliers(ex, rng):
    return Opts(outliers_x=True)


def slowmo(k):
    return lambda ex, rng: Opts(speed_factor=float(k))


def _ballistic(tl, h, n_steps=None):
    """Bar thrown up h metres from the current height and caught/landed at the same height."""
    v0 = np.sqrt(2 * G * h)
    T = 2 * v0 / G
    y0 = tl.cy
    n = max(2, int(round(T / 0.001)))
    tt = np.arange(1, n + 1) * (T / n)
    tl._add(n, y0 + v0 * tt - G * tt ** 2 / 2, np.full(n, tl.cx))
    tl.cy = y0
    tl.y[-1][-1] = y0


def dead_drop(lo, hi):
    """Deadlift: the last rep is dropped from lockout and bounces lo-hi."""
    def f(ex, rng):
        def rep(tl, rng, k, n, rest_h, rom, d_dur, u_dur, shape, sway, fp):
            if k < n - 1:
                return std_rep(tl, "deadlift", rest_h, rom, d_dur, u_dur, shape, sway, _u(rng, fp))
            r0 = tl.now
            tl.move(u_dur, y1=rest_h + rom, shape=shape, sway=sway)
            r1 = fin = tl.now
            tl.hold(_u(rng, fp))
            fout = tl.now
            # free fall
            T = np.sqrt(2 * rom / G)
            m = max(2, int(round(T / 0.001)))
            tt = np.arange(1, m + 1) * (T / m)
            tl._add(m, rest_h + rom - G * tt ** 2 / 2, np.full(m, tl.cx))
            tl.cy = rest_h
            tl.y[-1][-1] = rest_h
            h = _u(rng, (lo, hi))
            _ballistic(tl, h)
            if h > 0.06:
                _ballistic(tl, h * 0.3)
            return TrueRep(fin, fout, r0, r1, rom, rom / (r1 - r0))
        return Opts(hooks={"rep": rep})
    return f


def dead_bounce_before(lo, hi):
    def f(ex, rng):
        def pre(tl, rng, rest_h, rom0):
            _ballistic(tl, _u(rng, (lo, hi)))
            tl.hold(_u(rng, (1.0, 2.5)))
        return Opts(hooks={"pre": pre})
    return f


def bench_continuous(lo, hi):
    def f(ex, rng):
        T = _u(rng, (lo, hi))
        return Opts(down=(T * 0.5, T * 0.5), up=(T * 0.45, T * 0.45), far_pause=(0.0, 0.0), rest_pause=(0.0, 0.0))
    return f


def unrack_rerack_relift(lo, hi):
    """Unrack lo-hi, put the bar back on the rack, lift it again, then the normal set."""
    def f(ex, rng):
        def unrack(tl, rng, rack, rest_h, walk, rom0):
            tl.move(_u(rng, (0.6, 1.2)), y1=rest_h)
            tl.hold(_u(rng, (0.5, 1.5)))
            tl.move(_u(rng, (0.6, 1.2)), y1=rack)
            tl.hold(_u(rng, (0.5, 1.5)))
            tl.move(_u(rng, (0.6, 1.2)), y1=rest_h)
            tl.move(_u(rng, GEN["walk_s"][ex]), x1=walk, bob=GEN["walk_bob"], rng=rng)
            tl.hold(_u(rng, GEN["settle"]))
        return Opts(unrack=(lo, hi), hooks={"unrack": unrack})
    return f


def big_unrack_small_rom(ex, rng):
    return Opts(unrack=(0.13, 0.18), rom_range=(0.15, 0.22), tail=(30.0, 60.0))


def cut_first(ex, rng):
    return Opts(cut_in_first=True)


def partial_mid(ex, rng):
    st = {}

    def rep(tl, rng, k, n, rest_h, rom, d_dur, u_dur, shape, sway, fp):
        if "k" not in st:
            st["k"] = int(rng.integers(1, n - 1)) if n >= 3 else 1
        out = std_rep(tl, ex, rest_h, rom, d_dur, u_dur, shape, sway, _u(rng, fp))
        if k == st["k"]:
            tl.hold(_u(rng, GEN["rest_pause"][ex]))
            pr = rom * _u(rng, (0.3, 0.5))
            std_rep(tl, ex, rest_h, pr, d_dur * 0.5, u_dur * 0.5, "cos", 0.0, _u(rng, fp))
        return out
    return Opts(reps=int(rng.integers(3, 10)), hooks={"rep": rep})


def fail_last(ex, rng):
    """Squat, bench: the last rep comes up 10-50% of the way, stalls 0.5-2 s and sinks onto the safeties
    0-5 cm below the bottom, where it stays to the end of the recording. Not in the truth."""
    def rep(tl, rng, k, n, rest_h, rom, d_dur, u_dur, shape, sway, fp):
        if k < n - 1:
            return std_rep(tl, ex, rest_h, rom, d_dur, u_dur, shape, sway, _u(rng, fp))
        bottom = rest_h - rom
        tl.move(d_dur, y1=bottom, shape="cos", sway=sway)
        tl.hold(_u(rng, fp))
        f = _u(rng, (0.1, 0.5))
        tl.move(u_dur * f * 1.5, y1=bottom + f * rom, shape="cos")
        tl.hold(_u(rng, (0.5, 2.0)))
        tl.move(_u(rng, (0.4, 1.0)), y1=bottom - _u(rng, (0.0, 0.05)), shape="cos")
        return None

    def post(tl, rng, rest_h, rom0):
        tl.hold(_u(rng, (0.5, 2.0)))
    return Opts(reps=int(rng.integers(2, 10)), hooks={"rep": rep, "post": post})


def base(ex, rng):
    return Opts()


# name: (exercises, factory, doc final, doc live, extra)
SCENARIOS = {
    "기본 세트": (ALL, base, None, None),
    "먼 점에서 3~6초 멈춤": (ALL, far_pause(3, 6), None, None),
    "4~6초에 걸쳐 내림": (ALL, slow_down(4, 6), None, None),
    "데드: 터치 앤 고로 3~6 cm 튐": (("deadlift",), dead_touch_and_go, None, None),
    "벤치: rep 사이에 바를 8~12 cm 고쳐 잡음": (("bench",), regrip(), None, None),
    "첫 rep 앞에서만 8~12 cm 내려 1.2~2.5초 멈춤": (TOP, dip_pause((1.2, 2.5), True), None, None),
    "첫 rep 앞에서 8~12 cm 내려 0.4~0.9초 멈춤": (TOP, dip_pause((0.4, 0.9), True), None, None),
    "스쿼트: 워크아웃 중 3~6 cm 출렁임": (("squat",), walk_bob, None, None),
    "1회만 하고 바로 리랙": (ALL, one_rep, None, None),
    "rep 도중 원판이 0.6~1초 가림": (ALL, occlude(0.6, 1.0), None, None),
    "데드: 녹화 시작 1초 안에 당김": (("deadlift",), quick_start, None, None),
    "벤치: 언랙 이동의 40~80%를 첫 하강 중에 마침": (("bench",), bench_move_in_first(0.4, 0.8), None, None),
    "벤치: 언랙 이동의 20~50%를 첫 하강 중에 마침": (("bench",), bench_move_in_first(0.2, 0.5), None, None),
    "벤치: 마지막 rep을 락아웃 전에 랙에 검 (20~60%)": (("bench",), bench_rack_last(0.2, 0.6), None, None),
    "벤치: 마지막 rep을 락아웃 전에 랙에 검 (60~100%)": (("bench",), bench_rack_last(0.6, 1.0), None, None),
    "세트가 끝난 뒤 30~60초 더 녹화": (ALL, tail(30, 60), None, None),
    "올라오다 14 cm 넘게 가라앉았다 다시 올림": (ALL, double_move, None, None),
    "첫 rep을 6~10 cm 낮은 데서 시작": (TOP, first_low(0.06, 0.10), None, None),
    "첫 rep을 2~4.5 cm 낮은 데서 시작": (TOP, first_low(0.02, 0.045), None, None),
    "데드: 첫 rep을 9~15 cm 옆에 내려놓음": (("deadlift",), dead_first_side, None, None),
    "데드: 매 rep 5~12 cm 띄워 1.5~3초 멈췄다 당김": (("deadlift",), dead_float((1.5, 3.0), False), None, None),
    "데드: 첫 rep만 5~12 cm 띄워 멈췄다 당김": (("deadlift",), dead_float((1.5, 3.0), True), None, None),
    "데드: 5~12 cm 띄워 0.4~0.9초 멈췄다 당김": (("deadlift",), dead_float((0.4, 0.9), False), None, None),
    "가로 위치에도 이상치 1%": (ALL, x_outliers, None, None),
    "4배 슬로모션": (ALL, slowmo(4), None, None),
    "데드: 마지막 rep을 떨어뜨려 8~18 cm 튐": (("deadlift",), dead_drop(0.08, 0.18), None, None),
    "데드: 마지막 rep을 떨어뜨려 18~30 cm 튐": (("deadlift",), dead_drop(0.18, 0.30), None, None),
    # the 14 below the 0.995 line (doc final / live)
    "벤치: 고쳐 잡음 8~12 cm, ROM 0.15~0.22 m": (("bench",), regrip(None, (0.15, 0.22)), 0.548, 0.643),
    "벤치: 고쳐 잡음 10 cm, ROM 0.15~0.22 m": (("bench",), regrip(0.10, (0.15, 0.22)), 0.837, 0.918),
    "벤치: 고쳐 잡음 11 cm, ROM 0.15~0.22 m": (("bench",), regrip(0.11, (0.15, 0.22)), 0.408, 0.543),
    "벤치: 고쳐 잡음 12 cm, ROM 0.15~0.22 m": (("bench",), regrip(0.12, (0.15, 0.22)), 0.193, 0.238),
    "매 rep 8~12 cm 내려 1.2~2.5초 멈췄다 내려감": (TOP, dip_pause((1.2, 2.5), False), 0.998, 0.983),
    "녹화가 rep 도중에 시작 (잘린 rep은 정답에서 뺌)": (ALL, cut_first, 0.920, 0.805),
    "rep 도중 원판이 1~2초 가림": (ALL, occlude(1.0, 2.0), 0.909, 0.891),
    "큰 언랙 + 작은 ROM, 세트 뒤 30~60초 더 녹화": (TOP, big_unrack_small_rom, 0.999, 0.999),
    "벤치: rep당 0.85~1.21초로 쉬지 않고 반복": (("bench",), bench_continuous(0.85, 1.21), 0.940, 0.885),
    "벤치: rep당 1.04~1.38초로 쉬지 않고 반복": (("bench",), bench_continuous(1.04, 1.38), 0.972, 0.952),
    "큰 언랙(13~18 cm) 뒤 랙에 다시 놓았다 듦": (TOP, unrack_rerack_relift(0.13, 0.18), 0.958, 0.000),
    "언랙 3~10 cm 뒤 랙에 다시 놓았다 듦": (TOP, unrack_rerack_relift(0.03, 0.10), 1.000, 0.991),
    "8배 슬로모션": (ALL, slowmo(8), 0.222, 0.224),
    "데드: 첫 rep 전에 바가 18~30 cm 튐": (("deadlift",), dead_bounce_before(0.18, 0.30), 0.897, 0.372),
    # run separately
    "세트 중간에 부분 rep 1개": (ALL, partial_mid, 0.993, 0.999),
    "방향을 거꾸로 줌": (ALL, base, 0.005, 0.025),
}


# ---- extra situations used only by the threshold sweeps (근거·출처 tab, 임계값 표) --------------

def short_lockout(ex, rng):
    """Squat/bench: each rep comes back only 85-100% of the way; the next rep starts from there."""
    st = {"h": None}

    def rep(tl, rng, k, n, rest_h, rom, d_dur, u_dur, shape, sway, fp):
        start = tl.cy
        x0 = tl.cx
        tl.move(d_dur, y1=rest_h - rom, shape="cos", sway=sway)
        fin = tl.now
        tl.hold(_u(rng, fp))
        fout = r0 = tl.now
        top = rest_h - _u(rng, (0.0, 0.15)) * rom
        tl.move(u_dur, y1=top, x1=x0, shape=shape)
        return TrueRep(fin, fout, r0, tl.now, top - (rest_h - rom), (top - rest_h + rom) / (tl.now - r0))
    return Opts(hooks={"rep": rep})


EXTRA = {
    "락아웃이 0~15% 모자란 스쿼트·벤치": (TOP, short_lockout),
    "마지막 rep을 곧장 랙에 거는 벤치": (("bench",), bench_rack_last(0.0, 0.0)),
    "ROM 0.15~0.22 m인 스쿼트·벤치": (TOP, lambda ex, rng: Opts(rom_range=(0.15, 0.22))),
}
