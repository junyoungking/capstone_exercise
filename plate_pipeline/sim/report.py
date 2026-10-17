"""Compare results/*.json with the numbers in the design doc (iteration 16) and write results/compare.md.

A difference is "within sampling error" when it is inside the 95% interval of the difference of two
proportions (doc sets and our sets). For the 27 scenarios the doc only says ">= 0.995 for both".
"""
import json
import math
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")

DOC_COUNTS = {   # fps: noise -> (final, live); 30 fps from the iteration 16 log (3, 4 cm) and the figure (~1.0)
    15: {0.3: (0.997, 0.999), 1: (0.996, 0.999), 2: (0.997, 0.999), 3: (0.991, 0.987), 4: (0.880, 0.834)},
    30: {3: (0.998, 0.998), 4: (0.976, 0.887)},
    60: {0.3: (1.0, 1.0), 1: (1.0, 1.0), 2: (1.0, 1.0), 3: (1.0, 1.0), 4: (0.996, 0.969)},
}
DOC_TIME30 = {0.3: (1.000, 0.997), 1: (0.996, 0.996), 2: (0.993, 0.990), 3: (0.988, 0.986), 4: (0.959, 0.872)}
DOC_SPEED30 = {  # noise cm -> rmse (live recompute), bias %, over10 %, gap blank %, not computed %
    0.2: (0.015, -1.1, 0.2, 5.0, 0.0), 0.5: (0.021, -1.9, 2.2, 5.1, 0.0),
    1: (0.036, -2.9, 18.3, 5.2, 0.1), 2: (0.070, -4.1, 45.8, 4.5, 12.1),
}


def ci(p1, n1, p2, n2):
    return 1.96 * math.sqrt(max(p1 * (1 - p1), 1e-4) / n1 + max(p2 * (1 - p2), 1e-4) / n2)


def mark(ours, doc, n_ours, n_doc):
    if doc is None:
        return ""
    d = ours - doc
    return "같음" if abs(d) <= ci(ours, n_ours, doc, n_doc) else ("높음" if d > 0 else "낮음")


def main():
    out = ["# 문서(반복 16)와 비교", "",
           "\"같음\"은 두 비율의 차이가 표본 오차(95%) 안이라는 뜻이다. 생성기를 다시 만들었으므로 숫자가 같을 이유는 없고, 같은 방향과 크기인지를 본다.", ""]
    bc = json.load(open(os.path.join(R, "base_counts.json")))
    out += ["## 횟수가 맞은 세트 (최종 / 라이브)", "", "| 조건 | 이번 | 문서 | 판정 (최종, 라이브) |", "| --- | --- | --- | --- |"]
    for fps in (15, 30, 60):
        for nz in (0.3, 1, 2, 3, 4):
            k = f"{fps}fps_{nz:g}cm"
            if k not in bc:
                continue
            r = bc[k]
            d = DOC_COUNTS.get(fps, {}).get(nz)
            ds = f"{d[0]:.3f} / {d[1]:.3f}" if d else "그림 (약 1.0)"
            m = f"{mark(r['final'], d[0], r['sets'], 900)}, {mark(r['live'], d[1], r['sets'], 900)}" if d else ""
            out.append(f"| {fps} fps, 잡음 {nz:g} cm | {r['final']:.3f} / {r['live']:.3f} | {ds} | {m} |")
    out += ["", "## 30 fps, rep의 시각까지 맞음 (최종 / 라이브)", "", "| 잡음 | 이번 | 문서 | 판정 |", "| --- | --- | --- | --- |"]
    for nz, d in DOC_TIME30.items():
        r = bc.get(f"30fps_{nz:g}cm")
        if r:
            out.append(f"| {nz:g} cm | {r['final_time']:.3f} / {r['live_time']:.3f} | {d[0]:.3f} / {d[1]:.3f} | "
                       f"{mark(r['final_time'], d[0], r['sets'], 900)}, {mark(r['live_time'], d[1], r['sets'], 900)} |")
    sp = json.load(open(os.path.join(R, "base_speed.json")))
    out += ["", "## 평균 속도 (30 fps)", "",
            "| 잡음 | RMSE 최종 / rep 직후 재계산 (m/s) | 문서 rep 직후 | 편향 중앙값 % (문서) | 10% 넘게 틀림 % (문서) | 공백으로 비움 % (문서) | 계산 안 됨 % (문서) |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for nz, d in DOC_SPEED30.items():
        s = sp.get(f"30fps_{nz:g}cm")
        if not s:
            continue
        f, l = s["final"], s["live_recompute"]
        out.append(f"| {nz:g} cm | {f['rmse']:.3f} / {l['rmse']:.3f} | {d[0]:.3f} | {f['bias_med_pct']:.1f} ({d[1]}) | "
                   f"{f['over10_pct']:.1f} ({d[2]}) | {f['blank_gap_pct']:.1f} ({d[3]}) | {f['blank_other_pct']:.1f} ({d[4]}) |")
    bk = json.load(open(os.path.join(R, "break.json")))
    out += ["", "## 깨뜨리기 시험 (30 fps, 잡음 1 cm, 종목당 600세트)", "",
            "문서는 처음 27가지를 \"최종과 라이브 모두 0.995 이상, 리랙을 미완료로 적은 세트 5% 미만\"으로만 적었다.", "",
            "| 상황 | 세트 | 이번 (최종 / 라이브) | 문서 | 판정 | 리랙을 미완료로 |", "| --- | --- | --- | --- | --- | --- |"]
    for name, r in bk.items():
        dF, dL = r["doc"]
        if dF is None:
            ok = r["final"] >= 0.995 - ci(r["final"], r["sets"], 0.995, 1800) and r["live"] >= 0.995 - ci(r["live"], r["sets"], 0.995, 1800)
            ds, m = "0.995 이상", ("같음" if ok else "낮음")
        else:
            ds = f"{dF:.3f} / {dL:.3f}"
            m = f"{mark(r['final'], dF, r['sets'], r['sets'])}, {mark(r['live'], dL, r['sets'], r['sets'])}"
        extra = f"{100 * r['rerack_failed']:.1f}%"
        if name.startswith("세트 중간에 부분 rep"):
            extra += f"; 부분으로 표시 {r['partial_flag'][0]:.3f} / {r['partial_flag'][1]:.3f} (문서 0.913 / 0.908)"
        out.append(f"| {name} | {r['sets']} | {r['final']:.3f} / {r['live']:.3f} | {ds} | {m} | {extra} |")
    open(os.path.join(R, "compare.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
