"""Position noise (pipeline tab, 마지막 표 "위치 잡음": a resting bar filmed 10 s, sd of the centre <= 0.5 cm).

python -m eval.noise VIDEO [--model models/plate_tiny.onnx | --box x0,y0,x1,y1] [--plate-mm 450] [--skip 1]
The bar must not move in the clip (on the rack or the floor). A straight-line drift is removed first.
"""
import argparse

import numpy as np

from core.recog.plate import OnnxPlateDetector
from core.signal.height import height_signal
from server.jobs import recognize


def noise_cm(rows, plate_mm=450.0, skip_s=1.0):
    t, y, x = height_signal(rows, plate_mm)
    m = t >= t[0] + skip_s
    out = {}
    for name, v in (("vertical", y), ("horizontal", x)):
        r = v[m] - np.polyval(np.polyfit(t[m], v[m], 1), t[m])
        out[name] = 100 * float(np.std(r))
    out["frames"] = int(m.sum())
    out["plate_px"] = float(np.median([r["plate"]["size_px"][1] for r in rows if r.get("plate")]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default=None)
    ap.add_argument("--box", default=None)
    ap.add_argument("--plate-mm", type=float, default=450.0)
    ap.add_argument("--skip", type=float, default=1.0)
    a = ap.parse_args()
    det = OnnxPlateDetector(a.model) if a.model else None
    box = tuple(float(v) for v in a.box.split(",")) if a.box else None
    rows = recognize(a.video, det, box, rim=det is None)
    r = noise_cm(rows, a.plate_mm, a.skip)
    ok = "OK" if r["vertical"] <= 0.5 else "over 0.5 cm"
    print(f"vertical sd {r['vertical']:.2f} cm ({ok}), horizontal sd {r['horizontal']:.2f} cm, "
          f"{r['frames']} frames, plate {r['plate_px']:.0f} px")


if __name__ == "__main__":
    main()
