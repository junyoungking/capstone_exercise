"""Live calculation of the bar signal (pipeline tab, 라이브 계산 column). Never looks at future samples.

LiveCleaner.push(t, y, x) takes one raw sample and returns (t, y_f, x_f, gap_before) or None.
  start   : the first cfg.start_n valid samples -> their median starts the signal
  outliers: farther than out_speed * dt + out_add_m from the median of the last 3 accepted values is
            dropped; after accept_after drops in a row the sample is accepted anyway
  gaps    : nothing is produced; a gap longer than restart_gap_s discards the filter and starts again
  filter  : median of the last 3 accepted values, then a causal 2nd-order Butterworth (constant-step
            approximation)
"""
import numpy as np
from scipy.signal import butter, lfilter_zi

from ..config import DEFAULT, LiveCfg


class _Biquad:
    def __init__(self, fps, hz, start):
        b, a = butter(2, hz / (fps / 2.0))
        self.b, self.a = b, a
        zi = lfilter_zi(b, a) * start
        self.z1, self.z2 = float(zi[0]), float(zi[1])

    def step(self, v):
        b, a = self.b, self.a
        out = b[0] * v + self.z1
        self.z1 = b[1] * v - a[1] * out + self.z2
        self.z2 = b[2] * v - a[2] * out
        return out


def _med3(seq):
    a = seq[-3:]
    if len(a) == 1:
        return a[0]
    if len(a) == 2:
        return 0.5 * (a[0] + a[1])
    x, y, z = a
    return max(min(x, y), min(max(x, y), z))


class LiveCleaner:
    def __init__(self, fps, cfg: LiveCfg = DEFAULT.live):
        self.fps, self.cfg = fps, cfg
        self.last_in_t = None
        self.last_out_t = None
        self._reset()

    def _reset(self):
        self.started = False
        self.buf = []
        self.acc_y, self.acc_x = [], []
        self.last_acc_t = None
        self.rejects = 0
        self.fy = self.fx = None

    def push(self, t, y, x):
        cfg = self.cfg
        if self.last_in_t is not None and t - self.last_in_t > cfg.restart_gap_s:
            self._reset()
        self.last_in_t = t
        if not self.started:
            self.buf.append((y, x))
            if len(self.buf) < cfg.start_n:
                return None
            y0 = float(np.median([b[0] for b in self.buf]))
            x0 = float(np.median([b[1] for b in self.buf]))
            self.started = True
            self.acc_y, self.acc_x = [y0], [x0]
            self.last_acc_t = t
            self.fy = _Biquad(self.fps, cfg.filter_hz, y0)
            self.fx = _Biquad(self.fps, cfg.filter_hz, x0)
            return self._out(t, self.fy.step(y0), self.fx.step(x0))
        ref = _med3(self.acc_y)
        dt = t - self.last_acc_t
        if abs(y - ref) > cfg.out_speed * dt + cfg.out_add_m and self.rejects < cfg.accept_after:
            self.rejects += 1          # after accept_after drops in a row the next sample is accepted
            return None
        self.rejects = 0
        self.acc_y.append(y); self.acc_x.append(x)
        if len(self.acc_y) > 3:
            self.acc_y.pop(0); self.acc_x.pop(0)
        self.last_acc_t = t
        return self._out(t, self.fy.step(_med3(self.acc_y)), self.fx.step(_med3(self.acc_x)))

    def _out(self, t, yf, xf):
        g = 0.0 if self.last_out_t is None else t - self.last_out_t
        self.last_out_t = t
        return t, yf, xf, g


def clean_live_all(t, y, x, fps, cfg: LiveCfg = DEFAULT.live):
    """Convenience for tests: run the live cleaner over a whole recording. Returns arrays (t, y, x, gap)."""
    lc = LiveCleaner(fps, cfg)
    out = [r for r in (lc.push(a, b, c) for a, b, c in zip(t, y, x)) if r is not None]
    if not out:
        e = np.array([])
        return e, e, e, e
    a = np.array(out)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]
