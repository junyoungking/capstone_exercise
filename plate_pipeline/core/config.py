"""All tunable values in one place.

Names follow the 구현 설계 tab ("설정값"). Units: metres, seconds, m/s.
Only the groups used by the signal/judge/measure core live here for now.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LiveCfg:
    start_n: int = 5            # valid samples before the live signal starts (median)
    out_speed: float = 6.0      # m/s, outlier gate slope
    out_add_m: float = 0.08     # m, outlier gate offset
    accept_after: int = 4       # accept after this many consecutive rejects
    restart_gap_s: float = 0.5  # gap that resets the live filter
    filter_hz: float = 4.0      # causal 2nd-order Butterworth
    med_n: int = 3              # median of last accepted values before the filter


@dataclass(frozen=True)
class FinalCfg:
    nb_n: int = 3               # neighbours per side
    nb_s: float = 0.15          # neighbour window (s)
    out_add_m: float = 0.06     # two-sided gate offset
    out_acc: float = 6.0        # two-sided gate, m/s^2 * dt^2
    edge_add_m: float = 0.06    # one-sided gate offset (passes 2 and 3)
    edge_speed: float = 5.0     # one-sided gate, m/s * dt
    passes: int = 3
    interp_s: float = 0.5       # gaps up to this are interpolated
    filter_hz: float = 5.0      # zero-phase Butterworth
    filter_min_n: int = 20      # segments this short are not filtered


@dataclass(frozen=True)
class RepCfg:
    entry_m: float = 0.06       # leave the reference by this much -> in progress
    min_amp_m: float = 0.13     # minimum excursion for a rep
    ret: float = 0.7            # fraction of the excursion that must come back
    stall_s: float = 1.0        # small excursion: far point not updated this long -> cancel
    bounce_s: float = 0.4       # rep must take at least this long after entry
    fail_s: float = 20.0        # no return for this long -> failed attempt
    partial: float = 0.6        # amp below this x set median -> partial rep
    other_amp: float = 0.8      # "other spot": amp below this x median ...
    other_dx_m: float = 0.08    # ... and far point x differs by more than this
    gap_s: float = 0.5          # gap next to far point / entry -> skip depth / bounce checks
    restart_s: float = 3.0      # gap after which the state restarts at the current position


@dataclass(frozen=True)
class SpeedCfg:
    d_frac: float = 0.05        # endpoint offset d = 5% of ROM ...
    d_min_m: float = 0.015      # ... at least 1.5 cm
    end_avg_s: float = 0.1      # endpoint height = mean of +-0.1 s
    drop_m: float = 0.03        # raise ends when the bar comes back down this much
    blank_s: float = 0.2        # raw gap this long inside the raise -> no speed
    min_rom_m: float = 0.05
    live_flat_s: float = 0.3    # live: raise ended if no new high for this long
    live_pre_s: float = 1.5     # live recompute starts this long before the raise


@dataclass(frozen=True)
class Config:
    live: LiveCfg = field(default_factory=LiveCfg)
    final: FinalCfg = field(default_factory=FinalCfg)
    rep: RepCfg = field(default_factory=RepCfg)
    speed: SpeedCfg = field(default_factory=SpeedCfg)


DEFAULT = Config()
