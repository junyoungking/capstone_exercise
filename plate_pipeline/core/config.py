"""All tunable values in one place.

Names follow the 구현 설계 tab ("설정값"). Units: metres, seconds, m/s.
Only the groups used by the signal/judge/measure core live here for now.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrackCfg:
    # ByteTrack: weak evidence may maintain an identity, never initialize one.
    score_low: float = 0.05
    score_high: float = 0.3
    # Side view: the bar travels vertically; retain a half-diameter x corridor.
    lost_corridor_s: float = 0.3
    gate: float = 0.5
    reacquire_size_min: float = 0.75
    reacquire_size_max: float = 1.33
    # Fixed physical diameter; even walking changes apparent size slowly (~30%/s).
    scale_rate_s: float = 0.5
    # 0.15 diameter is about 7 cm for a 45 cm plate: reject preceding drift.
    drift_fraction: float = 0.15
    # Scale from measured sizes only (det, det_low, rim); KLT sizes are estimates.
    # Keep the window short: the apparent size does change within a rep with perspective
    # (public squat sample: the rigid ArUco tag on the plate grows 12-14% at the bottom).
    measured_half_window_s: float = 0.25
    max_pts: int = 150
    min_pts: int = 25
    inner: float = 0.85
    hub: float = 0.45
    fb_px: float = 1.0
    win: int = 21
    levels: int = 3


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
class SetCfg:
    gap_s: float = 45.0          # no rep and no excursion >= min_amp this long after the last rep -> close the set
    label_hold_s: float = 3.0    # exercise label different from the set's for this long (no excursion) -> close
    provisional_s: float = 5.0   # live: provisional report after this long without an excursion


@dataclass(frozen=True)
class CmpCfg:
    alpha: float = 0.025              # false "different" per value the multiplier table is built for
    table: str = "crit_table.json"    # in core/, made by sim/crit.py
    set_min_n: int = 4                # in-set: values needed
    set_group_n: int = 2              # in-set: first / last reps averaged
    ref_user_min_n: int = 3           # vs reference: my values needed (reference with >= short_ref_n values)
    short_ref_n: int = 3              # reference with fewer values is "short" (table C, my scatter only)
    short_ref_user_min_n: int = 4     # vs a short reference: my values needed
    min_change_s: float = 0.15
    min_change_m: float = 0.02
    floor_s: float = 0.07             # scatter floor for times (or 1/fps if larger)
    floor_m: float = 0.01             # scatter floor for distances


@dataclass(frozen=True)
class TextCfg:
    max_shown: int = 3
    time_digits: int = 2
    llm_polish: bool = False
    banned: tuple = ("좋", "나쁘", "잘못", "틀렸", "틀린", "올바", "정확한 자세", "해야", "하세요", "하십시오",
                     "권장", "추천", "위험", "부상")


@dataclass(frozen=True)
class PoseCfg:
    backend: str = "rtmpose"
    mode: str = "lightweight"
    score_reference: float = 0.6
    switch_margin: float = 0.5  # plate heights closer, and at least 25% closer
    every_n: int = 1
    num_poses: int = 2
    min_conf: float = 0.45
    vis_thr: float = 0.3
    near_plate: float = 0.5
    model_path: str = None


@dataclass(frozen=True)
class Config:
    live: LiveCfg = field(default_factory=LiveCfg)
    final: FinalCfg = field(default_factory=FinalCfg)
    rep: RepCfg = field(default_factory=RepCfg)
    speed: SpeedCfg = field(default_factory=SpeedCfg)
    sets: SetCfg = field(default_factory=SetCfg)
    cmp: CmpCfg = field(default_factory=CmpCfg)
    text: TextCfg = field(default_factory=TextCfg)
    pose: PoseCfg = field(default_factory=PoseCfg)


DEFAULT = Config()
