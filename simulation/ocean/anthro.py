"""
Human-made sources: vessel traffic, seismic airgun surveys, and the
instrument/sensor fault modes of the hydrophone + environmental package.

Vessels: Poisson traffic by class and hour-of-day, straight-line tracks with
closest-point-of-approach (CPA), Ross source levels, blade-rate/shaft/
machinery tonal comb with Doppler, and a Ross-shaped cavitation continuum.
"""

from dataclasses import dataclass

import numpy as np

from .physics import (KNOT_MS, db_to_upa, ross_source_level_db, shape_noise, transmission_loss_db,
                      vessel_broadband_psd_db)

# class: length range m, speed range kn, shaft Hz, blades, CPA range m, hourly weight shape, share
VESSEL_CLASSES = {
    "cargo":    dict(L=(150, 300), kn=(10, 16), shaft=(1.3, 1.9), blades=5, cpa=(1200, 3500), share=0.22, bb=12.0),
    "tanker":   dict(L=(180, 330), kn=(8, 14), shaft=(1.1, 1.6), blades=4, cpa=(1500, 3500), share=0.12, bb=16.0),
    "ferry":    dict(L=(40, 90), kn=(16, 24), shaft=(6.0, 10.0), blades=4, cpa=(250, 1500), share=0.14, bb=6.0),
    "fishing":  dict(L=(15, 35), kn=(4, 9), shaft=(7.0, 12.0), blades=3, cpa=(100, 1800), share=0.22, bb=8.0),
    "outboard": dict(L=(5, 12), kn=(14, 30), shaft=(35.0, 60.0), blades=3, cpa=(60, 900), share=0.30, bb=8.0),
}
VESSEL_TYPES = list(VESSEL_CLASSES)


@dataclass
class VesselPassage:
    cls: str
    t_cpa_s: float
    cpa_m: float
    speed_ms: float
    length_m: float
    shaft_hz: float
    blades: int
    source_level_db: float
    tonal_offsets: tuple


def schedule_vessels(n_windows, interval_min, site, rng, doy0=90):
    """Poisson near-field vessel passages over the whole run."""
    total_s = n_windows * interval_min * 60.0
    days = total_s / 86400.0
    shares = np.array([VESSEL_CLASSES[c]["share"] for c in VESSEL_TYPES])
    shares = shares / shares.sum()
    n_expected = site.vessel_rate_per_day * days
    n_v = rng.poisson(n_expected)
    passages = []
    # hour-of-day thinning: daytime-heavy traffic
    while len(passages) < n_v:
        t = rng.uniform(0, total_s)
        hour = (t / 3600.0) % 24
        p = 0.25 + 0.75 * np.exp(-0.5 * ((hour - 13.0) / 4.5) ** 2)
        if rng.random() > p:
            continue
        cls = VESSEL_TYPES[rng.choice(len(VESSEL_TYPES), p=shares)]
        c = VESSEL_CLASSES[cls]
        L = rng.uniform(*c["L"])
        kn = rng.uniform(*c["kn"])
        sl = ross_source_level_db(kn, L) + c["bb"] + rng.normal(0, 2.0)
        passages.append(VesselPassage(
            cls=cls, t_cpa_s=t, cpa_m=rng.uniform(*c["cpa"]), speed_ms=kn * KNOT_MS, length_m=L,
            shaft_hz=rng.uniform(*c["shaft"]), blades=c["blades"], source_level_db=sl,
            tonal_offsets=tuple(rng.uniform(0, 2 * np.pi, 10)),
        ))
    passages.sort(key=lambda p: p.t_cpa_s)
    return passages


def vessel_geometry(p, t_s, hyd_depth):
    """Range and radial velocity (positive = receding) at absolute time(s) t_s."""
    x = p.speed_ms * (np.asarray(t_s) - p.t_cpa_s)
    r = np.sqrt(x**2 + p.cpa_m**2 + (hyd_depth * 0.8) ** 2)
    vr = p.speed_ms * x / r * (1.0)
    return r, vr, x


def render_vessel(p, t0_s, n, fs, ctx, rng, max_range_m=14000.0):
    """Pressure (uPa) from one vessel over [t0_s, t0_s + n/fs). Returns (signal, mid_range)."""
    t = np.arange(n) / fs
    r, vr, x = vessel_geometry(p, t0_s + t, ctx["hyd_depth"])
    r_mid = float(r[n // 2])
    if r_mid > max_range_m:
        return None, r_mid
    c = ctx["c"]
    k = ctx["k"]

    def tl(f, rr):
        return transmission_loss_db(f, rr, ctx["temp"], ctx["sal"], ctx["ph"], ctx["hyd_depth"],
                                    ctx["water_depth"], spreading_k=k, stratification=ctx["strat"])

    # broadband cavitation / machinery continuum
    psd = lambda f: vessel_broadband_psd_db(f, p.source_level_db - 3.0) - tl(f, r_mid)  # noqa: E731
    bb = shape_noise(psd, None, n, fs, rng, f_min=15.0)
    bb = bb * (r_mid / r) ** (k / 20.0 * 1.0)

    # tonal comb: blade-rate harmonics + shaft harmonics + two machinery tones
    blade_rate = p.shaft_hz * p.blades
    freqs, rel_db = [], []
    for h in range(1, 9):
        freqs.append(blade_rate * h)
        rel_db.append(-3.0 * (h - 1))
    for h in (1, 2, 3):
        freqs.append(p.shaft_hz * h)
        rel_db.append(-8.0 - 4.0 * h)
    freqs.extend([30.0 + (p.length_m % 25), 60.0 + (p.length_m % 20)])
    rel_db.extend([-4.0, -7.0])
    tone_sl = p.source_level_db - 7.0
    sig = np.zeros(n)
    for i, (f0, rd) in enumerate(zip(freqs, rel_db)):
        if f0 > 0.45 * fs or f0 < 12.0:
            continue
        a = db_to_upa(tone_sl + rd - tl(f0, r)) * (1.0 + 0.1 * np.sin(2 * np.pi * 0.7 * t + i))
        f_inst = f0 / (1.0 + vr / c)                            # Doppler
        phase = 2 * np.pi * np.cumsum(f_inst) / fs + p.tonal_offsets[i % 10]
        sig += a * np.sin(phase)
    return sig + bb, r_mid


def render_airgun(n, fs, ctx, rng, range_m, sl_db=208.0):
    """One airgun pulse (10-200 Hz, dispersed by shallow-water propagation) at a random time."""
    out = np.zeros(n)
    t0 = rng.uniform(0.2, max(n / fs - 1.6, 0.3))
    m = int(1.4 * fs)
    t = np.arange(m) / fs
    env = (1 - np.exp(-t / 0.08)) * np.exp(-t / 0.35)
    burst = shape_noise(lambda f: np.where((f > 10) & (f < 220), 0.0, -80.0) - 10 * np.log10(np.maximum(f, 10) / 10), None, m, fs, rng)
    burst = burst / (burst.std() + 1e-12)
    tl = transmission_loss_db(70.0, range_m, ctx["temp"], ctx["sal"], ctx["ph"], ctx["hyd_depth"],
                              ctx["water_depth"], spreading_k=ctx["k"], stratification=ctx["strat"])
    amp = db_to_upa(sl_db - 20.0 - tl)
    i0 = int(t0 * fs)
    usable = min(m, n - i0)
    out[i0:i0 + usable] = (amp * env * burst)[:usable]
    return out


def schedule_airgun_campaigns(n_windows, interval_min, rng, prob=0.5):
    """0-2 multi-hour seismic survey campaigns; returns list of (start_idx, end_idx, range_m)."""
    camps = []
    if rng.random() < prob:
        total = n_windows
        for _ in range(int(rng.integers(1, 3))):
            dur = int(rng.integers(20, 90))
            if total - dur - 20 <= 20:
                continue
            s = int(rng.integers(20, total - dur - 10))
            camps.append((s, s + dur, float(rng.uniform(15000, 45000))))
    return camps


# ---------------------------------------------------------------- faults
FAULT_TYPES = ["hum", "dropout", "pops", "env_stuck"]


def schedule_faults(n_windows, rng, per_type_prob=0.4):
    """Episodes of instrument faults: list of dict(type, start, end, params)."""
    out = []
    for ft in FAULT_TYPES:
        for _ in range(int(rng.integers(1, 3))):
            if rng.random() > per_type_prob:
                continue
            dur = {"hum": int(rng.integers(8, 40)), "dropout": int(rng.integers(1, 6)),
                   "pops": int(rng.integers(4, 18)), "env_stuck": int(rng.integers(6, 36))}[ft]
            if n_windows - dur - 30 <= 30:
                continue
            s = int(rng.integers(30, n_windows - dur - 5))
            out.append({"type": ft, "start": s, "end": s + dur,
                        "hz": float(rng.choice([50.0, 60.0])), "level": float(rng.uniform(0.004, 0.03))})
    return out
