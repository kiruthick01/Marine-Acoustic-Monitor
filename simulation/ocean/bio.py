"""
Living-being models for the ocean simulator.

Five acoustically-active groups, each with presence driven by the simulated
environment (season, time of day, temperature, plankton bloom) and with
received level set by source level, range and transmission loss:

  snapping shrimp  continuous broadband crackle chorus (temperature/diel)
  fish chorus      harmonic grunt trains at dusk/night in the spawning season
  dolphins         pod arrivals (dawn/dusk-biased), FM whistles + click trains
  baleen whales    seasonal passage, long tonal/FM song units at low frequency
  (+ implicit)     the models' presence/range tracks are exported as ground truth

Spectral shapes follow published call characteristics at the level of
"typical band, duration, structure", not any single recording. The aim is that
each group occupies a physically distinct time-frequency region.
"""

import numpy as np
from scipy import signal

from .physics import db_to_upa, transmission_loss_db


# --------------------------------------------------------------------------
# presence / behaviour tracks (per window, whole-timeline)
# --------------------------------------------------------------------------
def _diel_gauss(hour, centre, width):
    d = (hour - centre + 12.0) % 24.0 - 12.0
    return np.exp(-0.5 * (d / width) ** 2)


def simulate_bio_state(env, site, rng):
    """Per-window biological state arrays aligned with the environment frame."""
    n = len(env)
    hour = env["hour"].values
    doy = env["doy"].values % 365
    temp = env["temperature_c"].values
    bloom = env["bloom"].values
    wind = env["wind_ms"].values

    night = 1.0 - env["irradiance"].values
    shrimp_rate = site.shrimp_density * 70.0 * np.exp(0.11 * (temp - 15.0)) * (0.75 + 0.5 * night)
    shrimp_rate *= (temp > 9.0) * (1.0 + 0.1 * rng.standard_normal(n))
    shrimp_rate = np.clip(shrimp_rate, 0, 600)

    spawn = 1.0 / (1.0 + np.exp(-(doy - 95.0) / 12.0)) * (1.0 / (1.0 + np.exp((doy - 270.0) / 12.0)))
    fish_level = site.fish_density * spawn * _diel_gauss(hour, 21.5, 3.2) * (temp > 11.5)
    fish_level *= np.clip(1.0 - (wind - 8.0) / 8.0, 0.2, 1.0)  # fish chorus less audible/active in rough weather
    fish_f0 = 90.0 + 5.0 * (temp - 15.0) + rng.normal(0, 3, n)
    fish_callers = np.maximum(1, np.round(fish_level * 22 + rng.normal(0, 1, n))).astype(int) * (fish_level > 0.05)

    # dolphins: two-state Markov chain, dawn/dusk-biased arrival hazard
    dol_present = np.zeros(n, dtype=bool)
    dol_range = np.full(n, np.nan)
    dol_n = np.zeros(n, dtype=int)
    p_stay = 0.8
    state, log_r, size = False, np.log(400.0), 3
    for i in range(n):
        hazard = site.dolphin_density * 0.03 * (0.4 + _diel_gauss(hour[i], 6.5, 2.5) + _diel_gauss(hour[i], 18.5, 2.5)) \
            * (0.6 + 0.6 * bloom[i]) * (1.0 if wind[i] < 12 else 0.4)
        if state:
            state = rng.random() < p_stay
            log_r += rng.normal(0, 0.35)
        else:
            if rng.random() < hazard:
                state, log_r, size = True, np.log(rng.uniform(60, 600)), int(rng.integers(1, 8))
        if state:
            dol_present[i] = True
            dol_range[i] = float(np.clip(np.exp(log_r), 25, 2500))
            dol_n[i] = size

    # whales: seasonal migration passage, multi-hour presence bouts
    whale_present = np.zeros(n, dtype=bool)
    whale_range = np.full(n, np.nan)
    state, remaining, log_r = False, 0, np.log(3000.0)
    for i in range(n):
        seasonal = np.exp(-0.5 * ((doy[i] - 85.0) / 55.0) ** 2) + 0.05
        hazard = site.whale_density * 0.0035 * seasonal
        if state:
            remaining -= 1
            log_r += rng.normal(0, 0.1)
            if remaining <= 0:
                state = False
        elif rng.random() < hazard:
            state, remaining, log_r = True, int(rng.integers(10, 55)), np.log(rng.uniform(1200, 7000))
        if state:
            whale_present[i] = True
            whale_range[i] = float(np.clip(np.exp(log_r), 500, 12000))

    return {
        "shrimp_rate": shrimp_rate, "fish_level": fish_level, "fish_f0": fish_f0, "fish_callers": fish_callers,
        "dolphin_present": dol_present, "dolphin_range": dol_range, "dolphin_n": dol_n,
        "whale_present": whale_present, "whale_range": whale_range,
    }


# --------------------------------------------------------------------------
# waveform synthesis (all return pressure in uPa)
# --------------------------------------------------------------------------
def _tl(f, r, ctx):
    return transmission_loss_db(
        f, r, ctx["temp"], ctx["sal"], ctx["ph"], ctx["hyd_depth"], ctx["water_depth"],
        spreading_k=ctx["k"], stratification=ctx["strat"],
    )


def render_shrimp(n, fs, rate, ctx, rng):
    out = np.zeros(n)
    dur = n / fs
    n_snaps = rng.poisson(rate * dur)
    if n_snaps == 0:
        return out
    r = np.sqrt(rng.uniform(5.0**2, 160.0**2, n_snaps))
    fc = rng.uniform(3500, 8500, n_snaps)
    amp_db = 160.0 + rng.normal(0, 4.0, n_snaps) - _tl(fc, r, ctx)  # dB re uPa peak
    amp = db_to_upa(amp_db)
    pos = rng.integers(0, n - 40, n_snaps)
    tk = np.arange(24) / fs
    for p, a, f in zip(pos, amp, fc):
        out[p:p + 24] += a * np.exp(-tk / 0.00018) * np.sin(2 * np.pi * f * tk)
    return out


def render_fish_chorus(n, fs, f0, callers, level, ctx, rng):
    out = np.zeros(n)
    dur = n / fs
    for _ in range(int(callers)):
        r = rng.uniform(15, 500)
        f_i = f0 * rng.uniform(0.85, 1.2)
        for _ in range(rng.poisson(1.6)):
            cdur = rng.uniform(0.12, 0.45)
            t0 = rng.uniform(0, max(dur - cdur, 1e-3))
            m = int(cdur * fs)
            t = np.arange(m) / fs
            env = np.hanning(m) ** 0.7
            amod = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(35, 70) * t)  # pulsatile grunt
            w = sum((0.8 ** k) * np.sin(2 * np.pi * f_i * (k + 1) * t * (1 + 0.04 * t / cdur) + rng.uniform(0, 6.28))
                    for k in range(4))
            sl = 138.0 + rng.normal(0, 3.0)
            amp = db_to_upa(sl - _tl(f_i * 2, r, ctx))
            i0 = int(t0 * fs)
            out[i0:i0 + m] += amp * env * amod * w / 2.2
    return out


def render_dolphins(n, fs, size, rng_range, ctx, rng):
    out = np.zeros(n)
    dur = n / fs
    for _ in range(int(size)):
        r = max(rng_range + rng.normal(0, 40), 20.0)
        for _ in range(rng.poisson(0.7)):                      # whistles
            wd = rng.uniform(0.4, 1.2)
            t0 = rng.uniform(0, max(dur - wd, 1e-3))
            m = int(wd * fs)
            t = np.arange(m) / fs
            f_start = rng.uniform(3500, 7500)
            contour = f_start + rng.uniform(-1500, 3000) * t / wd + rng.uniform(200, 900) * np.sin(2 * np.pi * rng.uniform(1.5, 5) * t)
            contour = np.clip(contour, 2500, 10200)
            phase = 2 * np.pi * np.cumsum(contour) / fs
            w = np.sin(phase)
            if contour.max() < 5200:
                w = w + 0.4 * np.sin(2 * phase)
            amp = db_to_upa(160.0 + rng.normal(0, 3) - _tl(contour.mean(), r, ctx))
            i0 = int(t0 * fs)
            out[i0:i0 + m] += amp * np.hanning(m) * w
        if rng.random() < 0.55:                                 # click train (low part of the click spectrum)
            n_cl = int(rng.integers(8, 40))
            ici = rng.uniform(0.02, 0.08)
            t0 = rng.uniform(0, max(dur - n_cl * ici, 1e-3))
            amp = db_to_upa(176.0 + rng.normal(0, 4) - _tl(7000.0, r, ctx))
            for j in range(n_cl):
                p = int((t0 + j * ici) * fs)
                if p + 30 >= n:
                    break
                k = np.arange(30) / fs
                out[p:p + 30] += amp * np.exp(-k / 0.0001) * np.sin(2 * np.pi * rng.uniform(5000, 9500) * k)
    return out


def render_whale(n, fs, rng_range, ctx, rng):
    out = np.zeros(n)
    dur = n / fs
    t_cursor = rng.uniform(0, 1.0)
    base = rng.uniform(110, 260)
    while t_cursor < dur - 0.3:
        ud = rng.uniform(0.8, 2.6)
        m = int(min(ud, dur - t_cursor) * fs)
        if m < int(0.2 * fs):
            break
        t = np.arange(m) / fs
        sweep = rng.uniform(-0.5, 1.4)
        f = base * (1 + sweep * t / ud) + rng.uniform(5, 25) * np.sin(2 * np.pi * rng.uniform(0.5, 3) * t)
        phase = 2 * np.pi * np.cumsum(f) / fs
        noisy = rng.random() < 0.25
        w = sum((0.7 ** k) * np.sin((k + 1) * phase) for k in range(5))
        if noisy:
            w = w * (0.5 + 0.5 * rng.standard_normal(m) * 0.5)
        env = np.hanning(m) ** 0.5
        amp = db_to_upa(166.0 + rng.normal(0, 2.5) - _tl(f.mean() * 1.5, rng_range, ctx))
        i0 = int(t_cursor * fs)
        out[i0:i0 + m] += amp * env * w / 2.5
        t_cursor += ud + rng.uniform(0.1, 0.9)
        base *= rng.uniform(0.8, 1.25)
    return out
