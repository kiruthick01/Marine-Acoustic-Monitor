"""
Scenario runner: environment + living beings + vessels + airguns + faults ->
hydrophone audio + environmental sensors -> pipeline features + labels.

One run = one site, N days, duty-cycle windows every `interval_min`. Windows are
rendered independently (deterministic per (seed, window)) and in parallel.
Outputs (under out_dir):
  windows.pkl   per-window table: truth drivers, sensor readings, labels, features
  mel.npy       float16 log-mel spectrograms (n_windows, 64, frames) for the CNN
  events.json   fronts, vessel passages, airgun campaigns, faults, bio tracks
  audio/        a handful of example wav files per event class
"""

import json
import os
import time
from dataclasses import asdict

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal
from scipy.io import wavfile

from simulation.pipeline.feature_extraction import extract_acoustic_features
from simulation.pipeline.signal_conditioning import condition_signal

from . import anthro, bio
from .physics import shape_noise, wenz_psd_db
from .weather import SITE_A, Site, simulate_environment

FS = 22050
WINDOW_S = 5.0
N_SAMP = int(FS * WINDOW_S)
SENS_DB = -180.0            # hydrophone sensitivity, dB re 1 V/uPa
PREAMP_DB = 30.0            # preamp gain
SELF_NOISE_PSD_DB = 28.0    # electronics + thermal self-noise (dB re uPa^2/Hz, input referred)
N_MELS = 64
HOP = 1024
BANDS = [(20, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 11000)]
SNR_BANDS = {"vessel": (50, 1000), "dolphin": (3000, 10500), "whale": (80, 1500),
             "fish": (60, 800), "airgun": (10, 220)}
SNR_THRESH_DB = {"vessel": 10.0, "dolphin": 10.0, "whale": 10.0, "fish": 10.0, "airgun": 10.0}


def _band_power(x, f1, f2):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    return float(spec[(f >= f1) & (f < f2)].sum()) + 1e-30


def _bg_profile(background):
    f, _, sb = signal.spectrogram(background, FS, nperseg=2048, noverlap=1024)
    return f, sb.mean(axis=1) + 1e-30


def _snr_db(comp, bg_profile, band):
    """
    Peak time-frequency detectability: 99th-percentile (over spectrogram bins in
    `band`) of component power / mean background power in that frequency bin.
    Band-power ratios badly understate short or narrowband sources (whistles,
    tonals, pulses) that are plainly visible on a spectrogram.
    """
    f, sbm = bg_profile
    _, _, sc = signal.spectrogram(comp, FS, nperseg=2048, noverlap=1024)
    m = (f >= band[0]) & (f < band[1])
    ratio = sc[m] / sbm[m][:, None]
    return 10 * np.log10(np.percentile(ratio, 99) + 1e-30)


def counts_from_upa(p_upa, gain_db_drift=0.0):
    """uPa -> hydrophone chain -> 16-bit ADC full-scale-normalised samples."""
    v = p_upa * 10 ** ((SENS_DB + PREAMP_DB + gain_db_drift) / 20.0)
    v = np.clip(v, -1.0, 1.0)
    return (np.round(v * 32767.0) / 32767.0).astype(np.float32)


def acoustic_extras(a):
    """Calibrated (non-denoised) descriptors that the Stage-0 denoiser would erase."""
    out = {}
    spec = np.abs(np.fft.rfft(a.astype(np.float64))) ** 2 / len(a) ** 2
    f = np.fft.rfftfreq(len(a), 1.0 / FS)
    for lo, hi in BANDS:
        out[f"band_{lo}_{hi}_db"] = 10 * np.log10(spec[(f >= lo) & (f < hi)].sum() * 2 + 1e-14)
    low = (f > 20) & (f < 400)
    ps = spec[low]
    out["lf_peak_prominence_db"] = 10 * np.log10(ps.max() / (np.median(ps) + 1e-20) + 1e-9)
    out["kurtosis"] = float(((a - a.mean()) ** 4).mean() / (a.var() ** 2 + 1e-20))
    out["crest_factor"] = float(np.abs(a).max() / (a.std() + 1e-9))
    hum = 0.0
    for hz in (50.0, 60.0):
        for h in (1, 2, 3):
            k = (f > hz * h - 1.0) & (f < hz * h + 1.0)
            hum = max(hum, spec[k].sum() / (np.median(spec[(f > hz * h - 30) & (f < hz * h + 30)]) * k.sum() + 1e-20))
    out["hum_ratio_db"] = 10 * np.log10(hum + 1e-6)
    out["clip_fraction"] = float((np.abs(a) > 0.999).mean())
    out["zero_fraction"] = float((np.abs(a) < 1e-6).mean())
    sl = np.polyfit(np.log10(f[(f > 30) & (f < 10000)]), 10 * np.log10(spec[(f > 30) & (f < 10000)] + 1e-20), 1)[0]
    out["spectral_slope"] = float(sl)
    return out


def synthesize_window(i, sh, want_components=False):
    """Render one window. `sh` is the shared scenario description dict."""
    site: Site = sh["site"]
    env = sh["env"][i]
    st = {k: v[i] for k, v in sh["bio"].items()}
    rng = np.random.default_rng([sh["seed"], i, 7])
    t0 = i * sh["interval_min"] * 60.0
    ctx = dict(temp=env["temperature_c"], sal=env["salinity_psu"], ph=env["ph"], strat=env["stratification"],
               c=env["sound_speed_ms"], k=site.spreading_k, hyd_depth=site.hydrophone_depth_m,
               water_depth=site.water_depth_m + env["tide_m"])

    # ---- ambient: wind + shipping + rain + thermal + flow, then electronics self-noise
    psd = lambda f: wenz_psd_db(f, env["wind_ms"], shipping=site.shipping_index, rain_mm_h=env["rain_mm_h"],  # noqa: E731
                                flow_ms=env["flow_ms"])
    ambient = shape_noise(psd, None, N_SAMP, FS, rng)
    selfn = shape_noise(lambda f: np.full_like(f, SELF_NOISE_PSD_DB, dtype=float), None, N_SAMP, FS, rng)
    shrimp = bio.render_shrimp(N_SAMP, FS, st["shrimp_rate"], ctx, rng) if st["shrimp_rate"] > 1 else np.zeros(N_SAMP)
    background = ambient + selfn + shrimp

    bgp = _bg_profile(background)
    comps = {}
    lab = {}
    if st["fish_callers"] > 0 and st["fish_level"] > 0.05:
        comps["fish"] = bio.render_fish_chorus(N_SAMP, FS, st["fish_f0"], st["fish_callers"], st["fish_level"], ctx, rng)
    if st["dolphin_present"]:
        comps["dolphin"] = bio.render_dolphins(N_SAMP, FS, st["dolphin_n"], st["dolphin_range"], ctx, rng)
    if st["whale_present"]:
        comps["whale"] = bio.render_whale(N_SAMP, FS, st["whale_range"], ctx, rng)

    # ---- vessels (loudest audible one defines the vessel label)
    v_best = dict(snr=-99.0, cls=None, range=np.nan, speed=np.nan)
    v_sum = np.zeros(N_SAMP)
    n_vessel_active = 0
    for j in sh["vessel_candidates"][i]:
        p = sh["passages"][j]
        sig, r_mid = anthro.render_vessel(p, t0, N_SAMP, FS, ctx, rng)
        if sig is None:
            continue
        n_vessel_active += 1
        v_sum += sig
        snr = _snr_db(sig, bgp, SNR_BANDS["vessel"])
        if snr > v_best["snr"]:
            v_best = dict(snr=snr, cls=p.cls, range=r_mid, speed=p.speed_ms)
    if n_vessel_active:
        comps["vessel"] = v_sum

    # ---- airgun campaign
    ag_range = None
    for (s, e, rng_m) in sh["airgun"]:
        if s <= i < e and rng.random() < 0.55:
            comps["airgun"] = anthro.render_airgun(N_SAMP, FS, ctx, rng, rng_m)
            ag_range = rng_m

    total = background.copy()
    for k, c in comps.items():
        total = total + c

    # ---- faults acting on the audio path
    fault_audio = None
    t = np.arange(N_SAMP) / FS
    for f in sh["faults"]:
        if f["start"] <= i < f["end"]:
            if f["type"] == "hum":
                hz = f["hz"]
                hum = sum((0.5 ** h) * np.sin(2 * np.pi * hz * (h + 1) * t + rng.uniform(0, 6.28)) for h in range(5))
                total = total + hum * f["level"] / 10 ** ((SENS_DB + PREAMP_DB) / 20.0)
                fault_audio = "hum"
            elif f["type"] == "dropout":
                total = total * 0.0 + selfn * 0.05
                fault_audio = "dropout"
            elif f["type"] == "pops":
                for _ in range(int(rng.integers(2, 9))):
                    p0 = int(rng.integers(0, N_SAMP - 800))
                    kk = np.arange(800) / FS
                    total[p0:p0 + 800] += rng.choice([-1, 1]) * 4e7 * np.exp(-kk / 0.004) * np.sin(2 * np.pi * rng.uniform(80, 700) * kk)
                fault_audio = "pops"

    # ---- hydrophone chain (sensitivity slowly degrades with biofouling)
    drift_db = -0.03 * (t0 / 86400.0)
    audio = counts_from_upa(total, drift_db)

    # ---- labels from per-source SNR against the (ambient+electronics+shrimp) background
    snrs = {k: _snr_db(c, bgp, SNR_BANDS[k]) for k, c in comps.items()}
    for k in SNR_BANDS:
        lab[f"snr_{k}_db"] = snrs.get(k, np.nan)
        lab[f"is_{k}"] = bool(k in snrs and snrs[k] > SNR_THRESH_DB[k] and fault_audio != "dropout")
    lab["vessel_class"] = v_best["cls"] if lab["is_vessel"] else None
    lab["vessel_range_m"] = v_best["range"] if v_best["cls"] else np.nan
    lab["n_vessels_active"] = n_vessel_active
    lab["airgun_range_m"] = ag_range if ag_range else np.nan
    lab["is_shrimp"] = bool(st["shrimp_rate"] > 20)
    lab["fault_audio"] = fault_audio

    # ---- environmental sensor model (noise, drift, faults)
    es = sh["sensors"][i]
    # ---- features
    cond, diag = condition_signal(audio, FS)
    feats = extract_acoustic_features(cond, FS)
    feats["snr_after_db"] = diag["snr_after_db"]
    feats.update(acoustic_extras(audio))
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=audio.astype(np.float32), sr=FS, n_fft=2048, hop_length=HOP,
                                       n_mels=N_MELS, fmin=20, fmax=11000, power=2.0),
        ref=1.0, amin=1e-12, top_db=None,
    ).astype(np.float16)

    row = {"window_index": i, **lab, **{f"f_{k}": v for k, v in feats.items()},
           "bio_dolphin_n": st["dolphin_n"], "bio_dolphin_range_m": st["dolphin_range"],
           "bio_whale_range_m": st["whale_range"], "bio_shrimp_rate": st["shrimp_rate"],
           "bio_fish_level": st["fish_level"], "bio_fish_callers": st["fish_callers"]}
    if want_components:
        return row, mel, audio, comps, background
    return row, mel


def _render_chunk(idx, sh):
    out = [synthesize_window(i, sh) for i in idx]
    return out


def _sensor_model(env, faults, rng):
    n = len(env)
    day = env["elapsed_minutes"].values / 1440.0
    cols = {}
    t = env["temperature_c"].values + rng.normal(0, 0.02, n)
    ph = env["ph"].values + rng.normal(0, 0.008, n) - 0.0015 * day            # biofouling pH drift
    tu = env["turbidity_ntu"].values * (1 + rng.normal(0, 0.04, n)) + rng.normal(0, 0.08, n) + 0.05 * day
    sa = env["salinity_psu"].values + rng.normal(0, 0.03, n)
    stuck = np.zeros(n, dtype=bool)
    for f in faults:
        if f["type"] == "env_stuck":
            s, e = f["start"], f["end"]
            tu[s:e], ph[s:e] = tu[s], ph[s]
            stuck[s:e] = True
    spikes = rng.random(n) < 0.003
    tu = tu + spikes * rng.uniform(3, 10, n)
    out = pd.DataFrame({"temperature_c": t, "ph": ph, "turbidity_ntu": np.clip(tu, 0.05, None), "salinity_psu": sa})
    out["temp_roc"] = out["temperature_c"].diff().fillna(0.0)
    out["ph_roc"] = out["ph"].diff().fillna(0.0)
    out["turbidity_roc"] = out["turbidity_ntu"].diff().fillna(0.0)
    out["salinity_roc"] = out["salinity_psu"].diff().fillna(0.0)
    out["fault_env"] = stuck
    return out


def build_scenario(site=SITE_A, n_days=30, seed=0, start_doy=90, interval_min=10.0, airgun_prob=0.5, fault_prob=0.4):
    n = int(n_days * 1440 / interval_min)
    rng = np.random.default_rng([seed, 99])
    env = simulate_environment(n, site=site, interval_min=interval_min, start_doy=start_doy, seed=seed)
    fronts = env.attrs["fronts"]
    biost = bio.simulate_bio_state(env, site, rng)
    passages = anthro.schedule_vessels(n, interval_min, site, rng)
    airgun = anthro.schedule_airgun_campaigns(n, interval_min, rng, prob=airgun_prob)
    faults = anthro.schedule_faults(n, rng, per_type_prob=fault_prob)
    sensors = _sensor_model(env, faults, rng)

    # candidate vessels per window: those whose CPA time is within reach
    t_cpa = np.array([p.t_cpa_s for p in passages]) if passages else np.zeros(0)
    speed = np.array([p.speed_ms for p in passages]) if passages else np.zeros(0)
    reach = 14000.0 / np.maximum(speed, 1.0) + 10.0
    cand = []
    for i in range(n):
        t = i * interval_min * 60.0
        cand.append(list(np.nonzero(np.abs(t - t_cpa) < reach)[0]) if len(t_cpa) else [])

    sh = dict(
        site=site, seed=seed, interval_min=interval_min, passages=passages, airgun=airgun, faults=faults,
        env=env.to_dict("records"), bio=biost, sensors=sensors.to_dict("records"), vessel_candidates=cand,
    )
    meta = dict(fronts=fronts, airgun=airgun, faults=faults,
                passages=[{**asdict(p), "tonal_offsets": None} for p in passages], site=asdict(site),
                seed=seed, n_windows=n, interval_min=interval_min, start_doy=start_doy)
    return sh, env, sensors, meta


def run_scenario(site=SITE_A, n_days=30, seed=0, start_doy=90, interval_min=10.0, out_dir=None, n_jobs=-1,
                 save_examples=True, verbose=True, airgun_prob=0.5, fault_prob=0.4):
    t0 = time.time()
    sh, env, sensors, meta = build_scenario(site, n_days, seed, start_doy, interval_min, airgun_prob, fault_prob)
    n = meta["n_windows"]
    step = max(int(np.ceil(n / 32)), 8)
    chunks = [list(range(s, min(s + step, n))) for s in range(0, n, step)]
    res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_render_chunk)(c, sh) for c in chunks)
    rows, mels = [], []
    for chunk in res:
        for r, m in chunk:
            rows.append(r)
            mels.append(m)
    lab = pd.DataFrame(rows).set_index("window_index", drop=False)
    df = pd.concat([env.set_index("window_index", drop=False), lab.drop(columns=["window_index"], errors="ignore")], axis=1)
    # observed sensors replace truth under the same names; keep truth with _true suffix
    for c in ["temperature_c", "ph", "turbidity_ntu", "salinity_psu"]:
        df[f"{c}_true"] = df[c]
    obs = sensors.set_index(df.index)
    for c in obs.columns:
        df[c] = obs[c].values if c in df.columns else obs[c].values
    df["storm"] = (df["wind_ms"] > 10.5) | (df["rain_mm_h"] > 2.0) | (df["runoff_index"] > 0.5)
    df["fault_any"] = df["fault_audio"].notna() | df["fault_env"]
    df["anomaly"] = df["is_vessel"] | df["is_airgun"] | df["storm"] | df["fault_any"]
    df["site"] = site.name
    df["seed"] = seed
    mel = np.stack(mels)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        df.to_pickle(os.path.join(out_dir, "windows.pkl"))
        np.save(os.path.join(out_dir, "mel.npy"), mel)
        with open(os.path.join(out_dir, "events.json"), "w") as f:
            json.dump(meta, f, default=float)
        if save_examples:
            _save_examples(df, sh, os.path.join(out_dir, "audio"))
    if verbose:
        print(f"[{site.name} seed={seed}] {n} windows in {time.time() - t0:.1f}s | "
              f"anomaly {df['anomaly'].mean():.1%} vessel {df['is_vessel'].mean():.1%} storm {df['storm'].mean():.1%} "
              f"dolphin {df['is_dolphin'].mean():.1%} whale {df['is_whale'].mean():.1%} fish {df['is_fish'].mean():.1%} "
              f"airgun {df['is_airgun'].mean():.1%} fault {df['fault_any'].mean():.1%}")
    return df, mel, meta


def _save_examples(df, sh, audio_dir, per_class=2):
    os.makedirs(audio_dir, exist_ok=True)
    picks = {}
    for name, mask in {
        "vessel": df["is_vessel"], "dolphin": df["is_dolphin"], "whale": df["is_whale"], "fish": df["is_fish"],
        "airgun": df["is_airgun"], "storm": df["storm"] & ~df["is_vessel"], "hum": df["fault_audio"] == "hum",
        "pops": df["fault_audio"] == "pops", "calm": ~df["anomaly"] & ~df["is_dolphin"] & ~df["is_whale"] & ~df["is_fish"],
    }.items():
        idx = df.index[mask].tolist()
        for j in idx[:: max(len(idx) // per_class, 1)][:per_class]:
            picks[f"{name}_{j}"] = j
    for name, j in picks.items():
        _, _, audio, _, _ = synthesize_window(j, sh, want_components=True)
        wavfile.write(os.path.join(audio_dir, f"{name}.wav"), FS, audio)
