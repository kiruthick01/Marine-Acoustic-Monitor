"""Loading + feature-set definitions shared by the detection / classification scripts."""

import glob
import os

import numpy as np
import pandas as pd

from simulation.pipeline.feature_extraction import extract_environmental_features

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "ocean")

ACO_OLD = (
    [f"f_mfcc_{i}_{s}" for i in range(1, 14) for s in ("mean", "std")]
    + [f"f_{n}_{s}" for n in ("spectral_centroid", "zero_crossing_rate", "rms_energy", "spectral_flatness") for s in ("mean", "std")]
)
BAND_COLS = [f"f_band_{lo}_{hi}_db" for lo, hi in
             [(20, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 11000)]]
EXTRA_COLS = ["f_lf_peak_prominence_db", "f_kurtosis", "f_crest_factor", "f_hum_ratio_db", "f_clip_fraction",
              "f_zero_fraction", "f_spectral_slope", "f_snr_after_db"]
ACO_EXT = ACO_OLD + BAND_COLS + EXTRA_COLS   # (detection scripts: unchanged feature set)
ENV_RAW = ["temperature_c", "ph", "turbidity_ntu", "salinity_psu", "temp_roc", "ph_roc", "turbidity_roc", "salinity_roc"]
# causal sensor-health context: consecutive windows with a bit-identical reading (stuck sensor) and short-term variability
TEMPORAL = ["ph_flat_run", "turb_flat_run", "env_flat_run", "ph_std6_log", "turb_std6_log"]
COMB = ["f_comb_f0_hz", "f_comb_strength", "f_lofar_tones"]
CTX = ["hour_sin", "hour_cos"]

LABELS = ["is_vessel", "is_dolphin", "is_whale", "is_fish", "is_airgun", "storm", "fault_any"]
VESSEL_CLASSES = ["cargo", "tanker", "ferry", "fishing", "outboard"]


def _flat_run(x, cap=12):
    """Number of consecutive windows (ending at each one) whose reading is identical to the previous window's."""
    flat = np.concatenate([[False], np.diff(x) == 0.0])
    out = np.zeros(len(x))
    run = 0
    for i, f in enumerate(flat):
        run = min(run + 1, cap) if f else 0
        out[i] = run
    return out


def add_temporal(df):
    df = df.copy()
    df["ph_flat_run"] = _flat_run(df["ph"].values)
    df["turb_flat_run"] = _flat_run(df["turbidity_ntu"].values)
    df["env_flat_run"] = np.maximum(df["ph_flat_run"], df["turb_flat_run"])
    for c, name in (("ph", "ph_std6_log"), ("turbidity_ntu", "turb_std6_log")):
        df[name] = np.log10(df[c].rolling(6, min_periods=2).std().fillna(df[c].std()) + 1e-6)
    return df


def load_lofar(name):
    return np.load(os.path.join(OUT_DIR, name, "lofar.npy"))


def load_run(name):
    d = os.path.join(OUT_DIR, name)
    df = pd.read_pickle(os.path.join(d, "windows.pkl"))
    mel = np.load(os.path.join(d, "mel.npy"))
    df = add_temporal(df)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    env_old = pd.DataFrame([extract_environmental_features(r) for _, r in df.iterrows()], index=df.index)
    for c in env_old.columns:
        df[f"old_{c}"] = env_old[c]
    df["run"] = name
    for c in LABELS:
        df[c] = df[c].astype(bool)
    return df, mel


def old_env_cols(df):
    return [c for c in df.columns if c.startswith("old_")]


def list_runs():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT_DIR, "*_s*"))
                  if os.path.isdir(p) and not os.path.basename(p).startswith("rich_"))
