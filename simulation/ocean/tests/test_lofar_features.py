"""Tests for the narrowband comb estimator and the causal sensor-health features."""

import numpy as np
import pandas as pd

from simulation.ocean.dataset import _flat_run, add_temporal
from simulation.ocean.lofar import FS, comb_features, lofar_whitened


def _comb_signal(blade_rate, n_harm=8, snr_db=10.0, seed=0, secs=5.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * secs)) / FS
    x = rng.normal(0, 1.0, len(t))
    amp = 10 ** (snr_db / 20) * 0.02
    for h in range(1, n_harm + 1):
        x += amp * np.sin(2 * np.pi * blade_rate * h * t + rng.uniform(0, 6.28))
    return x


def test_comb_recovers_blade_rate():
    for f0 in (5.0, 7.5, 32.0, 120.0):
        est = comb_features(lofar_whitened(_comb_signal(f0, snr_db=14.0)))
        assert abs(est["comb_f0_hz"] - f0) < 0.8, (f0, est)
        assert est["comb_strength"] > 0.2


def test_comb_absent_in_pure_noise():
    est = comb_features(lofar_whitened(np.random.default_rng(1).normal(0, 1, FS * 5)))
    assert est["comb_strength"] < 0.35 and est["lofar_tones"] <= 3


def test_flat_run_counts_identical_readings():
    x = np.array([1.0, 1.1, 1.1, 1.1, 1.1, 1.2, 1.3, 1.3])
    assert _flat_run(x).tolist() == [0, 0, 1, 2, 3, 0, 0, 1]
    assert _flat_run(np.ones(30), cap=12).max() == 12


def test_stuck_sensor_visible_only_with_temporal_context():
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({"ph": 8.0 + rng.normal(0, 0.01, n), "turbidity_ntu": 2 + rng.normal(0, 0.1, n)})
    df.loc[10:20, "ph"] = df.loc[10, "ph"]
    out = add_temporal(df)
    assert out["ph_flat_run"].iloc[20] == 10 and out["ph_flat_run"].iloc[9] == 0
    assert out["ph_std6_log"].iloc[20] < out["ph_std6_log"].iloc[5] - 1.0
