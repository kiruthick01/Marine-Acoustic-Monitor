"""
Edge-feasibility benchmark: per-window feature + inference cost on one CPU thread.

The deployment target is a Raspberry Pi duty-cycling one 5 s window every 10 min, so the budget is generous
(minutes), but the numbers show how much headroom there is. Pi figures are NOT measured here; the script reports
host timings and the model size, and the README states the Pi estimate as an assumption.

  python simulation/scripts/ocean_edge_benchmark.py
"""

import json
import os
import sys
import time

import librosa
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ocean.dataset import COMB, ENV_RAW, CTX, LABELS, OUT_DIR, TEMPORAL, VESSEL_CLASSES  # noqa: E402
from simulation.ocean.lofar import comb_features, lofar_whitened  # noqa: E402
from simulation.ocean.models import Net  # noqa: E402
from simulation.ocean.scenario import FS, HOP, N_MELS, N_SAMP, acoustic_extras  # noqa: E402
from simulation.pipeline.feature_extraction import extract_acoustic_features  # noqa: E402
from simulation.pipeline.signal_conditioning import condition_signal  # noqa: E402

torch.set_num_threads(1)


def timeit(fn, n=15):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000.0), float(np.percentile(ts, 90) * 1000.0)


def main():
    rng = np.random.default_rng(0)
    audio = (rng.normal(0, 0.02, N_SAMP)).astype(np.float32)
    out = {}
    out["condition_signal_ms"] = timeit(lambda: condition_signal(audio, FS))
    cond, _ = condition_signal(audio, FS)
    out["classic_features_ms"] = timeit(lambda: extract_acoustic_features(cond, FS))
    out["calibrated_extras_ms"] = timeit(lambda: acoustic_extras(audio))
    out["lofar_comb_ms"] = timeit(lambda: comb_features(lofar_whitened(audio)))
    out["logmel_ms"] = timeit(lambda: librosa.power_to_db(librosa.feature.melspectrogram(
        y=audio, sr=FS, n_fft=2048, hop_length=HOP, n_mels=N_MELS, fmin=20, fmax=11000, power=2.0)))
    n_vec = len(ENV_RAW + CTX + TEMPORAL + COMB)
    n_lofar = int((600 - 8) / (FS / 32768))
    for width in (24, 16, 12):
        net = Net(n_vec, len(LABELS), len(VESSEL_CLASSES), width, use_lofar=True).eval()
        x, v, l = torch.randn(1, 1, 64, 108), torch.randn(1, n_vec), torch.randn(1, 1, n_lofar)
        with torch.no_grad():
            out[f"cnn_w{width}_forward_ms"] = timeit(lambda: net(x, v, l))
        out[f"cnn_w{width}_params"] = sum(p.numel() for p in net.parameters())
        out[f"cnn_w{width}_size_mb_fp32"] = sum(p.numel() for p in net.parameters()) * 4 / 1e6
    total = (out["condition_signal_ms"][0] + out["classic_features_ms"][0] + out["calibrated_extras_ms"][0]
             + out["lofar_comb_ms"][0] + out["logmel_ms"][0] + out["cnn_w24_forward_ms"][0])
    out["total_per_window_ms_host_1thread"] = total
    out["duty_cycle_window_s"] = 600.0
    out["fraction_of_duty_cycle_at_10x_slower_cpu"] = total * 10 / 1000.0 / 600.0
    with open(os.path.join(OUT_DIR, "edge_benchmark.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(f"{k:45s} {v}")


if __name__ == "__main__":
    main()
