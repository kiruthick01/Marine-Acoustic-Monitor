"""
Day-block bootstrap confidence intervals for the headline mAP numbers (windows within a day are autocorrelated, so whole
days are resampled). Reads classification_preds.npz written by ocean_classify.py.

  python simulation/scripts/ocean_bootstrap.py
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ocean.dataset import LABELS, OUT_DIR  # noqa: E402

DAY = 144
N_BOOT = 400
MODELS = ["GB-v1-Aonly", "GB-v2-multi", "CNN-Aonly-v1", "CNN-multi-v1", "CNN-multi-v2"]


def map_of(Y, P):
    v = [average_precision_score(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
    return float(np.mean(v)) if v else float("nan")


def main():
    Z = np.load(os.path.join(OUT_DIR, "classification_preds.npz"))
    tests = [k[2:] for k in Z.files if k.startswith("Y/")]
    rng = np.random.default_rng(0)
    out = {}
    for t in tests:
        Y = Z[f"Y/{t}"]
        n_days = int(np.ceil(len(Y) / DAY))
        days = [np.arange(d * DAY, min((d + 1) * DAY, len(Y))) for d in range(n_days)]
        draws = [np.concatenate([days[i] for i in rng.integers(0, n_days, n_days)]) for _ in range(N_BOOT)]
        out[t] = {}
        for m in MODELS:
            P = Z[f"{m}/{t}"]
            vals = np.array([map_of(Y[ix], P[ix]) for ix in draws])
            out[t][m] = {"mAP": map_of(Y, P), "ci95": [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]}
        # paired difference v2-multi minus v1-Aonly (original design)
        d = np.array([map_of(Y[ix], Z[f"CNN-multi-v2/{t}"][ix]) - map_of(Y[ix], Z[f"CNN-Aonly-v1/{t}"][ix]) for ix in draws])
        out[t]["paired_gain_multi-v2_minus_Aonly-v1"] = {"mean": float(np.nanmean(d)),
                                                         "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
        print(t, {m: f"{out[t][m]['mAP']:.3f} [{out[t][m]['ci95'][0]:.3f},{out[t][m]['ci95'][1]:.3f}]" for m in MODELS}, flush=True)
    with open(os.path.join(OUT_DIR, "bootstrap_results.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
