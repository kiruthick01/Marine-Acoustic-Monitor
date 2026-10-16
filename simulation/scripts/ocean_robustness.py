"""
Robustness of the main classifier (multi-site, v2 inputs) to deployment faults it was not trained against explicitly:

  gain       hydrophone sensitivity error, +-d dB applied to the log-mel (calibration drift beyond the +-3 dB augmentation)
  mel noise  extra per-bin spectral noise (2 / 4 dB std)
  env outage environmental-sensor channels replaced by the training median (sensor / I2C failure)
  no LOFAR   narrowband branch zeroed (e.g. disabled to save CPU)
  no health  sensor-health context features neutralised

Evaluated on the same test slices as ocean_classify.py (A weather, B spring tail, E winter tail).

  python simulation/scripts/ocean_robustness.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import ocean_classify as oc  # noqa: E402
import ocean_eval_extra as ee  # noqa: E402
from simulation.ocean.dataset import ENV_RAW, LABELS, OUT_DIR, TEMPORAL  # noqa: E402
from simulation.ocean.models import Data, predict  # noqa: E402


def perturbed(v, sl, kind, level, rng):
    d = v.data(sl)
    X, V, L = d.X.clone(), d.V.clone(), None if d.L is None else d.L.clone()
    cols = v.cols
    if kind == "gain":
        X = (X.float() + level / v.norm.sd.mean()).half()
    elif kind == "mel_noise":
        X = (X.float() + torch.tensor(rng.normal(0, level, X.shape), dtype=torch.float32) / float(v.norm.sd.mean())).half()
    elif kind == "env_outage":
        idx = [cols.index(c) for c in ENV_RAW]
        V[:, idx] = 0.0
    elif kind == "no_lofar":
        L = torch.zeros_like(L)
    elif kind == "no_health":
        idx = [cols.index(c) for c in TEMPORAL]
        V[:, idx] = 0.0
    return Data(X, V, L, d.Y, d.C)


def main():
    v = ee.load_variant("multi-v2")
    names = {"A weather": ("harbor_shelf_s2", 0.0), "B spring": ("open_coast_s10", 0.25), "E winter": ("offshore_shelf_s40", 0.25)}
    slices = {k: oc.Run(n).cut(a, 1.0) for k, (n, a) in names.items()}
    rng = np.random.default_rng(0)
    cases = [("none", 0)] + [("gain", g) for g in (-10, -6, -3, 3, 6, 10)] + [("mel_noise", 2), ("mel_noise", 4),
                                                                              ("env_outage", 0), ("no_lofar", 0), ("no_health", 0)]
    out = {}
    for kind, lvl in cases:
        key = f"{kind}{'' if kind in ('none', 'env_outage', 'no_lofar', 'no_health') else f'_{lvl}'}"
        out[key] = {}
        for k, sl in slices.items():
            P, VC = predict(v.net, perturbed(v, sl, kind, lvl, rng))
            sc = oc.eval_scores(sl, P, VC, np.full(len(LABELS), 0.5))
            out[key][k] = {"mAP": sc["mAP"], "ap": sc["ap"], "vessel_type_acc": sc.get("vessel_type_acc", float("nan"))}
        print(f"{key:14s} " + " ".join(f"{k}: {out[key][k]['mAP']:.3f}" for k in slices), flush=True)
    with open(os.path.join(OUT_DIR, "robustness_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
