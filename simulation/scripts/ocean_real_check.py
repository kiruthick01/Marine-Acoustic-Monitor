"""
Sanity check of the simulator-trained classifier on REAL ship-noise recordings (ShipsEar sample set, 4 files).

This is NOT a validation of real-world accuracy: four recordings, uncalibrated (no uPa reference), unknown hydrophone.
It checks three things the simulator could have got badly wrong:
  1. do real vessel passages light up the `vessel` output and real rain/ambient does not,
  2. does the narrowband comb estimator find a plausible shaft/blade periodicity in real ship noise,
  3. is the real spectral shape inside the simulated envelope (no gross simulation gap).

Level handling: recordings are in arbitrary digital units, so ONE global gain is applied to all files, chosen so the
quietest real recording (rain/ambient) has its 10th-percentile mel level equal to the simulator's 10th percentile
(noise-floor alignment). Environment / clock inputs are set to the training median (neutral).

  python simulation/scripts/ocean_real_check.py --data <dir with the extracted ShipsEar sample wav files>
"""

import argparse
import glob
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ocean.dataset import LABELS, OUT_DIR, VESSEL_CLASSES  # noqa: E402
from simulation.ocean.lofar import comb_features, lofar_whitened  # noqa: E402
from simulation.ocean.models import Net, Norm, predict  # noqa: E402
from simulation.ocean.scenario import FS, HOP, N_MELS, N_SAMP  # noqa: E402

META = {  # from the ShipsEar file names / class table (class A fishing..E ambient)
    "adventure_maniobra": "passenger / ro-ro vessel manoeuvring",
    "marDeCangas_Entra": "passenger ferry arriving",
    "H3_zodiac": "small motorboat (zodiac)",
    "H3_lluvia": "rain, no vessel (ambient)",
}


def windows(path):
    y, sr = sf.read(path, dtype="float32")
    y = librosa.resample(y, orig_sr=sr, target_sr=FS)
    n = len(y) // N_SAMP
    return y[: n * N_SAMP].reshape(n, N_SAMP)


def mel_of(w):
    return librosa.power_to_db(librosa.feature.melspectrogram(
        y=w, sr=FS, n_fft=2048, hop_length=HOP, n_mels=N_MELS, fmin=20, fmax=11000, power=2.0),
        ref=1.0, amin=1e-12, top_db=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="multi-v2")
    a = ap.parse_args()
    ck = torch.load(os.path.join(OUT_DIR, "models", f"cnn_{a.model}.pt"), weights_only=False)
    norm = Norm.from_state(ck["norm"])
    net = Net(len(ck["cols"]), len(LABELS), len(VESSEL_CLASSES), ck["width"], use_lofar=True)
    net.load_state_dict(ck["state"])
    net.eval()

    files = {os.path.basename(p): windows(p) for p in sorted(glob.glob(os.path.join(a.data, "*.wav")))}
    mels = {k: np.stack([mel_of(w) for w in v]) for k, v in files.items()}
    amb = [k for k in mels if "lluvia" in k][0]
    # noise-floor alignment against simulated calm windows
    import pandas as pd
    sim = np.load(os.path.join(OUT_DIR, "harbor_shelf_s2", "mel.npy")).astype(np.float32)
    df = pd.read_pickle(os.path.join(OUT_DIR, "harbor_shelf_s2", "windows.pkl"))
    sim_ref = np.percentile(sim[~df["anomaly"].values], 10)
    gain_db = sim_ref - np.percentile(mels[amb], 10)
    print(f"global gain {gain_db:+.1f} dB (sim p10 {sim_ref:.1f} dB vs real ambient p10 {np.percentile(mels[amb], 10):.1f} dB)")

    out = {"global_gain_db": float(gain_db), "files": {}}
    vec_neutral = np.tile(norm.vmed, (1, 1))
    for k, m in mels.items():
        lof = np.stack([lofar_whitened(w) for w in files[k]])
        comb = [comb_features(l) for l in lof]
        n = len(m)
        vec = np.tile(norm.vmed, (n, 1))
        names = ck["cols"]
        for j, c in enumerate(names):                      # fill comb features from the real signal
            if c in ("f_comb_f0_hz", "f_comb_strength", "f_lofar_tones"):
                vec[:, j] = [d[c[2:]] for d in comb]
        from simulation.ocean.models import Data
        d = Data(norm.mel(m + gain_db), norm.vec(vec), norm.lofar(lof), torch.zeros(n, len(LABELS)),
                 torch.full((n,), -1))
        P, VC = predict(net, d)
        key = [s for s in META if s in k][0]
        row = {"description": META[key], "n_windows": n,
               "mean_prob": {l: float(P[:, i].mean()) for i, l in enumerate(LABELS)},
               "vessel_detected_fraction(p>0.5)": float((P[:, 0] > 0.5).mean()),
               "median_comb_f0_hz": float(np.median([c["comb_f0_hz"] for c in comb])),
               "median_comb_strength": float(np.median([c["comb_strength"] for c in comb])),
               "vessel_type_probs_on_detected": {c: float(VC[P[:, 0] > 0.5][:, i].mean()) if (P[:, 0] > 0.5).any() else None
                                                 for i, c in enumerate(VESSEL_CLASSES)}}
        out["files"][k] = row
        print(f"{k:38s} vessel mean p={row['mean_prob']['is_vessel']:.2f} detected {row['vessel_detected_fraction(p>0.5)']:.0%}"
              f" | storm mean p={row['mean_prob']['storm']:.2f} fault mean p={row['mean_prob']['fault_any']:.2f}"
              f" | comb f0 {row['median_comb_f0_hz']:.1f} Hz (strength {row['median_comb_strength']:.2f})")
    # spectral-envelope check: real vs simulated mel level percentiles after alignment
    real_all = np.concatenate([m + gain_db for m in mels.values()])
    q = [5, 50, 95]
    out["mel_percentiles_db"] = {"real_aligned": [float(np.percentile(real_all, p)) for p in q],
                                 "sim_calm": [float(np.percentile(sim[~df["anomaly"].values], p)) for p in q],
                                 "sim_vessel": [float(np.percentile(sim[df["is_vessel"].values], p)) for p in q]}
    with open(os.path.join(OUT_DIR, "real_check.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
