"""
Detector ablation: why storm-type anomalies are missed, and what a
different scorer buys.

Reproduces the subspace-anomaly-dilution finding documented in
docs/ml-pipeline.md Stage 2's known-limitation note. Across N seeded
simulation runs, compares two scorers under the identical calibration-fit /
mean + 3*std thresholding protocol evaluate.py uses:

1. **Isolation Forest on the joint vector** (the current pipeline,
   simulation/pipeline/anomaly_detection.BaselineAnomalyDetector). Detects
   acoustically-driven anomalies (vessel/biological, which shift 20+ of the
   42 joint dims) near-perfectly, but largely misses storm runoff, which
   deviates in only ~2-3 environmental dims -- Isolation Forest picks split
   features uniformly at random, so 34 normal-looking acoustic dims dilute
   the trees and the storm point isolates no faster than normal points.
   Not a scaling problem (Isolation Forest is per-feature scale-invariant;
   standardizing first produces identical scores) and not a weak-signal
   problem (storm windows sit at turbidity |z| ~ 20-35 vs calibration).

2. **Mean-squared-z scoring**: each window's score is the mean over
   features of its squared z-score against calibration mean/std -- the
   degenerate (identity-reconstruction, diagonal-Gaussian) form of Stage
   2's other sanctioned candidate, autoencoder reconstruction error. A
   |z| ~ 30 deviation in even one dim dominates the mean, so subspace
   anomalies surface; the cost is more false alarms overall.

Neither dominates -- the printed table shows the precision/recall trade so
the choice (keep Isolation Forest, switch, or combine) can be made on
numbers rather than intuition.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# allow running as `python simulation/scripts/detector_ablation.py` (repo
# root isn't on sys.path in that form, only via `python -m ...`)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.pipeline.anomaly_detection import BaselineAnomalyDetector
from simulation.scripts.evaluate import _confusion_matrix, _precision_recall_f1
from simulation.scripts.run_simulation import GROUND_TRUTH_PATH, run_simulation

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "detector_ablation.json")

# Matches BaselineAnomalyDetector's default flagging rule (mean +
# threshold_sigma * std of calibration scores), applied identically to both
# scorers so the comparison isolates the scoring function, not the threshold.
THRESHOLD_SIGMA = 3.0


def _mean_squared_z_flags(cal_df: pd.DataFrame, ev_df: pd.DataFrame) -> list:
    """
    Score every evaluation window as the mean squared z-score of its
    features against calibration statistics, then flag windows whose score
    exceeds mean + THRESHOLD_SIGMA * std of the calibration windows' own
    scores -- the same thresholding rule BaselineAnomalyDetector applies to
    its Isolation Forest scores.
    """
    mean = cal_df.mean()
    std = cal_df.std().replace(0, 1e-9)
    cal_scores = (((cal_df - mean) / std) ** 2).mean(axis=1).to_numpy()
    ev_scores = (((ev_df - mean) / std) ** 2).mean(axis=1).to_numpy()
    threshold = cal_scores.mean() + THRESHOLD_SIGMA * cal_scores.std()
    return list(ev_scores > threshold)


def run_ablation(n_seeds: int = 30, n_windows: int = 150, base_seed: int = 0,
                 results_path: str = OUTPUT_PATH) -> dict:
    """
    Run both scorers across n_seeds independent simulations and aggregate
    overall + storm-type precision/recall/F1 (mean/std across seeds).
    """
    per_seed = []

    for i in range(n_seeds):
        seed = base_seed + i
        run_simulation(n_windows=n_windows, seed=seed)
        with open(GROUND_TRUTH_PATH) as f:
            run = json.load(f)

        windows = run["windows"]
        cal_n = run["calibration_windows"]
        cal, ev = windows[:cal_n], windows[cal_n:]
        if not ev:
            continue

        cal_df = pd.DataFrame([w["feature_vector"] for w in cal])
        ev_df = pd.DataFrame([w["feature_vector"] for w in ev])
        actual = [bool(w["true_anomaly"]) for w in ev]
        storm_actual = [w.get("anomaly_type") == "storm" for w in ev]

        detector = BaselineAnomalyDetector(random_state=0).fit(cal_df)
        flags_iforest = [bool(detector.score(ev_df.iloc[j])["is_anomaly"]) for j in range(len(ev_df))]
        flags_msz = _mean_squared_z_flags(cal_df, ev_df)

        row = {"seed": seed}
        for name, flags in (("isolation_forest", flags_iforest), ("mean_squared_z", flags_msz)):
            row[name] = {
                "overall": _precision_recall_f1(_confusion_matrix(flags, actual)),
                "storm": _precision_recall_f1(_confusion_matrix(flags, storm_actual)),
            }
        per_seed.append(row)
        print(
            f"seed {seed}: iforest f1={row['isolation_forest']['overall']['f1']:.3f} "
            f"(storm {row['isolation_forest']['storm']['f1']:.3f})  "
            f"msz f1={row['mean_squared_z']['overall']['f1']:.3f} "
            f"(storm {row['mean_squared_z']['storm']['f1']:.3f})"
        )

    aggregate = {"n_seeds": len(per_seed), "n_windows": n_windows, "scorers": {}}
    for name in ("isolation_forest", "mean_squared_z"):
        aggregate["scorers"][name] = {
            scope: {
                metric: {
                    "mean": float(np.mean([r[name][scope][metric] for r in per_seed])),
                    "std": float(np.std([r[name][scope][metric] for r in per_seed])),
                }
                for metric in ("precision", "recall", "f1")
            }
            for scope in ("overall", "storm")
        }
    aggregate["per_seed"] = per_seed

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{len(per_seed)} seeds (mean +/- std)")
    print(f"{'':24s}{'IsolationForest':>22s}{'mean-squared-z':>22s}")
    for scope in ("overall", "storm"):
        for metric in ("precision", "recall", "f1"):
            cells = []
            for name in ("isolation_forest", "mean_squared_z"):
                s = aggregate["scorers"][name][scope][metric]
                cells.append(f"{s['mean']:.3f} +/- {s['std']:.3f}")
            print(f"{scope + ' ' + metric:24s}{cells[0]:>22s}{cells[1]:>22s}")
    print(f"\nResults written: {os.path.abspath(results_path)}")

    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Isolation Forest vs mean-squared-z scoring across seeded simulation runs."
    )
    parser.add_argument("--n-seeds", type=int, default=30, help="Number of independent seeded runs.")
    parser.add_argument("--n-windows", type=int, default=150, help="Duty-cycle windows per run.")
    parser.add_argument("--base-seed", type=int, default=0, help="First seed used; seeds are base..base+n-1.")
    parser.add_argument("--results-out", default=OUTPUT_PATH, help="Path to write detector_ablation.json to.")
    args = parser.parse_args()

    run_ablation(
        n_seeds=args.n_seeds,
        n_windows=args.n_windows,
        base_seed=args.base_seed,
        results_path=args.results_out,
    )
