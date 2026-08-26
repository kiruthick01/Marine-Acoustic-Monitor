"""
Multi-seed evaluation: mean +/- std precision/recall/F1 across several
independent simulated runs.

Why this exists: simulation/README.md and the top-level README report
precision/recall/F1 from a *single* run_simulation.py + evaluate.py pass.
Anomaly injection (biological calls, vessel events, whether a storm event
happens at all and where) is randomized per DECISIONS.md's design, so a
single run's numbers are one sample from a distribution, not a stable
benchmark -- notably susceptible to an unusually easy or hard draw (e.g. a
storm event covering most of the evaluation window, as the single-run
numbers previously reported here happened to include).

This script runs run_simulation() + evaluate.compute_results() across
several seeds, aggregates each metric's mean and standard deviation, and
writes both the per-seed results and the aggregate to
simulation/output/multi_seed_results.json.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.scripts.evaluate import BREAKDOWN_ANOMALY_TYPES, compute_results
from simulation.scripts.run_simulation import GROUND_TRUTH_PATH, run_simulation

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "multi_seed_results.json")


def _mean_std(values: list) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def run_multi_seed(
    n_seeds: int = 20,
    n_windows: int = 150,
    window_interval_minutes: float = 10,
    base_seed: int = 0,
    results_path: str = OUTPUT_PATH,
) -> dict:
    """
    Run n_seeds independent simulations (seeds base_seed..base_seed+n_seeds-1)
    and aggregate overall + per-anomaly-type precision/recall/F1.

    Each seed gets its own run_simulation() call (fresh synthetic audio,
    environmental series, and anomaly injection draws), then
    evaluate.compute_results() fits a fresh detector on that run's own
    calibration period and scores its own evaluation set -- runs are fully
    independent, matching how a single evaluate.py invocation works today.

    Returns:
        Aggregate dict with per-seed raw results and mean/std for overall
        and per-type precision/recall/F1, plus how many seeds produced a
        usable evaluation set (a seed can yield none if anomaly injection
        left too few post-calibration windows -- see run_simulation.py's
        MIN_TRUE_NEGATIVE_FRACTION warning).
    """
    per_seed_results = []
    skipped_seeds = []

    for i in range(n_seeds):
        seed = base_seed + i
        run_simulation(n_windows=n_windows, window_interval_minutes=window_interval_minutes, seed=seed)
        result = compute_results(GROUND_TRUTH_PATH)
        if result is None:
            skipped_seeds.append(seed)
            continue
        result["seed"] = seed
        per_seed_results.append(result)
        print(
            f"seed {seed}: precision={result['overall']['precision']:.3f} "
            f"recall={result['overall']['recall']:.3f} f1={result['overall']['f1']:.3f}"
        )

    if not per_seed_results:
        raise RuntimeError(
            f"All {n_seeds} seeds produced no usable evaluation set -- increase --n-windows."
        )

    overall_agg = {
        metric: _mean_std([r["overall"][metric] for r in per_seed_results])
        for metric in ("precision", "recall", "f1")
    }

    per_type_agg = {}
    for anomaly_type in BREAKDOWN_ANOMALY_TYPES:
        per_type_agg[anomaly_type] = {
            metric: _mean_std([r["per_anomaly_type"][anomaly_type][metric] for r in per_seed_results])
            for metric in ("precision", "recall", "f1")
        }

    aggregate = {
        "n_seeds_requested": n_seeds,
        "n_seeds_used": len(per_seed_results),
        "skipped_seeds": skipped_seeds,
        "n_windows": n_windows,
        "window_interval_minutes": window_interval_minutes,
        "overall": overall_agg,
        "per_anomaly_type": per_type_agg,
        "per_seed": per_seed_results,
    }

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{len(per_seed_results)}/{n_seeds} seeds used" + (f" ({len(skipped_seeds)} skipped)" if skipped_seeds else ""))
    print("-- Overall (mean +/- std across seeds) --")
    for metric in ("precision", "recall", "f1"):
        stats = overall_agg[metric]
        print(f"{metric.capitalize():10s}: {stats['mean']:.3f} +/- {stats['std']:.3f}")
    print(f"\nResults written: {os.path.abspath(results_path)}")

    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the simulation across multiple seeds and report mean +/- std precision/recall/F1."
    )
    parser.add_argument("--n-seeds", type=int, default=20, help="Number of independent seeded runs.")
    parser.add_argument("--n-windows", type=int, default=150, help="Duty-cycle windows per run.")
    parser.add_argument(
        "--window-interval-minutes", type=float, default=10, help="Minutes between duty-cycle windows."
    )
    parser.add_argument("--base-seed", type=int, default=0, help="First seed used; seeds are base..base+n-1.")
    parser.add_argument("--results-out", default=OUTPUT_PATH, help="Path to write multi_seed_results.json to.")
    args = parser.parse_args()

    run_multi_seed(
        n_seeds=args.n_seeds,
        n_windows=args.n_windows,
        window_interval_minutes=args.window_interval_minutes,
        base_seed=args.base_seed,
        results_path=args.results_out,
    )
