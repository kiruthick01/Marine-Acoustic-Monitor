"""
Anomaly detector evaluation.

Loads the ground-truth anomaly metadata written by run_simulation.py, fits
simulation/pipeline/anomaly_detection.BaselineAnomalyDetector on the
designated calibration-period windows (assumed normal, matching Stage 2's
calibration period in docs/ml-pipeline.md), scores every remaining window,
and prints:

- An overall confusion matrix and precision/recall/F1 against the known
  injected anomalies (biological calls, vessel events, storm runoff)
  recorded in the ground truth.
- A per-anomaly-type breakdown (vessel, storm) with its own confusion
  matrix and precision/recall/F1, since the detector is a single binary
  anomaly/not-anomaly classifier -- this shows whether it's systematically
  better at catching one injected anomaly type than another, which the
  overall numbers alone would hide.
- A mean SNR-improvement sanity check for Stage 0 signal conditioning
  (simulation/pipeline/signal_conditioning.py), using the before/after
  diagnostic every window's ground truth carries.

All of the above is also written to simulation/output/evaluation_results.json
as structured data, for downstream tooling/notebooks that want the numbers
without re-parsing this script's print output.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# allow running as `python simulation/scripts/evaluate.py` (repo root isn't
# on sys.path in that form, only via `python -m simulation.scripts...`)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.pipeline.anomaly_detection import BaselineAnomalyDetector

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "ground_truth.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "evaluation_results.json")

# Anomaly types broken out individually in the per-type report. Biological
# calls are deliberately excluded here even though the ground truth tracks
# them (run_simulation.py's `anomaly_type`) -- vessel and storm are the two
# types explicitly called for in this breakdown; adding biological later is
# just another entry in this list.
BREAKDOWN_ANOMALY_TYPES = ["vessel", "storm"]


def _confusion_matrix(predicted_flags: list, actual_flags: list) -> dict:
    """
    Count TP/FP/FN/TN from parallel lists of predicted and actual booleans.
    """
    true_positives = false_positives = false_negatives = true_negatives = 0
    for predicted, actual in zip(predicted_flags, actual_flags):
        if predicted and actual:
            true_positives += 1
        elif predicted and not actual:
            false_positives += 1
        elif not predicted and actual:
            false_negatives += 1
        else:
            true_negatives += 1

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }


def _precision_recall_f1(confusion: dict) -> dict:
    """
    Precision: of windows the detector flagged, how many were real
    anomalies (of the type this confusion matrix's "actual" is scoped to).
    Recall: of real anomalies (of that type), how many the detector caught.
    F1: harmonic mean of the two, a single number balancing both.
    """
    tp = confusion["true_positives"]
    fp = confusion["false_positives"]
    fn = confusion["false_negatives"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def _print_confusion_and_metrics(label: str, confusion: dict, metrics: dict) -> None:
    print(f"-- {label} --")
    print(f"True positives:  {confusion['true_positives']}")
    print(f"False positives: {confusion['false_positives']}")
    print(f"False negatives: {confusion['false_negatives']}")
    print(f"True negatives:  {confusion['true_negatives']}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print()


def evaluate(ground_truth_path: str = GROUND_TRUTH_PATH, results_path: str = RESULTS_PATH) -> None:
    """
    Fit the detector on the calibration period, score the rest, and report
    overall + per-anomaly-type metrics plus a signal-conditioning SNR
    sanity check.

    Args:
        ground_truth_path: path to the ground_truth.json written by
            run_simulation.py.
        results_path: path to write the structured evaluation_results.json
            summary to.
    """
    with open(ground_truth_path) as f:
        run = json.load(f)

    windows = run["windows"]
    calibration_windows = run["calibration_windows"]

    calibration_set = windows[:calibration_windows]
    evaluation_set = windows[calibration_windows:]

    if not evaluation_set:
        print("No windows after the calibration period to evaluate against -- run with more --n-windows.")
        return

    calibration_df = pd.DataFrame([w["feature_vector"] for w in calibration_set])
    detector = BaselineAnomalyDetector(random_state=0).fit(calibration_df)

    predicted_flags = []
    for window in evaluation_set:
        feature_vector = pd.Series(window["feature_vector"])
        result = detector.score(feature_vector)
        predicted_flags.append(bool(result["is_anomaly"]))

    true_anomaly_flags = [bool(w["true_anomaly"]) for w in evaluation_set]

    # --- Overall confusion matrix / precision / recall / F1 ---
    overall_confusion = _confusion_matrix(predicted_flags, true_anomaly_flags)
    overall_metrics = _precision_recall_f1(overall_confusion)

    # --- Per-anomaly-type breakdown ---
    # For each type, "actual" is re-scoped to just that type -- windows with
    # a different injected anomaly type, or none, are all treated as
    # negatives for this subset, per the task: this measures how well the
    # single binary detector does specifically against that one anomaly
    # type, not how well it'd do at classifying type (it doesn't).
    per_type_results = {}
    for anomaly_type in BREAKDOWN_ANOMALY_TYPES:
        type_flags = [w.get("anomaly_type") == anomaly_type for w in evaluation_set]
        type_confusion = _confusion_matrix(predicted_flags, type_flags)
        type_metrics = _precision_recall_f1(type_confusion)
        per_type_results[anomaly_type] = {"confusion_matrix": type_confusion, **type_metrics}

    # --- Signal conditioning SNR sanity check ---
    # Computed across every simulated window (calibration + evaluation),
    # not just the evaluation set, since Stage 0 conditioning runs
    # unconditionally on every capture regardless of the calibration split.
    snr_before = [w["signal_conditioning"]["snr_before_db"] for w in windows]
    snr_after = [w["signal_conditioning"]["snr_after_db"] for w in windows]
    snr_improvement = [after - before for before, after in zip(snr_before, snr_after)]

    snr_stats = {
        "mean_snr_before_db": float(np.mean(snr_before)),
        "mean_snr_after_db": float(np.mean(snr_after)),
        "mean_snr_improvement_db": float(np.mean(snr_improvement)),
        "n_windows": len(windows),
    }

    n_eval = len(evaluation_set)
    n_true_anomalies = sum(true_anomaly_flags)

    # --- Print report ---
    print(f"Calibration period: {calibration_windows} windows (assumed normal, used to fit the baseline)")
    print(f"Evaluation period:  {n_eval} windows ({n_true_anomalies} with a true injected anomaly)")
    print()

    _print_confusion_and_metrics("Overall", overall_confusion, overall_metrics)

    for anomaly_type in BREAKDOWN_ANOMALY_TYPES:
        result = per_type_results[anomaly_type]
        _print_confusion_and_metrics(
            f"Per-type: {anomaly_type}", result["confusion_matrix"], result
        )

    print("-- Signal conditioning (Stage 0) SNR sanity check --")
    print(f"Mean SNR before: {snr_stats['mean_snr_before_db']:.3f} dB")
    print(f"Mean SNR after:  {snr_stats['mean_snr_after_db']:.3f} dB")
    print(f"Mean improvement: {snr_stats['mean_snr_improvement_db']:.3f} dB (across {snr_stats['n_windows']} windows)")

    # --- Write structured results ---
    results = {
        "calibration_windows": calibration_windows,
        "evaluation_windows": n_eval,
        "n_true_anomalies": n_true_anomalies,
        "overall": {"confusion_matrix": overall_confusion, **overall_metrics},
        "per_anomaly_type": per_type_results,
        "signal_conditioning_snr": snr_stats,
    }
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written: {os.path.abspath(results_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the anomaly detector against simulation ground truth."
    )
    parser.add_argument(
        "--ground-truth",
        default=GROUND_TRUTH_PATH,
        help="Path to ground_truth.json written by run_simulation.py",
    )
    parser.add_argument(
        "--results-out",
        default=RESULTS_PATH,
        help="Path to write structured evaluation_results.json to",
    )
    args = parser.parse_args()
    evaluate(args.ground_truth, args.results_out)
