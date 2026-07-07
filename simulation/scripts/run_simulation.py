"""
End-to-end duty-cycle simulation runner.

Simulates N duty-cycle windows across a timeline (DECISIONS.md: "record N
seconds every M minutes"), using the synthetic generators to produce audio
and environmental data with randomly injected biological calls, vessel
noise events, and (at most) one storm runoff event. Each window is run
through signal conditioning (Stage 0, docs/ml-pipeline.md) and feature
extraction (Stage 1, docs/ml-pipeline.md), and written to Tier 1 (flat
audio files) and Tier 2 (SQLite), per docs/data-pipeline.md's storage
tiers.

Anomaly detection itself (Stage 2) is deliberately NOT run here -- this
script's job is generation, feature extraction, and storage. Detection and
scoring against ground truth happens in evaluate.py, so detector fitting
and threshold choices live in one place. Because the schema in
docs/data-pipeline.md still requires an `anomaly_flags` row per capture,
this script writes a placeholder (unscored) row for each window; a real
on-device deployment would instead score inline in this same loop
(docs/data-pipeline.md step 5-6).

The first `calibration_windows` windows are deliberately generated with no
anomalies injected, standing in for the initial calibration period Stage 2
(docs/ml-pipeline.md) treats as the normal baseline.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.io import wavfile

# allow running as `python simulation/scripts/run_simulation.py` (repo root
# isn't on sys.path in that form, only via `python -m simulation.scripts...`)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.data_generator.synthetic_audio import generate_duty_cycle_sample
from simulation.data_generator.synthetic_environmental import generate_environmental_series
from simulation.pipeline.feature_extraction import (
    build_joint_feature_vector,
    extract_acoustic_features,
    extract_environmental_features,
)
from simulation.pipeline.signal_conditioning import condition_signal
from simulation.pipeline.storage import init_db, insert_window_record

SAMPLE_RATE = 22050
WINDOW_DURATION_S = 5  # short duty-cycle capture window, kept small for a fast demo run

BIOLOGICAL_CALL_PROBABILITY = 0.12
VESSEL_EVENT_PROBABILITY = 0.08
STORM_EVENT_PROBABILITY = 0.6  # chance the run includes one storm/runoff event somewhere in the timeline

# The initial calibration period (Stage 2, docs/ml-pipeline.md) is assumed
# normal by procedure in a real deployment -- no anomalies are injected
# into it here so evaluate.py's baseline fit reflects that assumption
# correctly rather than accidentally calibrating on anomalous data.
CALIBRATION_FRACTION = 0.2
MIN_CALIBRATION_WINDOWS = 10

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
DB_PATH = os.path.join(OUTPUT_DIR, "db.sqlite")
GROUND_TRUTH_PATH = os.path.join(OUTPUT_DIR, "ground_truth.json")


def run_simulation(
    n_windows: int = 100,
    window_interval_minutes: float = 10,
    seed: int = None,
) -> None:
    """
    Run the full simulation and write outputs to simulation/output/.

    Args:
        n_windows: number of duty-cycle windows to simulate.
        window_interval_minutes: minutes between windows. Kept as the
            module-level default (10) unless overridden, matching
            synthetic_environmental.py's own default so environmental
            rate-of-change reflects a consistent real-world cadence.
        seed: optional RNG seed for reproducible runs.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(AUDIO_DIR, exist_ok=True)

    calibration_windows = min(n_windows, max(MIN_CALIBRATION_WINDOWS, int(n_windows * CALIBRATION_FRACTION)))

    # Environmental series is generated once for the whole timeline (rather
    # than per-window) because the storm runoff anomaly is a single
    # multi-window event with an onset and gradual recovery -- it can't be
    # injected independently window-by-window the way a biological call or
    # vessel passage can. The onset is restricted to after the calibration
    # period so the calibration period stays anomaly-free.
    storm_onset = None
    post_calibration_windows = n_windows - calibration_windows
    if post_calibration_windows > 15 and random.random() < STORM_EVENT_PROBABILITY:
        storm_onset = random.randint(calibration_windows, n_windows - 5)

    env_series, env_meta = generate_environmental_series(n_windows, inject_anomaly_at=storm_onset)

    storm_range = set()
    if env_meta["anomaly_injected"]:
        onset = env_meta["onset_window"]
        duration = env_meta["duration_windows"]
        storm_range = set(range(onset, onset + duration))

    db_conn = init_db(DB_PATH)

    # Arbitrary fixed epoch so timestamps (and thus audio filenames) are
    # reproducible across runs with the same seed.
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    windows_out = []
    progress_step = max(n_windows // 10, 1)

    for window_index in range(n_windows):
        timestamp = start_time + timedelta(minutes=window_index * window_interval_minutes)
        timestamp_utc = timestamp.isoformat()

        # calibration-period windows are forced anomaly-free; see module docstring
        if window_index < calibration_windows:
            audio_anomaly = None
        else:
            roll = random.random()
            if roll < BIOLOGICAL_CALL_PROBABILITY:
                audio_anomaly = "biological"
            elif roll < BIOLOGICAL_CALL_PROBABILITY + VESSEL_EVENT_PROBABILITY:
                audio_anomaly = "vessel"
            else:
                audio_anomaly = None

        audio, audio_meta = generate_duty_cycle_sample(
            duration_s=WINDOW_DURATION_S, sample_rate=SAMPLE_RATE, inject_anomaly=audio_anomaly
        )

        # --- Tier 1: raw audio to a flat file, named by capture timestamp ---
        audio_filename = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}.wav"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        wavfile.write(audio_path, SAMPLE_RATE, audio)

        # --- Stage 0: signal conditioning (bandpass + spectral denoise) ---
        # Runs on the raw captured audio before feature extraction sees it;
        # Tier 1 storage below still writes the raw (unconditioned) audio,
        # since conditioning is a feature-extraction-time step, not a
        # change to what's archived.
        conditioned_audio, conditioning_diagnostics = condition_signal(audio, SAMPLE_RATE)

        # --- Stage 1: feature extraction ---
        env_row = env_series.iloc[window_index]
        acoustic_features = extract_acoustic_features(conditioned_audio, SAMPLE_RATE)
        environmental_features = extract_environmental_features(env_row)
        joint_vector = build_joint_feature_vector(acoustic_features, environmental_features)

        # --- Tier 2: structured data to SQLite ---
        placeholder_anomaly_result = {"anomaly_score": 0.0, "is_anomaly": False}
        capture_id = insert_window_record(
            db_conn,
            timestamp_utc=timestamp_utc,
            audio_filename=audio_filename,
            duration_sec=WINDOW_DURATION_S,
            sample_rate_hz=SAMPLE_RATE,
            acoustic_features=acoustic_features,
            environmental_row=env_row,
            anomaly_result=placeholder_anomaly_result,
        )

        env_anomaly_active = window_index in storm_range
        true_anomaly = bool(audio_anomaly) or env_anomaly_active

        # Single consolidated label for evaluate.py's per-type breakdown,
        # collapsing the two independently-drawn anomaly channels
        # (audio_anomaly_type, env_anomaly_active) into one field. Audio
        # anomalies take priority in the rare case a window has both an
        # audio anomaly and falls inside the storm's active range, since
        # audio_anomaly_type is itself the more specific/rarer event.
        if audio_anomaly == "vessel":
            anomaly_type = "vessel"
        elif audio_anomaly == "biological":
            anomaly_type = "biological"
        elif env_anomaly_active:
            anomaly_type = "storm"
        else:
            anomaly_type = None

        windows_out.append(
            {
                "window_index": window_index,
                "capture_id": capture_id,
                "timestamp_utc": timestamp_utc,
                "audio_filename": audio_filename,
                "true_anomaly": true_anomaly,
                "anomaly_type": anomaly_type,
                "audio_anomaly_type": audio_anomaly,
                "audio_onset_s": audio_meta["onset_s"],
                "env_anomaly_active": env_anomaly_active,
                "env_anomaly_type": "storm_runoff" if env_anomaly_active else None,
                "signal_conditioning": conditioning_diagnostics,
                # stashed so evaluate.py can fit/score without re-running
                # feature extraction or parsing it back out of SQLite
                "feature_vector": joint_vector.to_dict(),
            }
        )

        if (window_index + 1) % progress_step == 0 or window_index == n_windows - 1:
            print(
                f"window {window_index + 1}/{n_windows}: "
                f"true_anomaly={true_anomaly} (audio={audio_anomaly}, env_storm={env_anomaly_active})"
            )

    db_conn.close()

    ground_truth = {
        "n_windows": n_windows,
        "window_interval_minutes": window_interval_minutes,
        "calibration_windows": calibration_windows,
        "windows": windows_out,
    }
    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(ground_truth, f)

    n_true_anomalies = sum(1 for w in windows_out if w["true_anomaly"])
    print(f"\nSimulated {n_windows} windows ({calibration_windows} calibration, {n_true_anomalies} with a true anomaly).")
    print(f"Audio:        {os.path.abspath(AUDIO_DIR)}")
    print(f"Database:     {os.path.abspath(DB_PATH)}")
    print(f"Ground truth: {os.path.abspath(GROUND_TRUTH_PATH)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the end-to-end duty-cycle simulation.")
    parser.add_argument("--n-windows", type=int, default=100, help="Number of duty-cycle windows to simulate.")
    parser.add_argument(
        "--window-interval-minutes", type=float, default=10, help="Minutes between duty-cycle windows."
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for a reproducible run.")
    args = parser.parse_args()

    run_simulation(
        n_windows=args.n_windows,
        window_interval_minutes=args.window_interval_minutes,
        seed=args.seed,
    )
