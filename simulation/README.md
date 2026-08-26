# Simulation

## What this is, and isn't

This `/simulation` layer is synthetic data and code standing in for real
hardware until it arrives -- see [../DECISIONS.md](../DECISIONS.md): no
hardware purchased, no field recordings exist yet. Everything under
`simulation/data_generator/` synthesizes
plausible hydrophone audio and environmental sensor readings (ambient noise,
biological calls, vessel passages, storm runoff events); `simulation/pipeline/`
implements the real Stage 1/2 feature extraction and anomaly detection
described in [../docs/ml-pipeline.md](../docs/ml-pipeline.md), running
against that synthetic data.

This is **not** a claim that the synthetic data matches real underwater
acoustics or water chemistry exactly (see the domain-reasoning comments in
each `synthetic_*.py` module for what's simplified and why). Its purpose is
to let the feature-extraction, storage, and anomaly-detection code be
written, run, and evaluated end-to-end now, so it's ready to point at real
sensor input once hardware exists, instead of being designed blind.

## How to run it

From the repo root:

```
pip install -r requirements.txt
python simulation/scripts/run_simulation.py
python simulation/scripts/evaluate.py
```

`run_simulation.py` simulates a run of duty-cycle windows (default 150),
randomly injecting some biological calls and vessel noise events, and
(sometimes) one multi-window storm runoff event, matching the "record N
seconds every M minutes" duty cycle from DECISIONS.md. It writes:

- `simulation/output/audio/*.wav` -- Tier 1 flat audio files, one per window.
- `simulation/output/db.sqlite` -- Tier 2 structured data (captures, feature
  vectors, environmental readings), schema matching
  [../docs/data-pipeline.md](../docs/data-pipeline.md).
- `simulation/output/ground_truth.json` -- which windows got which anomaly
  injected, plus each window's computed feature vector, for evaluation.

The first ~20% of windows are deliberately generated anomaly-free, standing
in for the initial calibration period Stage 2 (docs/ml-pipeline.md) treats
as the normal baseline.

Useful flags: `--n-windows`, `--window-interval-minutes`, `--seed` (for a
reproducible run).

`evaluate.py` loads that ground truth, fits
`BaselineAnomalyDetector` (Isolation Forest) on the calibration-period
windows only, scores every remaining window, and compares its flags against
the known injected anomalies.

## What the printed metrics mean

- **Precision**: of the windows the detector flagged as anomalous, what
  fraction actually had an injected anomaly. Low precision means too many
  false alarms (normal windows getting flagged).
- **Recall**: of the windows that actually had an injected anomaly, what
  fraction the detector caught. Low recall means real events are being
  missed.
- **F1**: the harmonic mean of precision and recall, a single number that
  only scores well if both are reasonably good -- useful for comparing runs
  or detector settings at a glance.

These numbers describe how well Isolation Forest, as configured today,
separates synthetic anomalies from synthetic ambient background --  they
say nothing yet about real-world performance, since no real hardware or
field data exists to validate against.

## Multi-seed evaluation (stable benchmark)

`run_simulation.py`'s anomaly injection is randomized (which windows get a
biological call or vessel event, whether a storm event happens at all and
where it lands), so a single `run_simulation.py` + `evaluate.py` pass is one
sample from a distribution, not a stable number -- an early single run
happened to land on an unusually easy draw. For a number worth reporting
(e.g. in a paper), use:

```
python simulation/scripts/multi_seed_evaluate.py --n-seeds 30
```

This runs `run_simulation()` + `evaluate.compute_results()` across 30
independent seeds (0..29 by default), aggregates mean +/- standard
deviation for overall and per-anomaly-type precision/recall/F1, and writes
`simulation/output/multi_seed_results.json` (per-seed results plus the
aggregate). The top-level [README](../README.md) reports this aggregate,
not a single run. Useful flags: `--n-seeds`, `--n-windows`,
`--window-interval-minutes`, `--base-seed`.
