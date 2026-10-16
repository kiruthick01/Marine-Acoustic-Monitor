# Project highlights: presentation and interview guide

Everything here is backed by a file in this repository. Where a claim has limits, the limit is stated next to it.
Numbers come from `simulation/ocean/results/*.json` (synthetic data unless marked otherwise).

## 30-second pitch

A low-cost, solar-powered marine monitoring node: a hydrophone plus temperature, pH, turbidity and salinity sensors on a
Raspberry Pi, flagging vessels, seismic surveys, storms, wildlife and its own sensor faults on-device, and sending only compact
summaries over a low-bandwidth link. With no hardware or field data yet, I built a physics-based ocean simulator with coupled weather,
water chemistry, animals, ships and a modelled sensor chain, and used it to design and stress-test the ML pipeline: one model that,
trained on three simulated sites, transfers to two sites it never saw (mean mAP 0.975 over six unseen weather/season/site tests, with a
41 ms per-window compute cost) without any new labels.

## Qualities that make it a strong project (and where the evidence is)

| # | Quality | Evidence |
|---|---|---|
| 1 | **Real problem, system view**: sensing, edge software, storage tiers, telemetry, ML, hardware spec, power budget | `docs/`, `edge/`, `tools/power_budget.py` |
| 2 | **Physics-grounded simulation, not random noise**: Mackenzie sound speed, Ainslie-McColm absorption with pH, Wenz noise, waveguide cutoff, Ross source levels, coupled fronts -> wind -> waves -> noise and rain -> runoff -> salinity/turbidity/pH | `simulation/ocean/physics.py`, `weather.py`, 5 physics tests |
| 3 | **Everything is coupled and labelled from ground truth** (per-source peak SNR audibility, storm cause, fault type) | `scenario.py` |
| 4 | **Leakage-free evaluation**: chronological and whole-run splits, two sites never used in training, two seasons each, adaptation slice excluded from test | README "Corpus and protocol" |
| 5 | **Baselines and ablations**: GB vs CNN, inputs x training-corpus grid, AdaBN, scratch, fine-tuning curves | fig 10, 11; `classification_results.json` |
| 6 | **Statistical care**: day-block bootstrap CIs, fine-tuning repeated over 3 seeds, event counts reported, small-n warnings | `bootstrap_results.json` |
| 7 | **Diagnosis-driven iteration**: each weak spot was traced to a root cause before fixing, and the fix was measured | table below, fig 9 |
| 8 | **Domain knowledge in the features**: narrowband LOFAR + blade-rate comb estimator (how sonar operators tell ships apart); causal flat-line sensor QC | `lofar.py`, `dataset.py`, tests |
| 9 | **Generalisation by design**: domain randomisation over depth, traffic, runoff, climate; transfer to unseen sites with zero labels | fig 10 |
| 10 | **Honest negative results**: AdaBN hurts storm detection, fine-tuning can cause negative transfer, airgun at the audibility limit is undetectable, cargo/tanker has a physical Bayes cap | README caveats |
| 11 | **Physical upper bound** for a hard sub-task (cargo vs tanker Bayes accuracy 0.88 from blade rate alone) instead of just "accuracy is low" | `extra_results.json` |
| 12 | **Robustness study**: calibration drift, noise, sensor outage, disabled branches | fig 15 |
| 13 | **Calibration and uncertainty**: reliability diagrams, per-label temperature scaling, ECE | fig 13 |
| 14 | **Edge feasibility measured**: 41 ms per window on one host thread, 1.4 MB model, 0.07 % of the duty cycle at 10x slower hardware | `edge_benchmark.json` |
| 15 | **Real-data smoke test** on four ShipsEar recordings, with the sim-to-real gap reported rather than hidden | README, `real_check.json` |
| 16 | **Reproducible**: seeded generators, one script per stage, committed result snapshots, unit tests, CI | `simulation/scripts`, `.github/workflows/ci.yml` |
| 17 | **Documentation discipline**: locked decisions, doc audit, limitations section, no unverifiable citations | `DECISIONS.md`, `docs/` |

## The story in one table: weak spots found, diagnosed, fixed

| Weak spot | Root cause (verified) | Fix | Before -> after |
|---|---|---|---|
| Instrument-fault AP 0.44 | 22 of 39 fault windows were stuck-sensor episodes; a flat-line cannot be seen in one window | causal run-length / variability features | AP 0.44 -> **1.00**; stuck-sensor recall 0 -> 1.0 |
| Cargo vs tanker at chance | 64 mel bins have 2-3 bins below 100 Hz; blade-rate comb is 4-9 Hz | 0.67 Hz LOFAR, whitening, comb estimator, 1-D branch | accuracy 0.5 -> 0.70-0.96, AUC 0.80-1.00 (cap 0.88 from physics) |
| Airgun AP ~0 at new site | no low-SNR examples, none in test slice | campaign in every run, wider range, multi-site data | AP ~0 -> 0.80 / 0.99 / 1.00 (B spring / E); B summer stays ~0, events at the audibility limit |
| Season gap 0.809 -> 0.837 after fine-tuning | one site, one season of training data | domain-randomised training | other-season mAP 0.941 (B), 0.989 (E), zero labels |

## Likely questions and straight answers

**Is the 0.975 a real-world number?** No. It is accuracy on simulated data produced by models I wrote, so it shows the pipeline can learn
that simulated physics and transfer across parameterised sites. The real-data evidence is a four-file smoke test (real vessels flagged,
real rain not flagged as a vessel) plus a measured sim-to-real spectral gap. Real validation needs deployment data.

**Why simulate at all?** No hardware or field data exists yet, and a simulator gives ground-truth labels for every source, which no
public dataset provides jointly for sound, water chemistry and sensor faults. The simulator also lets me test failure modes (stuck sensors,
gain drift) before buying hardware. Its risk, that it only teaches the model my assumptions, is why the claims are limited above.

**How did you avoid fooling yourself?** Whole held-out sites and seasons, chronological splits, a fixed adaptation slice excluded from test,
bootstrap CIs, ablations that isolate each ingredient, and reporting failures (AdaBN hurts, fine-tuning can hurt, B-summer airgun).

**What was the most interesting finding?** The first version's fix for the season gap was to fine-tune; the better answer turned out to be training
on diverse sites so fine-tuning is mostly unnecessary. And the interaction: multi-site data with the *old* inputs was worse at one site
(0.774), so diversity only helps once the features expose the discriminating signal (LOFAR for ships).

**Why is cargo vs tanker hard?** In the simulator their blade-rate ranges overlap by design (6-6.8 Hz), so even an oracle that knows the true blade
rate tops out at 0.88. Real ships overlap more. I report both the achieved and the physical bound.

**Why a CNN on a log-mel and not a big pretrained model?** The compute budget is a Pi duty-cycling one window per 10 minutes (the model is 0.34 M
parameters, 6 ms forward), and the discriminating cues at low frequency are not in a mel spectrogram, which is why a purpose-built LOFAR branch
helped more than model size would. Pretrained audio embeddings are listed as future work.

**What would you do with real data?** Calibrate the hydrophone chain, fit the noise-floor alignment on a quiet deployment week, evaluate the
vessel head on ShipsEar/DeepShip with the simulator-pretrained weights as initialisation, and measure the sim-to-real gap per label before trusting any of them.

**What did not work?** Unsupervised anomaly detection tops out around AUPRC 0.5 (0.494 best vs 0.394 for the original detector), because vessels
and storms are 10-30 % of windows, so "normal" is not rare and the calibration window is contaminated. Weekly recalibration did not help. Labels close the gap
(AUPRC 0.99 supervised), which motivates the labelled stage.

**How robust is it?** Insensitive to +-3 dB gain error and to losing all environmental sensors; a +-6 dB uncalibrated error costs up to 0.06 mAP; health context
matters where sensors fail (-0.077 at the site with stuck-sensor episodes).

## What to say about limitations before you are asked

1. Synthetic data, same author for simulator and model; one four-file real smoke test.
2. Held-out sites are the same simulator with different parameters.
3. Labels are thresholded truth, so boundary cases (wind 10 m/s, 10 dB audibility) are ambiguous by construction.
4. Small event counts for faults/airgun at some test slices; read differences under ~0.02 with the CIs.
5. Pi timings are extrapolated from host timings, not measured on a Pi.
6. Fishing boats are never audible at the 10 dB criterion, so the vessel-type head is effectively four-class.

## Possible next steps (in order of value)

1. Collect one real deployment week (or use DeepShip/ShipsEar in full) and measure per-label sim-to-real gap.
2. Measure actual Raspberry Pi 4/5 latency and power per window to replace the extrapolation.
3. Make fishing vessels audible in the simulator or merge the class; add more passenger-ship and sailing-boat variants.
4. Replace the 5 s mel window with a streaming detector for airgun pulses at the audibility limit.
5. Pretrained audio embeddings (BirdNET/AudioSet) as a comparison against the purpose-built branch.
