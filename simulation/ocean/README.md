# Physics-based ocean simulator

Second-generation simulation layer (the original toy generators in `simulation/data_generator/` are untouched).
Everything is **synthetic**: results below measure the pipeline on simulated data, not real-world accuracy.

## What is modelled

| Layer | Module | Physics / behaviour |
|---|---|---|
| Acoustics | `physics.py` | Mackenzie sound speed; Ainslie-McColm absorption (T, S, **pH**, depth); practical spreading + shallow-water waveguide cutoff + thermocline penalty; Wenz/Coates ambient noise (wind, shipping, thermal, turbulence) + rain + tidal-flow noise; Ross vessel source levels |
| Atmosphere / ocean | `weather.py` | Solar geometry, synoptic fronts (wind, pressure, rain), fetch-limited waves, 4-harmonic tides + currents, SST relaxation, stratification, runoff → salinity/turbidity/pH, diel photosynthesis pH, plankton bloom, O₂ |
| Living beings | `bio.py` | Snapping shrimp chorus (temperature, diel), fish chorus (spawning season, dusk), dolphin pods (dawn/dusk Markov arrivals, whistles + clicks, ranges), baleen whale song (seasonal passage) |
| Human sources | `anthro.py` | Vessel traffic (cargo, tanker, ferry, fishing, outboard) with CPA tracks, Doppler, blade-rate tonal comb + cavitation continuum; seismic airgun campaigns; instrument faults (50/60 Hz hum, dropout, cable pops, stuck env sensor) |
| Instruments | `scenario.py` | Hydrophone sensitivity + preamp + 16-bit ADC clipping, biofouling drift, env-sensor noise/spikes/drift/stuck |
| Labels | `scenario.py` | Per-source peak time-frequency SNR (≥10 dB = audible), storm = wind>10.5 m/s or rain>2 mm/h or runoff>0.5, fault, `anomaly` = vessel ∪ airgun ∪ storm ∪ fault |

Air is modelled only through wind, rain, pressure, air temperature and irradiance. Molecular-scale air dynamics have no
effect on a submerged sensor and are deliberately not simulated. Not modelled: full-wave/ray propagation, bathymetry,
Lloyd's-mirror interference, individual species call libraries (shapes are generic).

## Run

```
python simulation/scripts/ocean_generate.py     # 6 evaluation runs + 3 event-rich training runs (~35 min)
python simulation/scripts/ocean_detect.py       # Stage 2 unsupervised detectors (~15 min)
python simulation/scripts/ocean_classify.py     # Stage 3 CNN/GB + fine-tuning (~1 h on CPU)
python simulation/scripts/ocean_render.py       # figures
python simulation/scripts/ocean_scene_gif.py    # animated vessel + dolphin scene
```

Outputs under `simulation/output/ocean/` (git-ignored): `windows.pkl`, `mel.npy`, example `audio/`, `figures/`, `*_results.json`.
Sites: A = 20 m harbor shelf (spring s0-s2, summer s3), B = 60 m open coast (domain-shift target, s10 spring, s11 summer).
Rich runs `rich_harbor_s4..6` (15 d, forced seismic campaigns, frequent faults) are used for classifier training only.

## Results (30-day runs, 4320 windows each, 5 s every 10 min)

### Stage 2, unsupervised anomaly detection (3-day self-calibration, tuned on dev run s2, tested on 5 unseen runs)

| Method | AUROC | AUPRC | F1 |
|---|---|---|---|
| M0 existing IF, 34 acoustic + 8 env | 0.678 | 0.423 | 0.428 |
| M1 IF, calibrated band levels + tonality + clock | 0.727 | 0.501 | 0.446 |
| **M6 self-cleaned IF (trimmed refit)** | **0.732** | **0.542** | **0.497** |
| M3 Mahalanobis / M4 autoencoder / M5 ensemble | 0.61-0.65 | 0.35-0.41 | 0.37-0.38 |
| Supervised ceiling (labelled, other runs) | 0.993 | 0.986 | 0.940 |

Honest reading: with realistic event prevalence (vessels ~13 % of windows, storms 4-27 %) "normal" is not rare, so
label-free detection tops out near AUPRC 0.54. Calibration contamination, not the model class, is the main limit;
labelled data closes the gap, which supports the Stage 3 plan. Airgun recall is poor (rare in calibration, 2 runs).

### Stage 3, supervised multi-label classification (mAP; threshold-free)

| Test set | Gradient boosting | CNN (log-mel + env) |
|---|---|---|
| A, unseen weather | 0.881 | 0.891 |
| A, other season | 0.911 | 0.971 |
| B, zero-shot new site | 0.762 | 0.918 |
| B, new site + other season | 0.636 | 0.809 |

Fine-tuning at site B on 648 labelled windows (4.5 days), all layers, lr 1e-3, 3 inits:
zero-shot 0.918 → **0.963 ± 0.005** (same season), 0.809 → 0.837 (other season);
CNN trained from scratch on the same 648 windows: 0.809 ± 0.104 / 0.559 ± 0.088; GB on A+B-small: 0.890 / 0.755.

Known weak spots: instrument-fault detection on unseen weather (AP 0.44), vessel-type (cargo vs tanker indistinguishable,
accuracy ~0.53), airgun at a new site (not present in B training data).

## Not done

Real-data validation (docs/datasets.md lists ShipsEar, DeepShip, Watkins) is not run yet; no download was attempted.
