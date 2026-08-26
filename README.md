# Marine Ecosystem Monitoring System

**Status: (In Development) — no hardware purchased yet; hardware BOM/pricing and a power budget are still open.** Design documentation is locked (see [DECISIONS.md](DECISIONS.md)). Two implementation trees now exist: `simulation/` (synthetic-hardware proof of the ML pipeline, see below) and `edge/` (the real production Pi-side software — capture loop, calibration, telemetry, scheduler — runnable today in mock-hardware mode with zero hardware attached; see [docs/pi-implementation.md](docs/pi-implementation.md)).

![Simulation results](diagrams/results-chart.svg)

| Precision | Recall | F1 |
|-----------|--------|-----|
| 0.972 ± 0.037 | 0.690 ± 0.261 | 0.777 ± 0.174 |

Mean ± standard deviation across 30 independently seeded simulation runs (150 duty-cycle windows each, 30-window calibration period), via `simulation/scripts/multi_seed_evaluate.py --n-seeds 30`. Replaces an earlier single-run figure that happened to land on an easy draw. Two things this aggregate makes visible that a single run hides:

- **Recall has high run-to-run variance** (± 0.261) — how well the detector does depends heavily on which anomalies a given run happens to inject and how they overlap the storm event, not just the average case.
- **Per-anomaly-type breakdown is much weaker for storm/environmental anomalies than acoustic ones**: across the same 30 runs, vessel-type recall averages 1.00 (precision 0.32 ± 0.10, since the binary detector isn't type-aware) while storm-type precision/recall both average ~0.05 — the detector rarely fires specifically on the injected environmental anomaly. Root cause now verified (see [docs/ml-pipeline.md](docs/ml-pipeline.md) Stage 2's known-limitation note): the joint vector is 34 acoustic vs. 8 environmental dims and Isolation Forest picks split features uniformly at random, so an anomaly confined to the small environmental subspace gets diluted — the storm signal itself is huge (turbidity |z| ≈ 33 vs. calibration) and an env-only detector catches it. A measured candidate remedy (mean-squared-z scoring) trades overall precision 0.97 → 0.72 for storm recall 0.05 → 0.59; which side of that trade to take is an open design decision.

Still describes only how Isolation Forest separates *synthetic* anomalies from *synthetic* ambient background — no real hardware or field data exists yet to validate against. See [simulation/README.md](simulation/README.md) for methodology.

## Motivation

Marine ecosystems generate continuous acoustic and environmental signal — biological activity, equipment/vessel noise, water-quality shifts, anomalous events — that's expensive to monitor with human observers or connectivity-dependent systems. This project designs a low-cost, edge-computed monitoring unit that combines passive acoustic monitoring with environmental sensing (temperature, pH, turbidity, salinity), correlates them on-device into a single feature space, and flags anomalies locally without depending on continuous high-bandwidth connectivity. The goal is a platform that's deployable on a solar-powered buoy or a fixed dock/pier mount using the same core hardware and software, reporting compact summaries over a low-bandwidth link while retaining full-resolution data for later retrieval.

## Documentation

- [DECISIONS.md](DECISIONS.md) — locked design decisions; the reference other docs are kept consistent with.
- [docs/architecture.md](docs/architecture.md) — full system architecture: sensor -> Pi -> processing -> telemetry -> storage/output flow, edge/offsite compute split, buoy/dock deployment profiles.
- [docs/hardware-spec.md](docs/hardware-spec.md) — component list and roles, interface considerations, enclosure/watertight considerations. Spec-level, no BOM yet.
- [docs/data-pipeline.md](docs/data-pipeline.md) — storage tiers, duty-cycle sampling flow, bulk retrieval process, SQLite schema sketch.
- [docs/ml-pipeline.md](docs/ml-pipeline.md) — 3-stage ML pipeline: feature extraction, unsupervised anomaly detection, future supervised classification.
- [docs/related-work.md](docs/related-work.md) — prior art and how this project's design differs.
- [docs/datasets.md](docs/datasets.md) — what data the project needs (own deployment data, public vessel/bioacoustic/multimodal datasets, pretrained models) and which pipeline stage each serves.
- [docs/pi-implementation.md](docs/pi-implementation.md) — how `edge/` runs today in mock-hardware mode, the I2C wiring/address plan for when hardware exists, and systemd deployment.

## Implementation

- `simulation/` — synthetic-hardware proof that the ML pipeline design works end-to-end (see [simulation/README.md](simulation/README.md)).
- `edge/` — the real Pi-side production software: hardware abstraction layer (mock backend usable today, real backend stubbed for once hardware exists), capture loop, calibration, telemetry payload assembly, duty-cycle scheduler. Run `python -m edge.main --once` for a one-window smoke test. See [docs/pi-implementation.md](docs/pi-implementation.md).
- `tools/power_budget.py` — standalone power budget estimator (duty cycle vs. solar/battery sizing) using placeholder component draw figures, pending a priced BOM. Run `python tools/power_budget.py --help`.
- `deploy/marine-monitor.service` — systemd unit for running `edge/` unattended once real hardware is wired.

## Diagrams

`/diagrams` — architecture/data-flow diagrams derived from the docs above: system architecture, data storage tiers, ML pipeline (each embedded in its corresponding doc), and the simulation results chart shown at the top of this README.

## Documentation map

Suggested reading order, with why each doc sits where it does:

1. [docs/architecture.md](docs/architecture.md) — start here: the whole-system picture (sensor -> Pi -> processing -> telemetry -> storage/output) everything else is a component or detail of.
2. [docs/data-pipeline.md](docs/data-pipeline.md) — the storage/data-flow half of that architecture in detail: duty-cycle sampling, storage tiers, SQLite schema.
3. [docs/ml-pipeline.md](docs/ml-pipeline.md) — what happens to the data data-pipeline.md describes once it reaches Stage 1/2 feature extraction and anomaly detection.
4. [simulation/README.md](simulation/README.md) — the ml-pipeline.md design as running code: synthetic data standing in for real hardware, exercising the same pipeline end-to-end.
5. [docs/hardware-spec.md](docs/hardware-spec.md) — the physical components the architecture and pipeline above assume, once you want to know what they'd run on.
6. [docs/related-work.md](docs/related-work.md) — how the above design choices relate to existing prior art, useful once the design itself is understood.
7. [DECISIONS.md](DECISIONS.md) — last as a reference, not reading material: the locked decisions everything above is kept consistent with, for looking something up rather than reading start to end.
