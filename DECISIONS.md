# Design Decisions

Locked design decisions. Running reference — other docs in this repo must stay consistent with these.

## Deployment

Architecture is deployment-agnostic at the core; moored/floating solar+battery buoy is the primary reference platform. Fixed dock/pier mount is a documented lower-effort alternative using the same electronics.

## Telemetry

Hybrid tiered model. Edge (Pi) always runs FFT + feature extraction + anomaly detection locally. Low-bandwidth link (LoRa or low-bandwidth cellular) sends only compact payloads: feature summaries, environmental readings, anomaly alerts, on a duty cycle. Full-resolution raw audio + sensor logs stay on local storage (SD/SSD), retrieved during maintenance visits or opportunistic high-bandwidth sync.

## Sampling

Scheduled duty-cycle sampling (record N seconds every M minutes) as baseline. Triggered wake-on-threshold sampling documented as future enhancement, not built now.

## Processing cadence

Near-real-time within each wake window (capture -> FFT/feature extraction -> sleep). Heavier/batch analysis, including ML, happens offline during bulk data retrieval.

## Storage

Raw audio as flat WAV/FLAC files on local storage, indexed by timestamp filename. Structured data (feature vectors, environmental readings, telemetry/system health log, anomaly flags) in SQLite (WAL mode) on the Pi. SQLite rows reference audio filenames rather than embedding audio. Central time-series DB (TimescaleDB/InfluxDB) noted as future multi-buoy scaling path, not built now.

## ML pipeline, 3 stages

1. On-device feature extraction via Librosa/SciPy -- MFCCs, spectral centroid, zero-crossing rate, RMS energy, spectral flatness -- concatenated with normalized environmental sensor features including rate-of-change, into one joint acoustic+environmental feature vector per window.
2. Unsupervised anomaly detection (Isolation Forest or autoencoder reconstruction error) against a learned baseline from an initial calibration period -- viable from day one, no labels needed.
3. Supervised classification (random forest or small NN) as explicit future work, gated on accumulating labeled/reviewed anomalies or transfer learning from public bioacoustic datasets; training happens offline, only lightweight inference artifact deployed to Pi.

## Project status

No hardware purchased yet. Edge software, simulation pipeline, and tooling are implemented and tested against `edge/hal/mock.py`; hardware-blocked work (real HAL drivers, field calibration, filter/ADC tuning against real recordings, Stage 3 supervised classification) remains, per `docs/pi-implementation.md` and `docs/software-roadmap.md`. Public framing must stay honest about this.

## Related work to cite

Not claiming novelty on the base pattern:

- CORMA project (OGS Italy)
- Orcasound open-source PAM on Raspberry Pi
- University of Sao Paulo Pi-based recorder (PLOS ONE)
- Low-cost DIY hydrophone designs (CoPiDi etc.)
- Sethi et al. unsupervised eco-acoustic anomaly detection (bioRxiv)
- Multimodal underwater benchmark dataset (Ratilal et al. 2022)
