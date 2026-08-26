# Hardware Specification

Status: planning/architecture phase. No hardware purchased. This document is spec-level — component roles and interface requirements, not a bill of materials. Consistent with [DECISIONS.md](../DECISIONS.md) and [architecture.md](architecture.md).

## Component list and role

**Hydrophone (acoustic sensor)**
Primary sensing input. Captures underwater acoustic signal in scheduled duty-cycle windows (record N seconds every M minutes). Feeds the on-device FFT + feature extraction stage (MFCCs, spectral centroid, ZCR, RMS energy, spectral flatness) and, at full resolution, the local raw-audio store.

**Environmental sensors: temperature, pH, turbidity, salinity**
Secondary sensing inputs, sampled alongside each acoustic window. Values (plus rate-of-change) are normalized and concatenated with acoustic features into the joint feature vector used for anomaly detection. Also logged independently as structured environmental readings in SQLite, and included in the compact telemetry payload.

**IMU**
Provides orientation/motion data. On a buoy, distinguishes wave/mooring motion from genuine environmental or acoustic anomalies, and supports system health logging (e.g. detecting tilt/capsize risk, impact events). Not part of the core acoustic anomaly-detection feature vector unless later shown useful; treated as system-health/context sensor.

**Raspberry Pi (edge compute)**
Central compute node. Runs capture scheduling, FFT/feature extraction, unsupervised anomaly detection, local storage writes (flat audio files + SQLite WAL), and telemetry payload assembly/transmission. Only compute element in the system — no other onboard processor.

**Telemetry module (LoRa or low-bandwidth cellular)**
Transmits compact payloads only: feature summaries, environmental readings, anomaly alerts, on the same duty cycle as sampling. Does not carry raw audio. Choice between LoRa and low-bandwidth cellular is a deployment-specific decision (range, existing gateway infrastructure, cellular coverage at site), not fixed at the architecture level.

**Power system (solar panel + battery for buoy; solar+battery or shore power for dock)**
Sized to sustain duty-cycle sampling, edge compute bursts, and telemetry transmission between charge cycles. See [architecture.md](architecture.md) for the deployment-profile power notes.

## Key interface considerations

**Hydrophone / ADC**
Hydrophone output is analog and low-level, requiring a preamp and a dedicated ADC ahead of the Pi — the Pi has no native analog audio input. The ADC must support a sample rate covering the passband of interest (20 Hz–12 kHz per [ml-pipeline.md](ml-pipeline.md) Stage 0, i.e. ≥24 kHz sampling; the pipeline default is 22.05 kHz) and sufficient bit depth to preserve dynamic range across quiet ambient and loud transient events (16-bit minimum; 24-bit preferred for headroom without gain retuning).

Two candidate interfaces to the Pi, decided at BOM time:

- **USB audio codec** — the proven path: the peer-reviewed Pi recorder this project cites ([related-work.md](related-work.md), Caldas-Morgan et al. 2015) digitized its hydrophone through a USB codec, and USB audio class devices need no custom kernel/driver work. Costs: a few hundred mW of additional continuous draw while active, and codec choice constrains sample-rate/bit-depth options to consumer-audio profiles — both acceptable in this design, since capture is duty-cycled (the codec is powered only during wake windows) and consumer profiles (up to 96 kHz/24-bit) comfortably cover the 12 kHz passband.
- **I2S ADC** — lower power and full control of sample rate/bit depth, at the cost of device-tree/driver integration work and fewer off-the-shelf marine-grade options. Worth choosing only if the power budget (`tools/power_budget.py`, once fed real BOM figures) shows the USB codec's active-window draw materially affects solar/battery sizing.

Default assumption for planning: USB codec, on the strength of the published precedent — revisit only if the priced power budget argues otherwise. (Latency is not a factor either way: this is a duty-cycle recorder, not a real-time-response system.) SPI ADCs remain relevant for the auxiliary *low-rate* analog inputs (see environmental sensor bus below), not for audio-rate capture.

**Environmental sensor bus**
Temperature, pH, turbidity, and salinity sensors are typically I2C or analog-with-ADC devices. I2C is preferred where available for multi-sensor sharing over one bus (fewer GPIO pins, simpler wiring), with attention to address collisions if multiple sensors share default addresses. Sensors without native I2C output route through the same ADC used for auxiliary analog inputs, or a separate low-channel-count ADC if the hydrophone ADC is fully committed.

**IMU bus**
IMUs are standard I2C or SPI devices; low pin/power overhead, straightforward to share the same I2C bus as environmental sensors if address space allows.

**Power draw implications of duty-cycle sampling**
System is not continuously active at full draw — draw profile has two states: (1) sleep/idle between wake windows, minimal draw, sensors and compute powered down or in low-power mode; (2) active window, higher draw during capture + FFT/feature extraction + anomaly detection + telemetry transmission. Telemetry transmission is typically the highest instantaneous draw event in the cycle (especially cellular over LoRa). Duty cycle (N seconds every M minutes) is the primary lever for average power consumption and must be balanced against solar charge input and battery capacity for the buoy profile; sizing itself belongs in a future power budget, not this document.

## Enclosure / watertight considerations (buoy reference platform)

- Pi, ADC, telemetry module, and power electronics require a sealed, waterproof enclosure rated for continuous partial or full submersion exposure (splash, spray, and wave wash at minimum; full submersion resistance if mounted low on the buoy).
- Hydrophone and environmental sensors are the only elements intentionally in direct water contact — their cabling/connectors into the sealed electronics enclosure are the primary watertight risk point and require waterproof (marine-rated) connectors or a properly potted cable entry, not just a sealed box.
- Enclosure must accommodate thermal dissipation for the Pi under active-window load without compromising the seal (passive conduction to enclosure body, not fan-based active cooling).
- Biofouling on the hydrophone and environmental sensor surfaces will affect data quality over time; enclosure/mounting design should keep sensing surfaces accessible for cleaning at maintenance visits, without requiring the sealed electronics enclosure to be opened.
- Antenna (telemetry module) either mounted external to the sealed enclosure with a waterproof bulkhead feedthrough, or the enclosure material/design must not attenuate the telemetry signal if the antenna stays internal.
- Solar panel is inherently external and exposed; its wiring into the sealed enclosure is a second watertight cable-entry point distinct from the sensor entries above.
- Dock/pier profile has materially relaxed watertight requirements (typically above waterline, less wave exposure) but should use the same enclosure design as the buoy profile to keep hardware interchangeable across deployment profiles, per [architecture.md](architecture.md).
