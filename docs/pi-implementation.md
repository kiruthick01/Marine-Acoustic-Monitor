# Pi Implementation

Status: `edge/` now contains a real, runnable implementation of this document's
predecessor design docs (not simulation-only code). It runs today in **mock
hardware mode** — no hydrophone, ADC, environmental sensors, IMU, or
telemetry module purchased yet (see [DECISIONS.md](../DECISIONS.md)) — and is
built so switching to real hardware later is a config change plus filling in
driver stubs, not a rewrite. This doc covers: how to run it now, the I2C
wiring/address plan for when hardware exists, and how it deploys on a Pi.

## Two hardware modes, one codebase

`edge/config.yaml`'s `hardware.mode` selects between:

- **`mock`** (default) — `edge/hal/mock.py` generates realistic stand-in
  sensor data (reusing `simulation/data_generator/` internally). No hardware
  required. This is what you run today.
- **`real`** — `edge/hal/real.py`'s driver classes, wired against actual
  GPIO/I2C/SPI hardware. Every method currently raises `NotImplementedError`
  with a comment on exactly what to implement — see that file once hardware
  is bought.

Everything above the HAL layer (`edge/capture_loop.py`, `edge/telemetry.py`,
`edge/calibration.py`, `edge/scheduler.py`) is written only against
`edge/hal/interfaces.py` and is identical in both modes.

## Running it now (mock mode)

```bash
pip install -r requirements.txt

# One duty-cycle window, then exit -- fastest smoke test
python -m edge.main --once

# Full duty cycle, forever (Ctrl-C to stop) -- first run auto-calibrates
python -m edge.main

# Bounded demo run (10 windows) instead of forever
python -m edge.main --max-iterations 10
```

First run with no existing `edge/output/baseline_model.joblib` automatically
runs the calibration period (`edge/config.yaml`: `calibration.calibration_windows`,
default 30 windows) before scoring anything — matching
[docs/ml-pipeline.md](ml-pipeline.md) Stage 2's "learned baseline from an
initial calibration period." Subsequent runs load the saved baseline instead
of recalibrating; pass `--calibrate` to force a fresh calibration period.

Output lands in `edge/output/`: `audio/` (Tier 1 WAV files), `db.sqlite`
(Tier 2, schema per [docs/data-pipeline.md](data-pipeline.md)),
`baseline_model.joblib` (the fitted calibration baseline).

## I2C wiring / address plan

[docs/hardware-spec.md](hardware-spec.md) flagged this as a specific risk:
*"attention to address collisions if multiple sensors share default
addresses."* The plan below is what `edge/config.yaml`'s
`hardware.i2c.env_sensor_addresses` / `imu_address` already assume — verify
each against the datasheet of whatever part the priced BOM actually lands on
(these are common defaults for the sensor *classes* named, not a specific
purchased part number yet).

| Device | Planned address | Bus | Notes |
|---|---|---|---|
| Temperature (e.g. SHT31-class) | `0x44` | I2C-1 | `0x45` is the common alternate-address-pin option if `0x44` collides |
| pH (e.g. Atlas/DFRobot EZO-class) | `0x63` | I2C-1 | EZO-class boards are software-addressable — reassign in firmware if it collides, no hardware jumper needed |
| Turbidity (analog sensor) | `0x48` | I2C-1 | Not natively I2C — this is the address of the auxiliary ADC (e.g. ADS1115) channel it's wired into, per hardware-spec.md's "sensors without native I2C output route through the same ADC" note |
| Salinity/conductivity (e.g. EZO-class) | `0x64` | I2C-1 | Same EZO family as pH — confirm the two don't ship at the same default before wiring both |
| IMU (e.g. MPU6050/9250-class) | `0x68` | I2C-1 | `0x69` is the common AD0-pin alternate if it collides with anything else on the bus |

Collision-avoidance checklist once real parts are chosen:

1. Pull each candidate part's default I2C address from its datasheet —
   don't assume the table above matches the exact SKU purchased.
2. Where two candidates collide, prefer the one with a hardware address-select
   pin/jumper or software-reassignable address (EZO-class boards, most ADCs)
   over the one that's fixed.
3. The hydrophone audio interface (per hardware-spec.md: USB codec as the
   planning default, I2S as the power-budget-driven alternative) is **not**
   on this I2C bus either way — USB and I2S are separate buses, so it never
   competes with this address space.
4. Update `edge/config.yaml`'s `hardware.i2c.*` fields to match whatever the
   final wiring turns out to be; `edge/hal/real.py`'s drivers read addresses
   from config, not from hardcoded constants, specifically so a rewire doesn't
   require a code change.

## Deployment (systemd)

`deploy/marine-monitor.service` runs `python -m edge.main` as a long-lived
service, restarting on failure — see that file for the unit definition and
the install steps in its header comment. Not usable until real hardware
exists and `hardware.mode: real` is set (mock mode has no reason to run
unattended on the actual buoy/dock Pi), but it's written now so deployment
isn't a from-scratch task once hardware arrives.

## Known gaps / next steps (software side)

- ~~**Rate-of-change continuity across restarts**~~ — done: `CaptureLoop` now
  seeds `_prev_env_reading` from the most recent `environmental_readings` row
  on construction, so `*_roc` resumes correctly after a process restart.
- ~~**Recalibration workflow**~~ — done: `edge/recalibrate.py` re-fits a
  `BaselineAnomalyDetector` from an existing database over a date/window
  range and saves it under an incremented `baseline_version` without
  overwriting the previous model file.
- **`edge/hal/real.py` driver bodies**: stubbed with `NotImplementedError` and
  a comment on what each needs; the actual implementation work is gated on the
  priced BOM (docs/hardware-spec.md) and real part datasheets.
