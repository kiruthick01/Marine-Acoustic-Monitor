# Software Roadmap (pre-hardware)

Tasks below are everything left on the software side that does **not** need
real hardware in hand — all buildable/testable against `edge/hal/mock.py`
today. Each has a ready-to-paste prompt for a fresh Claude Code session in
this repo (each prompt is self-contained; Claude Code won't have this
conversation's context).

Recommended order: 1 → 2 → 3 → 4 → 7 → 5 → 6 → 8. Do 1 first no matter what
— everything else is easier to trust once you know the test suite actually
passes.

Explicitly **excluded** here (hardware-blocked, do later — see
`docs/pi-implementation.md`): writing `edge/hal/real.py`'s real drivers,
tuning the bandpass filter / ADC scaling against real recordings, the field
calibration procedure, and Stage 3 supervised classification (needs labeled
real-world anomalies, which don't exist until the device is deployed).

---

## 1. Verify and fix the test suite (Done)

**Why:** the sandbox this was built in had no internet access to install
scipy/scikit-learn/librosa/pytest, so `edge/capture_loop.py`,
`edge/hal/mock.py`, `edge/calibration.py`, `edge/scheduler.py`, and
`edge/main.py` were only verified by hand against the real function
signatures, never actually executed. `edge/config.py`, `edge/telemetry.py`,
and `simulation/pipeline/storage.py` *were* executed directly and had two
real bugs caught and fixed (telemetry payload exceeding its own byte budget,
`init_db()` crashing on a fresh checkout). The rest needs a real run to be
trusted.

**Prompt:**

```
Run `pip install -r requirements.txt` then `pytest simulation edge -v` in
this repo. Fix any failing tests. Before changing a test's expectations to
make it pass, first work out whether the bug is in the test or in the
implementation it's testing -- check the test's assumptions against
DECISIONS.md, docs/data-pipeline.md, and docs/ml-pipeline.md if unclear.
Report a summary of what failed and what you changed.
```

---

## 2. Make the capture loop resilient to sensor/telemetry failures (Done)

**Why:** right now, if any single HAL call in `CaptureLoop.run_one_window()`
raises an exception (a sensor read fails, telemetry send throws, the SD card
is briefly unavailable), the exception propagates all the way up and kills
the whole process. On an unattended buoy that means one bad reading silently
ends all future monitoring until someone makes a boat trip.

**Prompt:**

```
In edge/capture_loop.py, CaptureLoop.run_one_window() currently has no error
handling -- an exception from any HAL call (hydrophone.capture(),
env_sensors.read(), imu.read(), power.read(), telemetry.send()) propagates
uncaught. Make this resilient:

1. Wrap each HAL call so a failure in one sensor doesn't prevent the rest of
   the window from completing where possible (e.g. env sensor failure
   shouldn't stop audio capture and storage from still happening).
2. If a genuinely unrecoverable failure occurs, run_one_window() should
   raise a single well-defined exception type (not let arbitrary exceptions
   through) so callers can decide how to handle it.
3. In edge/scheduler.py, DutyCycleScheduler.run_forever() should catch that
   exception per iteration, log it, and continue to the next duty-cycle
   wake window instead of crashing the whole process.
4. Add a way for edge/hal/mock.py's mock classes to simulate intermittent
   failures (e.g. a failure_rate parameter like MockTelemetryLink already
   has), and add tests proving a single failed window doesn't stop
   subsequent windows from running.

Keep this consistent with docs/data-pipeline.md's duty-cycle model -- a
failed window should be logged and skipped, not retried mid-cycle.
```

---

## 3. Replace `print()` with real logging (Done)

**Why:** `edge/main.py`, `edge/calibration.py`, and `edge/scheduler.py`
currently just `print()`. That's fine for an interactive `--once` run, but
useless for a device running unattended for weeks — there's no way to see
what happened after the fact without a live terminal attached.

**Prompt:**

```
Replace the print() calls in edge/main.py, edge/calibration.py, and
edge/scheduler.py with Python's logging module. Add a rotating file handler
(logging.handlers.RotatingFileHandler) writing to a log file under
config.storage's output directory (add a log_path field to StorageConfig in
edge/config.py, default "edge/output/edge.log"), plus a console handler so
interactive runs (--once, etc.) still see output. Make the log level
configurable via edge/config.yaml. Keep the existing human-readable
per-window summary line format from edge/main.py's _print_window_summary().
Update or add tests as needed.
```

---

## 4. Fix rate-of-change continuity across process restarts (Done)

**Why:** `CaptureLoop` tracks the previous environmental reading in memory
only (`self._prev_env_reading`). Every process restart (crash, power blip,
manual restart) silently resets rate-of-change to 0 for one window, which
quietly degrades a real feature the anomaly detector relies on. Documented
as a known gap in `docs/pi-implementation.md`.

**Prompt:**

```
edge/capture_loop.py's CaptureLoop currently initializes
self._prev_env_reading = None and only ever updates it in memory, so it
resets on every process restart instead of resuming from the last stored
reading. Fix this: on CaptureLoop construction (or via an explicit resume()
method called once from edge/main.py's build_app() before the first
window), query the most recent row from the environmental_readings table
(joined through captures for correct timestamp ordering) in the already-
initialized db_conn, and seed self._prev_env_reading from it. Fall back to
None only if the table is empty (first-ever run). Add a test proving a new
CaptureLoop instance pointed at a DB with existing rows resumes correctly
(next window's roc reflects the last stored reading, not 0).
```

---

## 5. Build the recalibration workflow (Done)

**Why:** `docs/ml-pipeline.md` calls for recalibrating the anomaly baseline
after a seasonal shift, or once real flagged anomalies get manually
reviewed. `edge/calibration.py` only handles the very first calibration —
there's no way yet to re-fit on top of retrieved data or version the
baseline forward.

**Prompt:**

```
Build a recalibration workflow on top of edge/calibration.py's existing
BaselineAnomalyDetector-fitting logic. Add a script (edge/recalibrate.py or
a function in edge/calibration.py, your call) that:

1. Reads feature_vectors + environmental_readings rows from an existing
   edge/output/db.sqlite for a given date/window range (CLI args for
   start/end timestamp or window count).
2. Reconstructs the joint feature vectors from those rows (matching the
   shape simulation/pipeline/feature_extraction.build_joint_feature_vector()
   produces -- check how feature_vectors.mfcc is stored as a JSON blob in
   simulation/pipeline/storage.py's insert_window_record()).
3. Fits a new BaselineAnomalyDetector on that data.
4. Saves it under a NEW baseline_version (increment from
   config.calibration.baseline_version, e.g. v1 -> v2) without overwriting
   the previous model file, and updates edge/config.yaml accordingly (or
   prints the value the operator should set).

Add tests using a small mock-generated dataset (run a few CaptureLoop
windows against a tmp_path DB first, then recalibrate against it).
```

---

## 6. Bulk retrieval / maintenance export tool (Done)

**Why:** `docs/data-pipeline.md`'s design assumes someone can, at a
maintenance visit, pull Tier 1 (audio files) and Tier 2 (SQLite) off the
device in one go. No tooling for that exists yet, and it's fully testable
now against mock-generated `edge/output/` data.

**Prompt:**

```
Build tools/export_for_retrieval.py: a script that packages up
edge/output/audio/ and edge/output/db.sqlite (optionally filtered to a
date/window range via CLI args) into a single archive (zip or tar.gz) for a
maintenance-visit retrieval, per docs/data-pipeline.md's "Bulk retrieval /
sync process" section. Include a manifest file in the archive (JSON or
text) summarizing: window count, date range covered, number of anomaly
flags, total audio size. This should be safe to run while the capture loop
is still writing to the same SQLite file (WAL mode already supports
concurrent reads -- see simulation/pipeline/storage.py's init_db()). Add
tests: run a few CaptureLoop windows against a tmp_path output dir, then
verify the export produces a correct archive and manifest.
```

---

## 7. Config validation / fail-fast checks

**Why:** a nonsensical `edge/config.yaml` (e.g. `window_duration_s` longer
than the wake interval, an invalid `hardware.mode`, colliding I2C
addresses) currently only surfaces as a confusing crash somewhere deep in
`capture_loop.py`, not a clear error at startup.

**Prompt:**

```
Add a validate() method to EdgeConfig in edge/config.py, called at the end
of load_config(). Check:
- duty_cycle.window_duration_s < duty_cycle.window_interval_minutes * 60
- calibration.calibration_windows >= 5 (arbitrary sane minimum)
- hardware.mode in {"mock", "real"}
- hardware.telemetry.type in {"lora", "cellular"}
- hardware.i2c.env_sensor_addresses and hardware.i2c.imu_address are all
  valid hex strings (e.g. "0x44") and none of them collide with each other
Collect every violation found (not just the first) and raise one ValueError
listing all of them clearly. Add a test per validation rule, including one
proving a config with a real collision (two addresses in
env_sensor_addresses set to the same value) is caught.
```

---

## 8. CI so this doesn't regress silently

**Why:** there's currently no automated check that a future change doesn't
break the pipeline.

**Prompt:**

```
Add a GitHub Actions workflow at .github/workflows/tests.yml that runs on
push and pull_request to main: checks out the repo, sets up Python (matrix:
3.10 and 3.11), runs `pip install -r requirements.txt`, then
`pytest simulation edge`. Keep it minimal -- no deployment steps, just the
test gate.
```

---

## Checklist

- [x] 1. Verify and fix the test suite
- [x] 2. Resilient capture loop (sensor/telemetry failure handling)
- [x] 3. Real logging (replace print, rotating log file)
- [x] 4. Rate-of-change continuity across restarts
- [x] 5. Recalibration workflow
- [x] 6. Bulk retrieval / maintenance export tool
- [ ] 7. Config validation / fail-fast checks
- [ ] 8. CI test gate
