"""
Synthetic environmental sensor data generator.

Produces synthetic temperature/pH/turbidity/salinity time series matching
the environmental readings a monitoring buoy/dock unit would log once per
duty-cycle window (see docs/data-pipeline.md's `environmental_readings`
schema). Includes realistic diel (day/night) cycling in the baseline and,
optionally, a correlated multi-parameter storm/runoff anomaly event. Exists
so the anomaly-detection pipeline (docs/ml-pipeline.md) can be built and
tested before any real hardware or field data exists -- see DECISIONS.md,
project status is planning/no hardware yet.

Column names throughout (temperature_c, ph, turbidity_ntu, salinity_psu,
and their *_roc rate-of-change counterparts) intentionally match the
`environmental_readings` SQLite schema sketch in docs/data-pipeline.md, so
synthetic data here maps directly onto the real storage schema later.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Real duty-cycle sampling interval is a deployment decision (DECISIONS.md:
# "record N seconds every M minutes"). 10 minutes is a reasonable illustrative
# default for this generator -- frequent enough to resolve a diel cycle and a
# multi-day storm recovery with reasonable time resolution, infrequent enough
# to keep n_windows manageable for a demo run.
DEFAULT_WINDOW_INTERVAL_MINUTES = 10

MINUTES_PER_DAY = 24 * 60


def generate_baseline_readings(
    n_windows: int, window_interval_minutes: float = DEFAULT_WINDOW_INTERVAL_MINUTES
) -> pd.DataFrame:
    """
    Generate baseline (no-anomaly) environmental readings with realistic
    diel cycling plus small random sensor noise.

    Domain reasoning per parameter:
    - Temperature: shallow coastal water temperature tracks solar heating,
      peaking in mid-afternoon and reaching its minimum before dawn -- a
      standard diel thermal lag (water heats/cools slower than air, so the
      peak lags well past solar noon). Modeled as a sine wave with a phase
      offset so the peak falls around 14:00 local time.
    - pH: photosynthesis by algae/phytoplankton consumes dissolved CO2
      during daylight, which raises pH; respiration (no photosynthesis)
      dominates at night and adds CO2 back, lowering pH. This is a well
      documented diel pH cycle in productive coastal/estuarine water,
      modeled here as a smaller-amplitude sine wave in phase with daylight
      hours (peak near mid-afternoon, alongside peak photosynthetic
      activity and warmest water).
    - Turbidity: primarily driven by tidal mixing/resuspension and
      biological/weather events rather than a strong solar diel cycle.
      Modeled here as a flat baseline plus noise only -- tidal-driven
      variability is a real effect but out of scope for this generator's
      simplification.
    - Salinity: in the absence of a freshwater input event (rainfall,
      river discharge -- see `inject_storm_runoff_event`), coastal salinity
      is comparatively stable over a single day. Modeled as a flat baseline
      plus small noise.

    Args:
        n_windows: number of duty-cycle windows (rows) to generate.
        window_interval_minutes: minutes between successive windows,
            matching the real duty-cycle sampling interval (DECISIONS.md).

    Returns:
        DataFrame with one row per window and columns:
            window_index, elapsed_minutes,
            temperature_c, ph, turbidity_ntu, salinity_psu
    """
    window_index = np.arange(n_windows)
    elapsed_minutes = window_index * window_interval_minutes

    # fraction of the way through the current day, in [0, 1), used to phase
    # every diel cycle consistently against a 24h period
    time_of_day_frac = (elapsed_minutes % MINUTES_PER_DAY) / MINUTES_PER_DAY

    # Phase-shift so the sine peak lands at 14:00 (0.583 of the way through
    # the day) rather than at the sine function's natural peak (0.25):
    # sin(2*pi*(x - 0.25 + 0.583)) peaks when x = 0.583.
    peak_frac = 14.0 / 24.0
    diel_phase = 2 * np.pi * (time_of_day_frac - 0.25 + peak_frac)

    temperature_c = (
        18.0  # baseline coastal water temperature, deg C
        + 2.5 * np.sin(diel_phase)  # diel swing amplitude
        + np.random.normal(0, 0.15, n_windows)  # sensor/measurement noise
    )

    ph = (
        8.05  # baseline seawater pH
        + 0.08 * np.sin(diel_phase)  # smaller diel swing than temperature
        + np.random.normal(0, 0.01, n_windows)
    )

    turbidity_ntu = (
        3.0  # baseline clear-water turbidity, NTU
        + np.random.normal(0, 0.3, n_windows)
    )
    turbidity_ntu = np.clip(turbidity_ntu, 0, None)  # turbidity can't be negative

    salinity_psu = (
        35.0  # baseline open-coastal salinity, PSU
        + np.random.normal(0, 0.05, n_windows)
    )

    return pd.DataFrame(
        {
            "window_index": window_index,
            "elapsed_minutes": elapsed_minutes,
            "temperature_c": temperature_c,
            "ph": ph,
            "turbidity_ntu": turbidity_ntu,
            "salinity_psu": salinity_psu,
        }
    )


def inject_storm_runoff_event(
    readings: pd.DataFrame, onset_window: int, duration_windows: int
) -> pd.DataFrame:
    """
    Modify baseline readings to simulate a correlated storm/runoff anomaly.

    Real documented storm runoff impacts on coastal water quality show a
    consistent correlated multi-parameter pattern: turbidity rises sharply
    as rainfall mobilizes sediment and runoff carries particulates into the
    water; salinity drops at the same time because the runoff itself is
    freshwater diluting the local water column; pH becomes unstable
    (oscillates) as runoff chemistry and increased mixing disturb the
    water's normal buffering; and the whole event recovers gradually over
    roughly 2-3 days as sediment settles, tidal flushing dilutes and
    disperses the runoff, and the water column re-equilibrates. This
    correlated shape -- turbidity up and salinity down together, gradual
    multi-day recovery, unstable pH -- is the documented pattern this
    function reproduces, not an exact quantitative match to any one storm.

    Real-world timescale vs. this function's `duration_windows`: a real
    recovery runs roughly 2-3 days (~3000-4300 minutes). `duration_windows`
    is expressed in windows, not minutes, so it compresses or expands to
    that real-world timescale depending on `window_interval_minutes` used
    when `readings` was generated (e.g. at the module's default 10-minute
    interval, ~3.5 days is ~500 windows). For a demo run with few windows,
    pass a smaller `duration_windows` and treat it as a compressed stand-in
    for the same qualitative recovery shape, noted here rather than forcing
    a specific real-time duration.

    Args:
        readings: baseline DataFrame from `generate_baseline_readings`. Not
            modified in place; a modified copy is returned.
        onset_window: window index (row) at which the storm impact begins.
        duration_windows: number of windows the event (spike + recovery)
            spans, starting at `onset_window`.

    Returns:
        Modified copy of `readings` with the storm event applied to
        `turbidity_ntu`, `salinity_psu`, and `ph` for the affected windows.
    """
    out = readings.copy()
    n = len(out)
    end_window = min(onset_window + duration_windows, n)
    affected = end_window - onset_window
    if affected <= 0:
        return out

    # Peak magnitudes drawn per-event within realistic documented ranges
    # rather than fixed, so repeated simulation runs vary event severity.
    peak_turbidity_rise = np.random.uniform(20, 35)  # NTU above baseline
    peak_salinity_drop = np.random.uniform(2, 6)  # PSU below baseline, freshwater dilution
    peak_ph_drop = np.random.uniform(0.1, 0.3)  # pH units below baseline at peak instability
    ph_oscillation_amplitude = 0.1  # additional pH wobble on top of the drop

    # Rise is fast (rainfall/runoff reaching the site happens over hours,
    # much faster than the multi-day recovery), recovery is slow --
    # asymmetric envelope: linear ramp to peak over the first ~10% of the
    # event, then exponential decay back toward baseline for the rest.
    rise_windows = max(1, int(0.1 * affected))
    envelope = np.empty(affected)
    envelope[:rise_windows] = np.linspace(0, 1, rise_windows)
    decay_len = affected - rise_windows
    if decay_len > 0:
        # decay constant chosen so the envelope reaches ~5% of peak by the
        # end of duration_windows, i.e. "recovered" by the stated duration
        decay_tau = decay_len / 3.0
        decay_steps = np.arange(decay_len)
        envelope[rise_windows:] = np.exp(-decay_steps / decay_tau)

    idx = out.index[onset_window:end_window]

    # turbidity up, salinity down, together -- the correlated signature
    # documented for storm runoff, both driven by the same envelope since
    # both are direct consequences of the same freshwater/sediment pulse
    out.loc[idx, "turbidity_ntu"] = out.loc[idx, "turbidity_ntu"] + peak_turbidity_rise * envelope
    out.loc[idx, "salinity_psu"] = out.loc[idx, "salinity_psu"] - peak_salinity_drop * envelope

    # pH: net drop following the same envelope, plus an oscillation that
    # damps out along with the envelope -- representing buffering
    # instability during the event rather than a clean monotonic dip
    oscillation_cycles = 4
    progress = np.linspace(0, 1, affected)
    ph_oscillation = ph_oscillation_amplitude * np.sin(2 * np.pi * oscillation_cycles * progress)
    out.loc[idx, "ph"] = out.loc[idx, "ph"] - peak_ph_drop * envelope + ph_oscillation * envelope

    out.loc[out["turbidity_ntu"] < 0, "turbidity_ntu"] = 0  # turbidity can't go negative

    return out


def compute_rate_of_change(readings: pd.DataFrame) -> pd.DataFrame:
    """
    Add rate-of-change columns for each environmental parameter.

    Rate-of-change (this window vs. the previous window) is part of the
    feature design in docs/ml-pipeline.md: an anomaly like storm runoff is
    often more visible in how fast a parameter is moving than in its
    absolute value at a single window, since a rapid change stands out
    against a normally slow-moving baseline even before the absolute value
    leaves a "normal" range.

    Column names (temp_roc, ph_roc, turbidity_roc, salinity_roc) match the
    `environmental_readings` SQLite schema sketch in docs/data-pipeline.md.

    Args:
        readings: DataFrame with temperature_c, ph, turbidity_ntu,
            salinity_psu columns (as produced by
            `generate_baseline_readings` / `inject_storm_runoff_event`).

    Returns:
        Copy of `readings` with four additional columns appended:
        temp_roc, ph_roc, turbidity_roc, salinity_roc. The first row's
        rate-of-change is 0.0 (no prior window to compare against).
    """
    out = readings.copy()
    out["temp_roc"] = out["temperature_c"].diff().fillna(0.0)
    out["ph_roc"] = out["ph"].diff().fillna(0.0)
    out["turbidity_roc"] = out["turbidity_ntu"].diff().fillna(0.0)
    out["salinity_roc"] = out["salinity_psu"].diff().fillna(0.0)
    return out


def generate_environmental_series(
    n_windows: int, inject_anomaly_at: Optional[int] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate one full environmental time series across n_windows duty-cycle
    windows, matching the "record N seconds every M minutes" sampling model
    from DECISIONS.md, with rate-of-change columns already computed.

    This is the top-level entry point simulation scripts/tests should call:
    it builds the diel baseline, optionally injects one storm/runoff
    anomaly, computes rate-of-change, and returns both the series and
    ground-truth metadata so a downstream anomaly-detection run on this
    series can be scored against a known answer.

    Args:
        n_windows: number of duty-cycle windows (rows) to generate.
        inject_anomaly_at: window index at which to start a storm/runoff
            event, or None for a pure baseline series (negative example).

    Returns:
        (readings, metadata): `readings` is the DataFrame (window_index,
        elapsed_minutes, temperature_c, ph, turbidity_ntu, salinity_psu,
        temp_roc, ph_roc, turbidity_roc, salinity_roc). `metadata` is a
        dict of ground truth:
            {
                "n_windows": int,
                "window_interval_minutes": float,
                "anomaly_injected": bool,
                "anomaly_type": str or None,
                "onset_window": int or None,
                "duration_windows": int or None,
            }
    """
    window_interval_minutes = DEFAULT_WINDOW_INTERVAL_MINUTES
    readings = generate_baseline_readings(n_windows, window_interval_minutes)

    metadata = {
        "n_windows": n_windows,
        "window_interval_minutes": window_interval_minutes,
        "anomaly_injected": inject_anomaly_at is not None,
        "anomaly_type": None,
        "onset_window": None,
        "duration_windows": None,
    }

    if inject_anomaly_at is not None:
        # Real recovery is ~2-3 days; express that as windows at this
        # series' interval, capped to however many windows remain so a
        # short demo series still gets a (compressed) full event shape
        # rather than one truncated mid-recovery.
        real_recovery_minutes = 3 * MINUTES_PER_DAY
        target_duration_windows = int(real_recovery_minutes / window_interval_minutes)
        duration_windows = min(target_duration_windows, n_windows - inject_anomaly_at)

        readings = inject_storm_runoff_event(readings, inject_anomaly_at, duration_windows)

        metadata["anomaly_type"] = "storm_runoff"
        metadata["onset_window"] = inject_anomaly_at
        metadata["duration_windows"] = duration_windows

    readings = compute_rate_of_change(readings)

    return readings, metadata
