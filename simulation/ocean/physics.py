"""
Underwater-acoustics physics used by the ocean simulator.

Engineering-grade parametric models (Mackenzie, Ainslie-McColm, Wenz/Coates,
Ross, practical spreading) -- NOT a wave-equation solver. They are the
standard formulas used for sonar/passive-acoustic link budgets, so the
simulated levels, spectra and environment couplings are physically
meaningful, but absolute accuracy at a specific real site would need
site-specific propagation modelling (normal modes / ray tracing).

Units: pressure in micro-Pascals (uPa), levels in dB re 1 uPa (or re 1 uPa^2/Hz
for spectral densities), distances in metres, frequency in Hz unless noted.
"""

import numpy as np

KNOT_MS = 0.514444


def sound_speed_mackenzie(temp_c, salinity_psu, depth_m):
    """Mackenzie (1981) nine-term sound speed (m/s)."""
    T, S, D = temp_c, salinity_psu, depth_m
    return (
        1448.96 + 4.591 * T - 5.304e-2 * T**2 + 2.374e-4 * T**3
        + 1.340 * (S - 35) + 1.630e-2 * D + 1.675e-7 * D**2
        - 1.025e-2 * T * (S - 35) - 7.139e-13 * T * D**3
    )


def absorption_db_per_km(freq_hz, temp_c, salinity_psu, ph, depth_m):
    """
    Ainslie & McColm (1998) seawater absorption (dB/km). Depends on pH (boric
    acid relaxation), temperature, salinity and depth, so a changing
    environment changes how far sound carries.
    """
    f = np.asarray(freq_hz, dtype=float) / 1000.0  # kHz
    T, S, z = temp_c, salinity_psu, depth_m / 1000.0
    f1 = 0.78 * np.sqrt(S / 35.0) * np.exp(T / 26.0)
    f2 = 42.0 * np.exp(T / 17.0)
    a = (
        0.106 * (f1 * f**2 / (f1**2 + f**2)) * np.exp((ph - 8.0) / 0.56)
        + 0.52 * (1 + T / 43.0) * (S / 35.0) * (f2 * f**2 / (f2**2 + f**2)) * np.exp(-z / 6.0)
        + 0.00049 * f**2 * np.exp(-(T / 27.0 + z / 17.0))
    )
    return a


def waveguide_cutoff_hz(water_depth_m, c_water=1500.0, c_bottom=1700.0):
    """Lowest frequency a shallow-water waveguide supports (first mode)."""
    return c_water / (4.0 * water_depth_m * np.sqrt(1.0 - (c_water / c_bottom) ** 2))


def transmission_loss_db(
    freq_hz, range_m, temp_c, salinity_psu, ph, depth_m, water_depth_m,
    spreading_k=15.0, stratification=0.0,
):
    """
    One-way transmission loss (dB) = practical spreading + absorption
    + shallow-water low-frequency cutoff + a stratification penalty.

    spreading_k: 20 spherical, 10 cylindrical, 15 practical shallow water.
    Below the waveguide cutoff the loss rises steeply (sound can't propagate
    in a layer thinner than ~a quarter wavelength).
    stratification (0..1): summer thermocline bends sound toward the bottom
    -> extra loss growing with range (heuristic, ~4 dB per range decade).
    """
    f = np.asarray(freq_hz, dtype=float)
    r = np.maximum(np.asarray(range_m, dtype=float), 1.0)
    spread = spreading_k * np.log10(r)
    absorb = absorption_db_per_km(f, temp_c, salinity_psu, ph, depth_m) * r / 1000.0
    fc = waveguide_cutoff_hz(water_depth_m)
    below = np.maximum(np.log10(fc / np.maximum(f, 1.0)), 0.0)
    cutoff = 30.0 * below * np.minimum(np.log10(r) / 2.0, 1.5) + 6.0 * below
    strat = 4.0 * stratification * np.maximum(np.log10(r / 100.0), 0.0)
    return spread + absorb + cutoff + strat


def wenz_psd_db(freq_hz, wind_ms, shipping=0.5, rain_mm_h=0.0, flow_ms=0.0):
    """
    Ambient noise PSD (dB re 1 uPa^2/Hz) at the hydrophone: Coates/Wenz
    turbulence + distant shipping + wind-wave + thermal, plus a rain term
    (Ma & Nystuen style, simplified) and mooring flow noise.

    shipping: 0 (none) .. 1 (heavy) distant-traffic index.
    """
    f = np.maximum(np.asarray(freq_hz, dtype=float), 1.0)
    fk = f / 1000.0
    turb = 17 - 30 * np.log10(fk)
    ship = 40 + 20 * (shipping - 0.5) + 26 * np.log10(fk) - 60 * np.log10(fk + 0.03)
    wind = 50 + 7.5 * np.sqrt(max(wind_ms, 0.0)) + 20 * np.log10(fk) - 40 * np.log10(fk + 0.4)
    therm = -15 + 20 * np.log10(fk)
    power = 10 ** (turb / 10) + 10 ** (ship / 10) + 10 ** (wind / 10) + 10 ** (therm / 10)
    if rain_mm_h > 0.05:
        l5 = 52.0 + 15.0 * np.log10(rain_mm_h + 1e-3)  # level at 5 kHz
        shape = np.where(fk >= 5.0, -20 * np.log10(fk / 5.0), -12 * np.log10(5.0 / fk))
        power = power + 10 ** ((l5 + shape) / 10)
    if flow_ms > 0.02:
        # pseudo-sound from flow past the mooring/hydrophone: steep red spectrum
        flow = 68 + 40 * np.log10(flow_ms / 0.5) - 25 * np.log10(np.maximum(f, 5.0) / 20.0)
        power = power + 10 ** (flow / 10)
    return 10 * np.log10(power)


def ross_source_level_db(speed_kn, length_m):
    """Ross (1976) broadband vessel source level (dB re 1 uPa @ 1 m)."""
    return 175.0 + 60.0 * np.log10(max(speed_kn, 1.0) / 25.0) + 20.0 * np.log10(length_m / 300.0)


def vessel_broadband_psd_db(freq_hz, total_sl_db):
    """
    Source PSD (dB re 1 uPa^2/Hz @ 1 m) with Ross-like shape: flat-ish below
    ~100 Hz then -6 dB/octave, normalised numerically so the 10 Hz-10 kHz
    integral equals the requested broadband source level.
    """
    f = np.maximum(np.asarray(freq_hz, dtype=float), 1.0)
    shape = np.where(f < 100.0, 10 * np.log10(f / 100.0 + 0.05) * 0.5, -20 * np.log10(f / 100.0))
    grid = np.linspace(10, 10000, 4000)
    gshape = np.where(grid < 100.0, 10 * np.log10(grid / 100.0 + 0.05) * 0.5, -20 * np.log10(grid / 100.0))
    total = np.trapezoid(10 ** (gshape / 10), grid)
    offset = total_sl_db - 10 * np.log10(total)
    return shape + offset


def sea_state_hs(wind_ms):
    """Fully-developed significant wave height (m), Pierson-Moskowitz."""
    return 0.0217 * np.asarray(wind_ms, dtype=float) ** 2


def shape_noise(psd_db, freqs_hz, n_samples, sample_rate, rng, f_min=10.0):
    """
    Gaussian noise whose one-sided PSD is `psd_db(freqs)` (dB re uPa^2/Hz),
    returned in uPa. White noise of one-sided density S has variance S*fs/2.
    """
    white = rng.standard_normal(n_samples)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    s_lin = 10 ** (np.asarray(psd_db(f)) / 10.0)
    spec *= np.sqrt(s_lin * (sample_rate / 2.0))
    spec[f < f_min] = 0.0  # hydrophone/preamp high-pass
    return np.fft.irfft(spec, n=n_samples)


def db_to_upa(level_db):
    return 10 ** (np.asarray(level_db) / 20.0)
