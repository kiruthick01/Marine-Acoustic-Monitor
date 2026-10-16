"""
Coupled atmosphere-ocean state for the simulated site, on the duty-cycle grid.

Everything the sensors and the hydrophone feel is driven from a small set of
physical drivers so the environment is *coherent* (a front produces wind ->
waves -> noise, rain -> runoff -> low salinity / high turbidity / low pH, ...)
instead of independent noise channels:

  solar geometry (lat, day-of-year, hour) -> irradiance, diurnal heating
  synoptic fronts  -> pressure drop, wind surge/veer, rain, gusts
  wind -> fetch-limited waves (lagged) -> wave resuspension, mixing
  tides (M2,S2,K1,O1 harmonics) -> water level, current, ebb freshening
  air-sea heat exchange -> SST; mixing vs. heating -> stratification
  rain -> runoff index -> salinity dilution, turbidity pulse, pH drop
  irradiance + temperature -> diel pH (photosynthesis), plankton bloom

Air is modelled only through what couples to a submerged sensor: wind, rain,
pressure, air temperature, irradiance. (Molecular-scale air dynamics have no
observable effect on a hydrophone.)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .physics import sea_state_hs, sound_speed_mackenzie


@dataclass
class Site:
    name: str = "harbor_shelf"
    lat_deg: float = 36.0
    water_depth_m: float = 20.0
    hydrophone_depth_m: float = 10.0
    mean_salinity: float = 35.0
    fetch_factor: float = 0.55          # fraction of fully-developed sea reached
    tide_scale: float = 1.0
    river_influence: float = 1.0        # how strongly rain turns into runoff
    shipping_index: float = 0.6         # distant traffic 0..1
    vessel_rate_per_day: float = 10.0   # near-field passages
    dolphin_density: float = 1.0
    whale_density: float = 1.0
    shrimp_density: float = 1.0
    fish_density: float = 1.0
    spreading_k: float = 15.0
    base_temp_offset: float = 0.0
    seabed_c: float = 1700.0


SITE_A = Site()
SITE_B = Site(
    name="open_coast", lat_deg=41.0, water_depth_m=60.0, hydrophone_depth_m=25.0,
    mean_salinity=34.4, fetch_factor=0.9, tide_scale=0.6, river_influence=0.3,
    shipping_index=0.35, vessel_rate_per_day=6.0, dolphin_density=0.7,
    whale_density=1.6, shrimp_density=0.15, fish_density=0.6, spreading_k=17.0,
    base_temp_offset=-2.5,
)

# Further sites. A, C, D are used for training (domain randomisation over depth / traffic / runoff / climate);
# B and E are held out entirely (never seen in pre-training) to measure transfer to a new deployment.
SITE_C = Site(
    name="estuary_bay", lat_deg=38.5, water_depth_m=32.0, hydrophone_depth_m=14.0,
    mean_salinity=31.5, fetch_factor=0.4, tide_scale=1.4, river_influence=1.8,
    shipping_index=0.8, vessel_rate_per_day=16.0, dolphin_density=1.3,
    whale_density=0.3, shrimp_density=0.8, fish_density=1.4, spreading_k=14.0,
    base_temp_offset=1.5,
)
SITE_D = Site(
    name="strait_channel", lat_deg=47.0, water_depth_m=48.0, hydrophone_depth_m=20.0,
    mean_salinity=33.0, fetch_factor=0.7, tide_scale=1.8, river_influence=0.8,
    shipping_index=0.9, vessel_rate_per_day=20.0, dolphin_density=0.9,
    whale_density=1.0, shrimp_density=0.05, fish_density=1.0, spreading_k=16.0,
    base_temp_offset=-5.0,
)
SITE_E = Site(
    name="offshore_shelf", lat_deg=44.0, water_depth_m=110.0, hydrophone_depth_m=40.0,
    mean_salinity=34.8, fetch_factor=1.0, tide_scale=0.4, river_influence=0.05,
    shipping_index=0.5, vessel_rate_per_day=3.0, dolphin_density=0.5,
    whale_density=2.0, shrimp_density=0.05, fish_density=0.4, spreading_k=18.0,
    base_temp_offset=-3.5,
)
SITES = {s.name: s for s in (SITE_A, SITE_B, SITE_C, SITE_D, SITE_E)}


def _ou(n, tau, sigma, rng, x0=0.0):
    """Ornstein-Uhlenbeck path with unit step."""
    a = np.exp(-1.0 / tau)
    x = np.empty(n)
    x[0] = x0
    noise = rng.standard_normal(n) * sigma * np.sqrt(1 - a * a)
    for i in range(1, n):
        x[i] = a * x[i - 1] + noise[i]
    return x


def _lowpass(x, tau, init=None):
    a = np.exp(-1.0 / tau)
    y = np.empty_like(x, dtype=float)
    y[0] = x[0] if init is None else init
    for i in range(1, len(x)):
        y[i] = a * y[i - 1] + (1 - a) * x[i]
    return y


def solar_elevation_deg(lat_deg, doy, hour_local):
    decl = 23.44 * np.sin(2 * np.pi * (284 + doy) / 365.0)
    h = (hour_local - 12.0) * 15.0
    lat, d, hh = np.radians(lat_deg), np.radians(decl), np.radians(h)
    sin_el = np.sin(lat) * np.sin(d) + np.cos(lat) * np.cos(d) * np.cos(hh)
    return np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))


def simulate_environment(
    n_windows: int,
    site: Site = SITE_A,
    interval_min: float = 10.0,
    start_doy: int = 90,
    seed: int = 0,
    front_rate_per_day: float = 0.18,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by window with true (noise-free) drivers and
    ocean state. Sensor imperfections are applied separately (scenario.py).
    """
    rng = np.random.default_rng(seed)
    n = n_windows
    step = interval_min
    t_min = np.arange(n) * step
    t_day = t_min / 1440.0
    hour = (t_min / 60.0) % 24.0
    doy = start_doy + t_day
    steps_per_day = 1440.0 / step

    elev = solar_elevation_deg(site.lat_deg, doy, hour)
    irr = np.clip(np.sin(np.radians(elev)), 0, None)           # 0..1 clear-sky
    season = np.sin(2 * np.pi * (doy - 80) / 365.0)            # -1 winter .. +1 summer

    # ---- synoptic fronts: Poisson arrivals, each a lagged pulse in several vars
    wind_boost = np.zeros(n)
    rain_env = np.zeros(n)
    dir_shift = np.zeros(n)
    fronts = []
    day = 0.0
    idx = np.arange(n)
    while True:
        day += rng.exponential(1.0 / front_rate_per_day)
        if day >= t_day[-1]:
            break
        dur_h = rng.uniform(8, 30)
        peak_wind = rng.uniform(4, 11)
        has_rain = rng.random() < 0.65
        rain_peak = rng.gamma(2.0, 2.0) if has_rain else 0.0
        c = int(day * steps_per_day)
        w = dur_h * 60 / step
        env = np.exp(-0.5 * ((idx - c) / (w / 2.2)) ** 2)
        wind_boost += peak_wind * env
        rain_env += rain_peak * np.exp(-0.5 * ((idx - c - 0.15 * w) / (w / 3.5)) ** 2)
        dir_shift += rng.uniform(-90, 90) * (idx > c)
        fronts.append({"day": float(day), "duration_h": float(dur_h), "peak_wind_ms": float(peak_wind),
                       "peak_rain_mm_h": float(rain_peak)})

    # ---- wind: OU background + sea breeze + fronts
    wind_base = 4.0 + _ou(n, 1.5 * steps_per_day, 1.8, rng)
    sea_breeze = 1.4 * irr * (1 + 0.3 * season)
    wind10 = np.clip(wind_base + sea_breeze + wind_boost + rng.normal(0, 0.35, n), 0.2, None)
    gust = wind10 * (1 + np.abs(rng.normal(0, 0.12, n)))
    wind_dir = (200 + np.cumsum(rng.normal(0, 2.0, n)) * 0.3 + dir_shift) % 360

    # ---- pressure and rain
    pressure = 1013 + 0.5 * np.sin(2 * np.pi * hour / 12.0) + _ou(n, 3 * steps_per_day, 4.0, rng) - 1.1 * wind_boost
    rain = np.clip(rain_env * (0.6 + 0.8 * rng.random(n)) - 0.4, 0, None)
    rain = rain + (rng.random(n) < 0.004) * rng.exponential(0.5, n)

    # ---- air temperature
    air_t = 14 + 9 * season + site.base_temp_offset + 4.0 * np.sin(2 * np.pi * (hour - 9) / 24.0) \
        + _ou(n, 2 * steps_per_day, 1.5, rng) - 0.15 * wind_boost

    # ---- waves (lagged response to wind, fetch-limited)
    hs_eq = np.minimum(sea_state_hs(wind10) * site.fetch_factor, 6.0)
    hs = np.maximum(_lowpass(hs_eq, 6 * 60 / step), 0.05)
    wave_period = 3.5 + 0.72 * np.sqrt(hs * 9.81)

    # ---- tides: 4 harmonics, spring/neap emerges from M2+S2 beat
    th = t_min / 60.0 + rng.uniform(0, 12)
    h_tide = site.tide_scale * (
        0.80 * np.cos(2 * np.pi * th / 12.4206)
        + 0.25 * np.cos(2 * np.pi * th / 12.0 + 0.6)
        + 0.20 * np.cos(2 * np.pi * th / 23.9345 + 1.1)
        + 0.15 * np.cos(2 * np.pi * th / 25.8193 + 2.0)
    )
    dh_dt = np.gradient(h_tide, step / 60.0)                   # m/h
    current = 0.55 * dh_dt / (np.max(np.abs(dh_dt)) + 1e-9) * (1.0 if site.water_depth_m < 40 else 0.6)
    flow = np.abs(current)
    ebb = np.clip(-current, 0, None)                            # seaward flow brings river water

    # ---- sea-surface temperature: relaxation toward heating-dependent equilibrium
    t_eq_season = 14.5 + 6.5 * season + site.base_temp_offset
    t_eq = t_eq_season + 1.2 * irr + 0.25 * (air_t - t_eq_season)
    sst = np.empty(n)
    sst[0] = t_eq_season[0]
    mixing = 1.0 + 0.04 * np.maximum(wind10 - 8, 0)
    for i in range(1, n):
        sst[i] = sst[i - 1] + (t_eq[i] - sst[i - 1]) * (mixing[i] / (4.0 * steps_per_day))
    sst = sst + 0.35 * np.sin(2 * np.pi * (hour - 15) / 24.0) * (1 - np.minimum(wind10 / 12, 1))

    # ---- stratification collapses under strong wind, builds under heating
    strat_eq = np.clip(0.5 + 0.5 * season + 0.3 * irr, 0, 1)
    strat = np.empty(n)
    strat[0] = strat_eq[0] * 0.8
    for i in range(1, n):
        strat[i] = strat[i - 1] + (strat_eq[i] - strat[i - 1]) / (3 * steps_per_day) \
            - 0.004 * max(wind10[i] - 8, 0) * strat[i - 1]
        strat[i] = min(max(strat[i], 0.0), 1.0)
    t_deep = 11.0 + 3.0 * season + site.base_temp_offset * 0.5
    z_ratio = np.clip(site.hydrophone_depth_m / site.water_depth_m, 0.05, 1.0)
    temp_h = sst - strat * (sst - t_deep) * (0.35 + 0.65 * z_ratio)

    # ---- runoff index and everything it touches
    runoff_in = site.river_influence * _lowpass(rain, 3 * 60 / step, 0.0) * (1 + 0.3 * rng.random(n))
    runoff = np.clip(_lowpass(runoff_in, 18 * 60 / step, 0.0) / 4.0, 0, 1.5)
    salinity = site.mean_salinity - 3.2 * runoff - 0.35 * ebb * site.river_influence \
        + 0.1 * np.sin(2 * np.pi * th / 12.42) + 0.05 * season - 0.1 * strat
    resusp_wave = 0.9 * np.maximum(hs - 0.4, 0) ** 2 * (20.0 / site.water_depth_m)
    resusp_tide = 4.0 * np.maximum(flow - 0.25, 0) ** 2
    bloom = np.clip(np.exp(-0.5 * ((doy - 100.0) / 18.0) ** 2) * (1 + 0.25 * np.sin(doy)), 0, 1.4)
    turbid_pulse = 6.0 * _lowpass(runoff_in / 4.0, 6 * 60 / step, 0.0)
    settling = _lowpass(resusp_wave + resusp_tide + turbid_pulse, 10 * 60 / step, 0.0)
    turbidity = 1.6 + 1.4 * bloom + settling + 0.4 * np.abs(_ou(n, 5, 0.3, rng))
    sun_lag = _lowpass(irr, 4 * 60 / step, 0.0)
    ph = 8.06 - 0.012 * (temp_h - 15.0) + 0.045 * sun_lag - 0.13 * runoff + 0.03 * bloom - 0.02 * strat
    o2 = 8.0 - 0.18 * (temp_h - 10) + 0.5 * sun_lag * bloom - 1.2 * runoff

    df = pd.DataFrame({
        "window_index": np.arange(n),
        "elapsed_minutes": t_min,
        "doy": doy, "hour": hour,
        "sun_elevation_deg": elev, "irradiance": irr,
        "air_temp_c": air_t, "pressure_hpa": pressure,
        "wind_ms": wind10, "gust_ms": gust, "wind_dir_deg": wind_dir,
        "rain_mm_h": rain, "hs_m": hs, "wave_period_s": wave_period,
        "tide_m": h_tide, "current_ms": current, "flow_ms": flow,
        "sst_c": sst, "stratification": strat, "temperature_c": temp_h,
        "salinity_psu": salinity, "turbidity_ntu": turbidity, "ph": ph,
        "dissolved_o2_mg_l": o2, "runoff_index": runoff, "bloom": bloom,
        "sound_speed_ms": sound_speed_mackenzie(temp_h, salinity, site.hydrophone_depth_m),
    })
    df.attrs["fronts"] = fronts
    return df


def sound_speed_profile(depths_m, sst, t_deep, strat, salinity, water_depth_m):
    """Temperature/sound-speed vs. depth with a thermocline of strength `strat`."""
    z = np.asarray(depths_m, dtype=float)
    z_th = 0.45 * water_depth_m
    sig = 1.0 / (1.0 + np.exp(-(z - z_th) / (0.08 * water_depth_m)))
    t = sst - strat * (sst - t_deep) * sig - (1 - strat) * 0.05 * (sst - t_deep) * (z / water_depth_m)
    return t, sound_speed_mackenzie(t, salinity, z)
