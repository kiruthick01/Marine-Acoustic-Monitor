import numpy as np

from simulation.ocean import physics as ph
from simulation.ocean.weather import SITE_A, simulate_environment


def test_sound_speed_reference_value():
    assert abs(ph.sound_speed_mackenzie(15, 35, 10) - 1506.9) < 0.5


def test_absorption_increases_with_frequency_and_falls_with_ph():
    a = ph.absorption_db_per_km([100, 1000, 10000], 15, 35, 8.05, 10)
    assert a[0] < a[1] < a[2]
    assert ph.absorption_db_per_km(1000, 15, 35, 7.9, 10) < ph.absorption_db_per_km(1000, 15, 35, 8.2, 10)


def test_transmission_loss_grows_with_range_and_below_cutoff():
    tl = ph.transmission_loss_db(1000, [100, 1000], 15, 35, 8.05, 10, 20)
    assert tl[1] > tl[0]
    assert ph.transmission_loss_db(20, 1000, 15, 35, 8.05, 10, 20) > ph.transmission_loss_db(200, 1000, 15, 35, 8.05, 10, 20)


def test_shaped_noise_matches_requested_level():
    rng = np.random.default_rng(0)
    x = ph.shape_noise(lambda f: np.full_like(f, 60.0), None, 22050 * 20, 22050, rng, f_min=0.0)
    expected = 60.0 + 10 * np.log10(22050 / 2)
    assert abs(20 * np.log10(x.std()) - expected) < 0.5


def test_environment_coupling_rain_freshens_water():
    env = simulate_environment(2000, SITE_A, seed=1)
    rainy = env["runoff_index"] > 0.05
    if rainy.any():
        assert env["salinity_psu"][rainy].mean() < env["salinity_psu"][~rainy].mean()
