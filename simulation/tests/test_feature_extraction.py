import numpy as np
import pandas as pd
import pytest

from simulation.data_generator.synthetic_audio import generate_ambient_background
from simulation.pipeline.feature_extraction import (
    build_joint_feature_vector,
    extract_acoustic_features,
    extract_environmental_features,
)

SAMPLE_RATE = 22050


def test_extract_acoustic_features_returns_expected_keys_and_types():
    audio = generate_ambient_background(duration_s=1.0, sample_rate=SAMPLE_RATE)
    features = extract_acoustic_features(audio, SAMPLE_RATE)

    # 13 MFCCs x (mean + std) = 26, plus 4 other features x (mean + std) = 8
    assert len(features) == 34
    assert "mfcc_1_mean" in features
    assert "mfcc_13_std" in features
    for name in (
        "spectral_centroid_mean",
        "zero_crossing_rate_mean",
        "rms_energy_mean",
        "spectral_flatness_mean",
    ):
        assert name in features
        assert isinstance(features[name], float)


def test_extract_acoustic_features_values_are_finite():
    audio = generate_ambient_background(duration_s=1.0, sample_rate=SAMPLE_RATE)
    features = extract_acoustic_features(audio, SAMPLE_RATE)
    assert all(np.isfinite(v) for v in features.values())


def test_extract_environmental_features_normalizes_against_baseline():
    # a row exactly at the reference baseline for every parameter, with no
    # rate-of-change, should normalize to ~0 for every feature
    row = pd.Series(
        {
            "temperature_c": 18.0,
            "ph": 8.05,
            "turbidity_ntu": 3.0,
            "salinity_psu": 35.0,
            "temp_roc": 0.0,
            "ph_roc": 0.0,
            "turbidity_roc": 0.0,
            "salinity_roc": 0.0,
        }
    )
    features = extract_environmental_features(row)

    assert len(features) == 8
    for value in features.values():
        assert value == pytest.approx(0.0)


def test_build_joint_feature_vector_concatenates_without_collision():
    acoustic = {"a": 1.0, "b": 2.0}
    environmental = {"c": 3.0, "d": 4.0}
    joint = build_joint_feature_vector(acoustic, environmental)

    assert isinstance(joint, pd.Series)
    assert len(joint) == 4
    assert joint["a"] == 1.0
    assert joint["d"] == 4.0
