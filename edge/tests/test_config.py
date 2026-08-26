import pytest

from edge.config import EdgeConfig, load_config


def test_default_config_is_valid_mock_mode():
    config = EdgeConfig()
    assert config.hardware.mode == "mock"
    assert config.duty_cycle.window_duration_s > 0
    assert config.duty_cycle.window_interval_minutes > 0
    assert config.storage.log_path == "edge/output/edge.log"
    assert config.logging.level == "INFO"


def test_load_config_missing_file_returns_defaults(tmp_path):
    config = load_config(tmp_path / "does_not_exist.yaml")
    assert config == EdgeConfig()


def test_load_config_applies_yaml_overrides(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
duty_cycle:
  window_duration_s: 3.0
  window_interval_minutes: 15.0
hardware:
  mode: real
  i2c:
    bus_number: 2
"""
    )
    config = load_config(yaml_path)

    assert config.duty_cycle.window_duration_s == 3.0
    assert config.duty_cycle.window_interval_minutes == 15.0
    assert config.hardware.mode == "real"
    assert config.hardware.i2c.bus_number == 2
    # untouched nested fields keep their defaults
    assert config.hardware.i2c.imu_address == "0x68"


def test_load_config_applies_logging_overrides(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
storage:
  log_path: /tmp/custom-edge.log
logging:
  level: DEBUG
"""
    )
    config = load_config(yaml_path)

    assert config.storage.log_path == "/tmp/custom-edge.log"
    assert config.logging.level == "DEBUG"


def test_load_config_rejects_unknown_key(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("not_a_real_field: 123\n")

    try:
        load_config(yaml_path)
        assert False, "expected ValueError for unknown config key"
    except ValueError:
        pass


def test_default_repo_config_yaml_loads_cleanly():
    # edge/config.yaml itself (the shipped default file) must parse and
    # merge without error -- this is what edge/main.py loads with no args.
    config = load_config()
    assert config.hardware.mode == "mock"


def test_validate_rejects_window_longer_than_interval():
    config = EdgeConfig()
    config.duty_cycle.window_duration_s = 700.0
    config.duty_cycle.window_interval_minutes = 10.0
    with pytest.raises(ValueError, match="window_duration_s"):
        config.validate()


def test_validate_rejects_too_few_calibration_windows():
    config = EdgeConfig()
    config.calibration.calibration_windows = 4
    with pytest.raises(ValueError, match="calibration_windows"):
        config.validate()


def test_validate_rejects_invalid_hardware_mode():
    config = EdgeConfig()
    config.hardware.mode = "simulated"
    with pytest.raises(ValueError, match="hardware.mode"):
        config.validate()


def test_validate_rejects_invalid_telemetry_type():
    config = EdgeConfig()
    config.hardware.telemetry.type = "satellite"
    with pytest.raises(ValueError, match="telemetry.type"):
        config.validate()


def test_validate_rejects_non_hex_i2c_address():
    config = EdgeConfig()
    config.hardware.i2c.env_sensor_addresses["ph"] = "68"
    with pytest.raises(ValueError, match="hex string"):
        config.validate()


def test_validate_rejects_i2c_address_collision():
    config = EdgeConfig()
    config.hardware.i2c.env_sensor_addresses["ph"] = "0x44"  # collides with temperature
    with pytest.raises(ValueError, match="collision"):
        config.validate()


def test_validate_detects_imu_collision_with_env_sensor():
    config = EdgeConfig()
    config.hardware.i2c.imu_address = "0x63"  # collides with ph
    with pytest.raises(ValueError, match="collision"):
        config.validate()


def test_validate_collects_all_violations_in_one_error():
    config = EdgeConfig()
    config.hardware.mode = "bogus"
    config.hardware.telemetry.type = "bogus"
    config.calibration.calibration_windows = 1
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    message = str(exc_info.value)
    assert "hardware.mode" in message
    assert "telemetry.type" in message
    assert "calibration_windows" in message


def test_load_config_validates(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
duty_cycle:
  window_duration_s: 601.0
  window_interval_minutes: 10.0
"""
    )
    with pytest.raises(ValueError, match="window_duration_s"):
        load_config(yaml_path)
