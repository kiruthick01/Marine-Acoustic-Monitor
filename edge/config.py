"""
Edge deployment configuration.

Single source of truth for every tunable the production capture loop needs:
duty-cycle timing (DECISIONS.md: "record N seconds every M minutes"), audio/
signal-conditioning parameters (matching docs/ml-pipeline.md Stage 0
defaults), storage paths (Tier 1/2 per docs/data-pipeline.md), calibration
parameters (Stage 2), and the hardware backend selection (mock vs. real,
plus I2C bus/address plan -- see docs/pi-implementation.md).

Loaded from edge/config.yaml by default; every field has a code-level
default matching the locked design docs, so `EdgeConfig()` with no file at
all is already a valid mock-mode configuration -- config.yaml exists for
overriding those defaults per deployment site, not because a file is
strictly required.
"""

import dataclasses
from pathlib import Path
from typing import Dict, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclasses.dataclass
class DutyCycleConfig:
    # N seconds every M minutes -- DECISIONS.md's locked sampling model.
    window_duration_s: float = 5.0
    window_interval_minutes: float = 10.0


@dataclasses.dataclass
class AudioConfig:
    sample_rate_hz: int = 22050


@dataclasses.dataclass
class SignalConditioningConfig:
    # Matches simulation/pipeline/signal_conditioning.py's condition_signal()
    # high_hz=12000 override (wide enough for the real bioacoustic band of
    # interest, not just the synthetic whistle it was originally tuned for).
    bandpass_low_hz: float = 20.0
    bandpass_high_hz: float = 12000.0


@dataclasses.dataclass
class StorageConfig:
    # Relative to the repo root; matches simulation/'s own output/ layout so
    # the same downstream tooling (evaluate.py-style analysis) works on
    # either tree.
    audio_dir: str = "edge/output/audio"
    db_path: str = "edge/output/db.sqlite"
    baseline_model_path: str = "edge/output/baseline_model.joblib"
    log_path: str = "edge/output/edge.log"


@dataclasses.dataclass
class LoggingConfig:
    # Standard library logging level name ("DEBUG"/"INFO"/"WARNING"/...),
    # applied to the "edge" logger tree (edge/logging_setup.py) -- controls
    # both the rotating file handler and the console handler.
    level: str = "INFO"


@dataclasses.dataclass
class CalibrationConfig:
    # Number of duty-cycle windows treated as the initial "assumed normal"
    # calibration period -- docs/ml-pipeline.md Stage 2. 30 windows at the
    # default 10-minute interval is 5 hours; a real deployment should widen
    # this considerably (docs/pi-implementation.md recommends >= 1 full diel
    # cycle, i.e. >= 144 windows at a 10-minute interval) once real duty-
    # cycle timing is finalized against the power budget.
    calibration_windows: int = 30
    threshold_sigma: float = 5.0
    baseline_version: str = "v1"
    feature_vector_version: str = "v1"


@dataclasses.dataclass
class I2CConfig:
    bus_number: int = 1
    # Address plan -- see docs/pi-implementation.md for the full rationale
    # and collision-avoidance notes. Values here are the *planned* defaults;
    # real sensor default addresses must be checked against each part's
    # datasheet once the BOM is priced and may need to be re-mapped via each
    # sensor's address-select pins/jumpers if two chosen parts collide.
    env_sensor_addresses: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "temperature": "0x44",  # e.g. SHT31-class temp/RH combo breakout
            "ph": "0x63",  # e.g. Atlas Scientific / DFRobot EZO-class pH breakout
            "turbidity": "0x48",  # analog turbidity via ADS1115 ADC channel
            "salinity": "0x64",  # e.g. EZO-class conductivity breakout
        }
    )
    imu_address: str = "0x68"  # e.g. MPU6050/MPU9250-class default address


@dataclasses.dataclass
class TelemetryConfig:
    # DECISIONS.md: LoRa-vs-cellular is a deployment-specific decision, not
    # fixed at the architecture level -- selects which edge/hal/real.py
    # driver RealTelemetryLink wraps.
    type: str = "lora"  # "lora" | "cellular"
    max_payload_bytes: int = 220


@dataclasses.dataclass
class HardwareConfig:
    # "mock" runs entirely on synthetic data (edge/hal/mock.py) -- no
    # hardware required, safe to run on any dev machine today. "real" wires
    # up edge/hal/real.py's stub drivers, which raise NotImplementedError
    # until hardware exists and drivers are written (see that module).
    mode: str = "mock"
    i2c: I2CConfig = dataclasses.field(default_factory=I2CConfig)
    telemetry: TelemetryConfig = dataclasses.field(default_factory=TelemetryConfig)


@dataclasses.dataclass
class EdgeConfig:
    duty_cycle: DutyCycleConfig = dataclasses.field(default_factory=DutyCycleConfig)
    audio: AudioConfig = dataclasses.field(default_factory=AudioConfig)
    signal_conditioning: SignalConditioningConfig = dataclasses.field(
        default_factory=SignalConditioningConfig
    )
    storage: StorageConfig = dataclasses.field(default_factory=StorageConfig)
    calibration: CalibrationConfig = dataclasses.field(default_factory=CalibrationConfig)
    hardware: HardwareConfig = dataclasses.field(default_factory=HardwareConfig)
    logging: LoggingConfig = dataclasses.field(default_factory=LoggingConfig)

    def validate(self) -> None:
        """
        Fail fast on a nonsensical configuration.

        Collects every violation found and raises a single ValueError
        listing all of them, so an operator fixing a bad config.yaml sees
        the whole picture at once instead of one crash per edit.
        """
        problems = []

        window_s = self.duty_cycle.window_duration_s
        interval_s = self.duty_cycle.window_interval_minutes * 60
        if not window_s < interval_s:
            problems.append(
                f"duty_cycle.window_duration_s ({window_s}) must be shorter than "
                f"window_interval_minutes ({self.duty_cycle.window_interval_minutes} "
                f"min = {interval_s} s) -- the device can't record longer than it sleeps"
            )

        if self.calibration.calibration_windows < 5:
            problems.append(
                f"calibration.calibration_windows ({self.calibration.calibration_windows}) "
                f"must be >= 5 -- too few windows to fit a meaningful baseline"
            )

        if self.hardware.mode not in ("mock", "real"):
            problems.append(
                f"hardware.mode ({self.hardware.mode!r}) must be 'mock' or 'real'"
            )

        if self.hardware.telemetry.type not in ("lora", "cellular"):
            problems.append(
                f"hardware.telemetry.type ({self.hardware.telemetry.type!r}) "
                f"must be 'lora' or 'cellular'"
            )

        addresses = dict(self.hardware.i2c.env_sensor_addresses)
        addresses["imu"] = self.hardware.i2c.imu_address
        parsed: Dict[str, int] = {}
        for sensor, addr in addresses.items():
            key = (
                f"hardware.i2c.imu_address"
                if sensor == "imu"
                else f"hardware.i2c.env_sensor_addresses.{sensor}"
            )
            if not (
                isinstance(addr, str) and addr.lower().startswith("0x") and len(addr) > 2
            ):
                problems.append(f"{key} ({addr!r}) must be a hex string like '0x44'")
                continue
            try:
                parsed[sensor] = int(addr, 16)
            except ValueError:
                problems.append(f"{key} ({addr!r}) must be a hex string like '0x44'")

        seen: Dict[int, str] = {}
        for sensor, value in parsed.items():
            if value in seen:
                problems.append(
                    f"I2C address collision: {seen[value]!r} and {sensor!r} both use "
                    f"{hex(value)} -- re-map one via its address-select pins"
                )
            else:
                seen[value] = sensor

        if problems:
            raise ValueError(
                "Invalid edge configuration:\n"
                + "\n".join(f"  - {p}" for p in problems)
            )


def _merge_dataclass(instance, overrides: dict):
    """Apply a dict of overrides onto a dataclass instance, recursing into nested dataclasses."""
    for key, value in overrides.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown config key {key!r} for {type(instance).__name__}")
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: Optional[Path] = None) -> EdgeConfig:
    """
    Load configuration, layering YAML overrides on top of code-level
    defaults.

    Args:
        path: path to a YAML config file. Defaults to edge/config.yaml.
            If the file doesn't exist, returns pure code-level defaults
            (a valid mock-mode config) rather than raising.

    Returns:
        Fully-populated EdgeConfig.
    """
    config = EdgeConfig()
    path = path or DEFAULT_CONFIG_PATH
    path = Path(path)

    if not path.exists():
        config.validate()
        return config

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    config = _merge_dataclass(config, raw)
    config.validate()
    return config
