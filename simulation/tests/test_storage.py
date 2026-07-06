import pandas as pd

from simulation.pipeline.storage import init_db, insert_window_record


def test_init_db_creates_expected_tables_and_wal_mode(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(str(db_path))

    tables = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {
        "captures",
        "feature_vectors",
        "environmental_readings",
        "anomaly_flags",
        "system_health_log",
    } <= tables

    journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal_mode == "wal"

    conn.close()


def test_insert_window_record_writes_across_all_tables(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(str(db_path))

    acoustic_features = {
        "spectral_centroid_mean": 1000.0,
        "zero_crossing_rate_mean": 0.1,
        "rms_energy_mean": 0.05,
        "spectral_flatness_mean": 0.2,
        "mfcc_1_mean": -100.0,
        "mfcc_1_std": 10.0,
    }
    environmental_row = pd.Series(
        {
            "temperature_c": 18.0,
            "ph": 8.05,
            "turbidity_ntu": 3.0,
            "salinity_psu": 35.0,
            "temp_roc": 0.1,
            "ph_roc": 0.0,
            "turbidity_roc": 0.0,
            "salinity_roc": -0.05,
        }
    )
    anomaly_result = {"anomaly_score": 0.42, "is_anomaly": True}

    capture_id = insert_window_record(
        conn,
        timestamp_utc="2026-01-01T00:00:00Z",
        audio_filename="20260101T000000Z.wav",
        duration_sec=5.0,
        sample_rate_hz=22050,
        acoustic_features=acoustic_features,
        environmental_row=environmental_row,
        anomaly_result=anomaly_result,
    )

    assert capture_id == 1

    capture_row = conn.execute(
        "SELECT audio_filename, duration_sec, sample_rate_hz FROM captures WHERE capture_id = ?",
        (capture_id,),
    ).fetchone()
    assert capture_row == ("20260101T000000Z.wav", 5.0, 22050)

    feature_row = conn.execute(
        "SELECT spectral_centroid, zero_crossing_rate FROM feature_vectors WHERE capture_id = ?",
        (capture_id,),
    ).fetchone()
    assert feature_row == (1000.0, 0.1)

    env_row = conn.execute(
        "SELECT temperature_c, salinity_psu FROM environmental_readings WHERE capture_id = ?",
        (capture_id,),
    ).fetchone()
    assert env_row == (18.0, 35.0)

    anomaly_row = conn.execute(
        "SELECT anomaly_score, is_anomaly FROM anomaly_flags WHERE capture_id = ?",
        (capture_id,),
    ).fetchone()
    assert anomaly_row == (0.42, 1)

    conn.close()
