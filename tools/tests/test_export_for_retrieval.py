"""
Tests for tools/export_for_retrieval.py -- bulk retrieval export archive,
built from a small mock-generated CaptureLoop dataset.
"""

import json
import os
import sqlite3
import tarfile
import zipfile
from datetime import datetime, timezone

from edge.capture_loop import CaptureLoop
from edge.config import EdgeConfig
from edge.hal.factory import build_hardware
from simulation.pipeline.storage import init_db
from tools.export_for_retrieval import export_for_retrieval, infer_archive_format


def _make_config(tmp_path) -> EdgeConfig:
    config = EdgeConfig()
    config.duty_cycle.window_duration_s = 1.0
    config.audio.sample_rate_hz = 8000
    config.storage.audio_dir = str(tmp_path / "audio")
    config.storage.db_path = str(tmp_path / "db.sqlite")
    config.storage.baseline_model_path = str(tmp_path / "baseline_model.joblib")
    config.hardware.mode = "mock"
    return config


def _run_windows(config, n: int, db_conn=None):
    close_after = db_conn is None
    if db_conn is None:
        db_conn = init_db(config.storage.db_path)
    hardware = build_hardware(config)
    loop = CaptureLoop(config, hardware, db_conn)
    for _ in range(n):
        loop.run_one_window()
    if close_after:
        db_conn.close()
    return db_conn


def _distinct_audio_filenames(db_path, last_n=None):
    """
    audio_filename is second-resolution (named by capture timestamp), so
    back-to-back windows in these fast tests can legitimately collide onto
    the same file -- tests must compare against *distinct* filenames
    actually referenced, not raw window count, to stay correct regardless
    of test timing.
    """
    conn = sqlite3.connect(db_path)
    try:
        query = "SELECT audio_filename FROM captures ORDER BY capture_id"
        if last_n is not None:
            query = "SELECT audio_filename FROM (SELECT audio_filename, capture_id FROM captures ORDER BY capture_id DESC LIMIT ?) ORDER BY capture_id"
            rows = conn.execute(query, (last_n,)).fetchall()
        else:
            rows = conn.execute(query).fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def test_infer_archive_format():
    assert infer_archive_format("export.zip") == "zip"
    assert infer_archive_format("export.tar.gz") == "tar.gz"
    assert infer_archive_format("export.tgz") == "tar.gz"
    assert infer_archive_format("export.bin") == "zip"
    assert infer_archive_format("export.tar.gz", explicit_format="zip") == "zip"


def test_export_creates_zip_with_audio_db_and_manifest(tmp_path):
    config = _make_config(tmp_path)
    _run_windows(config, n=5)
    output_path = str(tmp_path / "export.zip")

    manifest = export_for_retrieval(config.storage.audio_dir, config.storage.db_path, output_path)

    assert manifest["window_count"] == 5
    assert os.path.exists(output_path)
    expected_filenames = _distinct_audio_filenames(config.storage.db_path)

    with zipfile.ZipFile(output_path) as zf:
        names = zf.namelist()
        assert "db.sqlite" in names
        assert "manifest.json" in names
        audio_names = {n for n in names if n.startswith("audio/")}
        assert audio_names == {f"audio/{f}" for f in expected_filenames}

        manifest_from_archive = json.loads(zf.read("manifest.json"))
        assert manifest_from_archive == manifest


def test_export_creates_tar_gz_with_audio_db_and_manifest(tmp_path):
    config = _make_config(tmp_path)
    _run_windows(config, n=4)
    output_path = str(tmp_path / "export.tar.gz")

    manifest = export_for_retrieval(config.storage.audio_dir, config.storage.db_path, output_path)

    assert manifest["window_count"] == 4
    expected_filenames = _distinct_audio_filenames(config.storage.db_path)
    with tarfile.open(output_path, "r:gz") as tf:
        names = tf.getnames()
        assert "db.sqlite" in names
        assert "manifest.json" in names
        audio_names = {n for n in names if n.startswith("audio/")}
        assert audio_names == {f"audio/{f}" for f in expected_filenames}


def test_export_manifest_totals_match_actual_audio_files(tmp_path):
    config = _make_config(tmp_path)
    _run_windows(config, n=3)
    output_path = str(tmp_path / "export.zip")

    manifest = export_for_retrieval(config.storage.audio_dir, config.storage.db_path, output_path)

    actual_total = sum(
        os.path.getsize(os.path.join(config.storage.audio_dir, f))
        for f in os.listdir(config.storage.audio_dir)
    )
    assert manifest["total_audio_bytes"] == actual_total
    assert manifest["anomaly_count"] == 0  # uncalibrated windows always score as non-anomalous


def test_export_last_n_windows_filters_audio_and_db_rows(tmp_path):
    config = _make_config(tmp_path)
    _run_windows(config, n=8)
    output_path = str(tmp_path / "export.zip")

    manifest = export_for_retrieval(
        config.storage.audio_dir, config.storage.db_path, output_path, last_n_windows=3
    )

    assert manifest["window_count"] == 3
    expected_filenames = _distinct_audio_filenames(config.storage.db_path, last_n=3)
    with zipfile.ZipFile(output_path) as zf:
        audio_names = {n for n in zf.namelist() if n.startswith("audio/")}
        assert audio_names == {f"audio/{f}" for f in expected_filenames}

        zf.extract("db.sqlite", path=str(tmp_path / "extracted"))
    extracted_conn = sqlite3.connect(str(tmp_path / "extracted" / "db.sqlite"))
    for table in ("captures", "feature_vectors", "environmental_readings", "anomaly_flags"):
        count = extracted_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert count == 3, f"expected 3 rows in {table}, got {count}"
    extracted_conn.close()


def test_export_start_filter_excludes_windows_before_cutoff(tmp_path):
    config = _make_config(tmp_path)
    db_conn = init_db(config.storage.db_path)
    hardware = build_hardware(config)
    loop = CaptureLoop(config, hardware, db_conn)
    loop.run_one_window()

    # this window's timestamp was captured before "now", so a start bound
    # of "now" excludes it.
    future_start = datetime.now(timezone.utc).isoformat()
    db_conn.close()

    output_path = str(tmp_path / "export.zip")
    manifest = export_for_retrieval(
        config.storage.audio_dir, config.storage.db_path, output_path, start=future_start
    )

    assert manifest["window_count"] == 0
    assert manifest["date_range"] == {"start": None, "end": None}
    with zipfile.ZipFile(output_path) as zf:
        assert [n for n in zf.namelist() if n.startswith("audio/")] == []
        assert "db.sqlite" in zf.namelist()  # still-valid, just-empty DB is included


def test_export_is_safe_while_db_still_open_for_writing(tmp_path):
    config = _make_config(tmp_path)
    db_conn = init_db(config.storage.db_path)
    hardware = build_hardware(config)
    loop = CaptureLoop(config, hardware, db_conn)
    for _ in range(3):
        loop.run_one_window()

    # db_conn is deliberately left open (as the live capture loop's writer
    # connection would be) while exporting -- this must not raise or block.
    output_path = str(tmp_path / "export.zip")
    manifest = export_for_retrieval(config.storage.audio_dir, config.storage.db_path, output_path)
    assert manifest["window_count"] == 3

    # the writer can keep writing after the export completes
    loop.run_one_window()
    db_conn.close()
