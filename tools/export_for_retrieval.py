"""
Bulk retrieval export -- packages Tier 1 raw audio (edge/output/audio/) and
a Tier 2 database (edge/output/db.sqlite) into one archive for a
maintenance-visit or opportunistic high-bandwidth sync, per
docs/data-pipeline.md's "Bulk retrieval / sync process" section.

Safe to run while the capture loop is still writing to the same SQLite
file: the source database is opened read-only (`file:...?mode=ro`), and
WAL mode (already enabled by simulation/pipeline/storage.py's init_db())
lets that read proceed concurrently with the writer without blocking
either side.

Usage:
    python -m tools.export_for_retrieval --output export.zip
    python -m tools.export_for_retrieval --last-n-windows 500 --output export.tar.gz
    python -m tools.export_for_retrieval --start 2026-01-01T00:00:00+00:00 \\
        --end 2026-02-01T00:00:00+00:00 --output january.zip
    python -m tools.export_for_retrieval --audio-dir /path/audio --db /path/db.sqlite --output export.zip
"""

import argparse
import io
import json
import logging
import os
import sqlite3
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from edge.config import load_config
from simulation.pipeline.storage import init_db

logger = logging.getLogger(__name__)

_CAPTURE_COLUMNS = ("capture_id", "timestamp_utc", "audio_filename", "duration_sec", "sample_rate_hz")


class _ArchiveWriter:
    """Thin common interface over zipfile/tarfile so export logic doesn't branch on format."""

    def __init__(self, path: str, fmt: str):
        self._fmt = fmt
        if fmt == "zip":
            self._zip = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED)
            self._tar = None
        elif fmt == "tar.gz":
            self._tar = tarfile.open(path, "w:gz")
            self._zip = None
        else:
            raise ValueError(f"unknown archive format {fmt!r} -- expected 'zip' or 'tar.gz'")

    def add_file(self, source_path: str, arcname: str) -> None:
        if self._zip is not None:
            self._zip.write(source_path, arcname)
        else:
            self._tar.add(source_path, arcname)

    def add_bytes(self, data: bytes, arcname: str) -> None:
        if self._zip is not None:
            self._zip.writestr(arcname, data)
        else:
            info = tarfile.TarInfo(name=arcname)
            info.size = len(data)
            self._tar.addfile(info, io.BytesIO(data))

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
        else:
            self._tar.close()

    def __enter__(self) -> "_ArchiveWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def infer_archive_format(output_path: str, explicit_format: Optional[str] = None) -> str:
    """Return "zip" or "tar.gz", from --format if given, else output_path's extension (default zip)."""
    if explicit_format:
        return explicit_format
    lower = str(output_path).lower()
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return "tar.gz"
    return "zip"


def _select_captures(
    conn: sqlite3.Connection,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    last_n_windows: Optional[int] = None,
) -> List[tuple]:
    """
    Fetch captures rows in chronological order, per the same start/end/
    last_n_windows selection convention as edge/recalibrate.py's
    _fetch_rows() (last_n_windows takes only the most recent N and ignores
    start/end; otherwise start/end bound timestamp_utc, either or both
    optional).
    """
    query = f"SELECT {', '.join(_CAPTURE_COLUMNS)} FROM captures"
    params: List = []

    if last_n_windows is not None:
        query += " ORDER BY timestamp_utc DESC, capture_id DESC LIMIT ?"
        params.append(last_n_windows)
        rows = conn.execute(query, params).fetchall()
        return list(reversed(rows))

    clauses = []
    if start is not None:
        clauses.append("timestamp_utc >= ?")
        params.append(start)
    if end is not None:
        clauses.append("timestamp_utc <= ?")
        params.append(end)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY timestamp_utc ASC, capture_id ASC"

    return conn.execute(query, params).fetchall()


def _build_filtered_db(
    source_conn: sqlite3.Connection,
    dest_path: str,
    capture_rows: List[tuple],
    *,
    health_start: Optional[str],
    health_end: Optional[str],
) -> None:
    """
    Build a standalone copy of the Tier 2 database at dest_path, containing
    only the selected captures and their one-to-one feature_vectors/
    environmental_readings/anomaly_flags rows, plus system_health_log rows
    within [health_start, health_end] -- system_health_log has no
    capture_id to join on (intentionally decoupled cadence, per
    docs/data-pipeline.md), so it's bounded by timestamp instead.
    """
    dest_conn = init_db(dest_path)
    capture_ids = [row[0] for row in capture_rows]

    dest_conn.executemany(
        """
        INSERT INTO captures (capture_id, timestamp_utc, audio_filename, duration_sec, sample_rate_hz)
        VALUES (?, ?, ?, ?, ?)
        """,
        capture_rows,
    )

    if capture_ids:
        placeholders = ",".join("?" * len(capture_ids))

        feature_rows = source_conn.execute(
            f"""
            SELECT capture_id, mfcc, spectral_centroid, zero_crossing_rate, rms_energy,
                   spectral_flatness, feature_vector_version
            FROM feature_vectors WHERE capture_id IN ({placeholders})
            """,
            capture_ids,
        ).fetchall()
        dest_conn.executemany(
            """
            INSERT INTO feature_vectors
                (capture_id, mfcc, spectral_centroid, zero_crossing_rate, rms_energy,
                 spectral_flatness, feature_vector_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            feature_rows,
        )

        env_rows = source_conn.execute(
            f"""
            SELECT capture_id, temperature_c, ph, turbidity_ntu, salinity_psu,
                   temp_roc, ph_roc, turbidity_roc, salinity_roc
            FROM environmental_readings WHERE capture_id IN ({placeholders})
            """,
            capture_ids,
        ).fetchall()
        dest_conn.executemany(
            """
            INSERT INTO environmental_readings
                (capture_id, temperature_c, ph, turbidity_ntu, salinity_psu,
                 temp_roc, ph_roc, turbidity_roc, salinity_roc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            env_rows,
        )

        anomaly_rows = source_conn.execute(
            f"""
            SELECT capture_id, anomaly_score, is_anomaly, baseline_version
            FROM anomaly_flags WHERE capture_id IN ({placeholders})
            """,
            capture_ids,
        ).fetchall()
        dest_conn.executemany(
            """
            INSERT INTO anomaly_flags (capture_id, anomaly_score, is_anomaly, baseline_version)
            VALUES (?, ?, ?, ?)
            """,
            anomaly_rows,
        )

    health_query = (
        "SELECT timestamp_utc, battery_voltage, solar_charge_w, enclosure_temp_c, imu_orientation, uptime_sec "
        "FROM system_health_log"
    )
    health_params: List = []
    health_clauses = []
    if health_start is not None:
        health_clauses.append("timestamp_utc >= ?")
        health_params.append(health_start)
    if health_end is not None:
        health_clauses.append("timestamp_utc <= ?")
        health_params.append(health_end)
    if health_clauses:
        health_query += " WHERE " + " AND ".join(health_clauses)
    health_rows = source_conn.execute(health_query, health_params).fetchall()
    dest_conn.executemany(
        """
        INSERT INTO system_health_log
            (timestamp_utc, battery_voltage, solar_charge_w, enclosure_temp_c, imu_orientation, uptime_sec)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        health_rows,
    )

    dest_conn.commit()
    dest_conn.close()


def export_for_retrieval(
    audio_dir: str,
    db_path: str,
    output_path: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    last_n_windows: Optional[int] = None,
    archive_format: Optional[str] = None,
) -> dict:
    """
    Build a retrieval archive at output_path: the selected windows' audio
    files under "audio/", a filtered "db.sqlite" containing only those
    windows' rows, and a "manifest.json" summary.

    Args:
        audio_dir: Tier 1 flat-file directory (config.storage.audio_dir).
        db_path: Tier 2 SQLite file (config.storage.db_path) -- opened
            read-only, safe to call while the capture loop is still writing
            to it (see module docstring).
        output_path: archive file to create.
        start, end: ISO 8601 timestamp bounds (inclusive), mutually
            exclusive with last_n_windows.
        last_n_windows: include only the most recent N stored windows.
        archive_format: "zip" or "tar.gz"; defaults to output_path's
            extension (zip if unrecognized).

    Returns:
        The manifest dict (also written into the archive as manifest.json):
        generated_at_utc, source_db_path, source_audio_dir, window_count,
        date_range ({start, end} of included windows, or {None, None} if
        window_count is 0), anomaly_count, total_audio_bytes, filter (the
        start/end/last_n_windows actually requested).
    """
    fmt = infer_archive_format(output_path, archive_format)

    source_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        capture_rows = _select_captures(source_conn, start=start, end=end, last_n_windows=last_n_windows)

        if capture_rows:
            health_start = min(row[1] for row in capture_rows)
            health_end = max(row[1] for row in capture_rows)
        else:
            health_start = health_end = None

        anomaly_count = 0
        if capture_rows:
            capture_ids = [row[0] for row in capture_rows]
            placeholders = ",".join("?" * len(capture_ids))
            anomaly_count = source_conn.execute(
                f"SELECT COUNT(*) FROM anomaly_flags WHERE capture_id IN ({placeholders}) AND is_anomaly = 1",
                capture_ids,
            ).fetchone()[0]

        # Keyed by filename, not appended per-row: two captures can
        # reference the same audio_filename (named by capture timestamp at
        # second resolution -- collides if two windows complete within the
        # same wall-clock second, e.g. a very short window_duration_s in
        # tests/demos). Without dedup that file would be added to the
        # archive once per referencing row (duplicate zip/tar entries) and
        # its size would be double-counted in total_audio_bytes.
        audio_files = {}
        for row in capture_rows:
            audio_filename = row[2]
            if audio_filename in audio_files:
                continue
            source_audio_path = os.path.join(audio_dir, audio_filename)
            if os.path.exists(source_audio_path):
                audio_files[audio_filename] = source_audio_path
            else:
                logger.warning("audio file referenced by capture but missing on disk, skipping: %s", source_audio_path)

        audio_entries: List[Tuple[str, str]] = [
            (source_path, filename) for filename, source_path in audio_files.items()
        ]
        total_audio_bytes = sum(os.path.getsize(path) for path, _ in audio_entries)

        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_db_path": str(db_path),
            "source_audio_dir": str(audio_dir),
            "window_count": len(capture_rows),
            "date_range": {"start": health_start, "end": health_end},
            "anomaly_count": anomaly_count,
            "total_audio_bytes": total_audio_bytes,
            "filter": {"start": start, "end": end, "last_n_windows": last_n_windows},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            filtered_db_path = os.path.join(tmp_dir, "db.sqlite")
            _build_filtered_db(
                source_conn, filtered_db_path, capture_rows, health_start=health_start, health_end=health_end
            )

            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            with _ArchiveWriter(output_path, fmt) as archive:
                archive.add_file(filtered_db_path, "db.sqlite")
                for source_audio_path, audio_filename in audio_entries:
                    archive.add_file(source_audio_path, f"audio/{audio_filename}")
                archive.add_bytes(json.dumps(manifest, indent=2).encode("utf-8"), "manifest.json")

        return manifest
    finally:
        source_conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Tier 1 audio + Tier 2 DB for bulk retrieval.")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to edge config YAML (default: edge/config.yaml)."
    )
    parser.add_argument(
        "--audio-dir", type=str, default=None, help="Tier 1 audio directory (default: config.storage.audio_dir)."
    )
    parser.add_argument("--db", type=str, default=None, help="Path to db.sqlite (default: config.storage.db_path).")
    parser.add_argument("--output", type=str, required=True, help="Output archive path, e.g. export.zip.")
    parser.add_argument(
        "--format",
        choices=["zip", "tar.gz"],
        default=None,
        help="Archive format (default: inferred from --output's extension, else zip).",
    )
    parser.add_argument("--last-n-windows", type=int, default=None, help="Include only the N most recent windows.")
    parser.add_argument("--start", type=str, default=None, help="ISO 8601 start timestamp, inclusive.")
    parser.add_argument("--end", type=str, default=None, help="ISO 8601 end timestamp, inclusive.")
    args = parser.parse_args()

    if args.last_n_windows is not None and (args.start is not None or args.end is not None):
        parser.error("--last-n-windows cannot be combined with --start/--end")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(Path(args.config)) if args.config else load_config()
    audio_dir = args.audio_dir or config.storage.audio_dir
    db_path = args.db or config.storage.db_path

    manifest = export_for_retrieval(
        audio_dir,
        db_path,
        args.output,
        start=args.start,
        end=args.end,
        last_n_windows=args.last_n_windows,
        archive_format=args.format,
    )

    print(f"Exported {manifest['window_count']} windows to {args.output}")
    print(f"Date range: {manifest['date_range']['start']} .. {manifest['date_range']['end']}")
    print(f"Anomaly flags: {manifest['anomaly_count']}")
    print(f"Total audio size: {manifest['total_audio_bytes']} bytes")


if __name__ == "__main__":
    main()
