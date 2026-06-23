"""SQLite schema and persistence helpers for the hackathon MVP.

The module intentionally uses Python's standard-library `sqlite3` package.
This keeps the Phase 1-2 data layer small while providing transactions,
indexes, foreign keys, and a portable database file.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

EVENT_COLUMNS = (
    "event_id",
    "event_type",
    "event_cause",
    "latitude",
    "longitude",
    "end_latitude",
    "end_longitude",
    "address",
    "end_address",
    "requires_road_closure",
    "start_datetime",
    "end_datetime",
    "status",
    "authenticated",
    "modified_datetime",
    "description",
    "vehicle_type",
    "corridor",
    "priority",
    "cargo_material",
    "breakdown_reason",
    "truck_age_years",
    "created_datetime",
    "route_path",
    "police_station",
    "zone",
    "junction",
    "closed_datetime",
    "resolved_datetime",
    "scheduled_duration_hours",
    "handling_duration_hours",
    "event_hour_ist",
    "event_weekday_ist",
    "canonical_text",
    "data_quality_flags",
    "source_row_number",
    "source_hash",
    "ingested_at",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    event_cause TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    end_latitude REAL,
    end_longitude REAL,
    address TEXT,
    end_address TEXT,
    requires_road_closure INTEGER NOT NULL
        CHECK (requires_road_closure IN (0, 1)),
    start_datetime TEXT NOT NULL,
    end_datetime TEXT,
    status TEXT,
    authenticated INTEGER CHECK (authenticated IN (0, 1) OR authenticated IS NULL),
    modified_datetime TEXT,
    description TEXT,
    vehicle_type TEXT,
    corridor TEXT,
    priority TEXT,
    cargo_material TEXT,
    breakdown_reason TEXT,
    truck_age_years REAL,
    created_datetime TEXT,
    route_path TEXT,
    police_station TEXT,
    zone TEXT,
    junction TEXT,
    closed_datetime TEXT,
    resolved_datetime TEXT,
    scheduled_duration_hours REAL,
    handling_duration_hours REAL,
    event_hour_ist INTEGER CHECK (
        event_hour_ist BETWEEN 0 AND 23 OR event_hour_ist IS NULL
    ),
    event_weekday_ist TEXT,
    canonical_text TEXT NOT NULL,
    data_quality_flags TEXT NOT NULL DEFAULT '[]',
    source_row_number INTEGER NOT NULL,
    source_hash TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_cause ON events(event_cause);
CREATE INDEX IF NOT EXISTS idx_events_corridor ON events(corridor);
CREATE INDEX IF NOT EXISTS idx_events_priority ON events(priority);
CREATE INDEX IF NOT EXISTS idx_events_closure ON events(requires_road_closure);
CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_datetime);
CREATE INDEX IF NOT EXISTS idx_events_police_station ON events(police_station);

CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    user_input TEXT NOT NULL,
    structured_event_json TEXT NOT NULL,
    similar_event_ids_json TEXT NOT NULL DEFAULT '[]',
    historical_stats_json TEXT NOT NULL DEFAULT '{}',
    risk_assessment_json TEXT NOT NULL DEFAULT '{}',
    resource_plan_json TEXT NOT NULL DEFAULT '{}',
    report TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY,
    case_id TEXT NOT NULL,
    actual_road_closure INTEGER CHECK (
        actual_road_closure IN (0, 1) OR actual_road_closure IS NULL
    ),
    actual_handling_duration_hours REAL CHECK (
        actual_handling_duration_hours >= 0
        OR actual_handling_duration_hours IS NULL
    ),
    actual_manpower INTEGER CHECK (
        actual_manpower >= 0 OR actual_manpower IS NULL
    ),
    actual_barricades INTEGER CHECK (
        actual_barricades >= 0 OR actual_barricades IS NULL
    ),
    support_vehicles_used TEXT NOT NULL DEFAULT '[]',
    observed_operational_impact TEXT,
    notes TEXT,
    submitted_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_case_id ON feedback(case_id);
"""


class DatabaseError(RuntimeError):
    """Raised when an SQLite operation cannot be completed safely."""


@contextmanager
def connect_database(database_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a configured SQLite connection and close it reliably."""

    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    try:
        yield connection
    finally:
        connection.close()


def initialize_database(database_path: Path) -> None:
    """Create the approved `events`, `cases`, and `feedback` tables."""

    try:
        with connect_database(database_path) as connection:
            connection.executescript(SCHEMA_SQL)
            connection.commit()
    except sqlite3.Error as exc:
        raise DatabaseError(
            f"Failed to initialize SQLite database at {database_path}: {exc}"
        ) from exc
    LOGGER.info("Initialized SQLite schema at %s.", database_path)


def _to_sql_value(value: Any) -> Any:
    """Convert Pandas/NumPy scalars into values supported by sqlite3."""

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bool):
        return int(value)
    return value


def _iter_event_rows(events: pd.DataFrame) -> Iterator[tuple[Any, ...]]:
    """Yield database-compatible tuples in the fixed event-column order."""

    missing_columns = sorted(set(EVENT_COLUMNS) - set(events.columns))
    if missing_columns:
        raise DatabaseError(
            f"Processed events are missing database columns: {missing_columns}"
        )

    for row in events.loc[:, EVENT_COLUMNS].itertuples(index=False, name=None):
        yield tuple(_to_sql_value(value) for value in row)


def replace_events(
    database_path: Path,
    events: pd.DataFrame,
    batch_size: int = 1_000,
) -> int:
    """Atomically replace historical events while preserving cases/feedback."""

    if events.empty:
        raise DatabaseError("Refusing to replace the events table with no rows.")
    if batch_size <= 0:
        raise DatabaseError("batch_size must be a positive integer.")

    placeholders = ", ".join("?" for _ in EVENT_COLUMNS)
    columns = ", ".join(EVENT_COLUMNS)
    insert_sql = f"INSERT INTO events ({columns}) VALUES ({placeholders})"
    rows = iter(_iter_event_rows(events))
    inserted = 0

    try:
        with connect_database(database_path) as connection:
            with connection:
                connection.execute("DELETE FROM events")
                while True:
                    batch: list[tuple[Any, ...]] = []
                    for _ in range(batch_size):
                        try:
                            batch.append(next(rows))
                        except StopIteration:
                            break
                    if not batch:
                        break
                    connection.executemany(insert_sql, batch)
                    inserted += len(batch)
    except (sqlite3.Error, DatabaseError) as exc:
        raise DatabaseError(
            f"Failed to replace historical events in {database_path}: {exc}"
        ) from exc

    LOGGER.info("Inserted %s historical events into SQLite.", inserted)
    return inserted


def get_table_counts(database_path: Path) -> dict[str, int]:
    """Return row counts for the three approved MVP tables."""

    try:
        with connect_database(database_path) as connection:
            return {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in ("events", "cases", "feedback")
            }
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to count SQLite rows: {exc}") from exc


def validate_event_store(database_path: Path, expected_rows: int) -> None:
    """Run compact integrity checks after an ingestion transaction."""

    checks = {
        "event_count": ("SELECT COUNT(*) FROM events", expected_rows),
        "unique_event_ids": (
            "SELECT COUNT(DISTINCT event_id) FROM events",
            expected_rows,
        ),
        "canonical_text_present": (
            "SELECT COUNT(*) FROM events WHERE canonical_text <> ''",
            expected_rows,
        ),
    }
    try:
        with connect_database(database_path) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise DatabaseError(f"SQLite integrity check failed: {integrity}")
            for name, (query, expected) in checks.items():
                actual = int(connection.execute(query).fetchone()[0])
                if actual != expected:
                    raise DatabaseError(
                        f"Event-store check '{name}' expected {expected}, got {actual}."
                    )
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to validate SQLite event store: {exc}") from exc

    LOGGER.info("SQLite integrity and event-store checks passed.")


def save_case(
    database_path: Path,
    *,
    case_id: str,
    user_input: str,
    structured_event_json: str,
    similar_event_ids_json: str,
    historical_stats_json: str,
    risk_assessment_json: str,
    resource_plan_json: str,
    report: str,
) -> None:
    """Insert or replace one completed copilot case."""

    try:
        with connect_database(database_path) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO cases (
                        case_id,
                        user_input,
                        structured_event_json,
                        similar_event_ids_json,
                        historical_stats_json,
                        risk_assessment_json,
                        resource_plan_json,
                        report
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(case_id) DO UPDATE SET
                        user_input = excluded.user_input,
                        structured_event_json = excluded.structured_event_json,
                        similar_event_ids_json = excluded.similar_event_ids_json,
                        historical_stats_json = excluded.historical_stats_json,
                        risk_assessment_json = excluded.risk_assessment_json,
                        resource_plan_json = excluded.resource_plan_json,
                        report = excluded.report
                    """,
                    (
                        case_id,
                        user_input,
                        structured_event_json,
                        similar_event_ids_json,
                        historical_stats_json,
                        risk_assessment_json,
                        resource_plan_json,
                        report,
                    ),
                )
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to save case {case_id}: {exc}") from exc
