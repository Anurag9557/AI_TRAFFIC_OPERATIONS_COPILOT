"""CSV-to-SQLite ingestion orchestration for historical traffic events."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import Settings
from src.data.preprocessing import (
    PreprocessingError,
    preprocess_events,
)
from src.schemas import IngestionSummary
from src.services.database import (
    DatabaseError,
    get_table_counts,
    initialize_database,
    replace_events,
    validate_event_store,
)

LOGGER = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """Raised when a complete CSV ingestion run fails."""


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 fingerprint of a source file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_source_csv(csv_path: Path) -> pd.DataFrame:
    """Read the provided CSV with explicit UTF-8 and null handling."""

    if not csv_path.exists():
        raise IngestionError(f"Source CSV does not exist: {csv_path}")
    if not csv_path.is_file():
        raise IngestionError(f"Source CSV path is not a file: {csv_path}")

    try:
        events = pd.read_csv(
            csv_path,
            encoding="utf-8",
            low_memory=False,
            keep_default_na=True,
            na_values=["NULL", "null"],
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise IngestionError(f"Unable to read source CSV {csv_path}: {exc}") from exc

    LOGGER.info("Read %s rows and %s columns from %s.", *events.shape, csv_path)
    return events


def ingest_csv(
    csv_path: Path,
    database_path: Path,
    settings: Settings,
    batch_size: int = 1_000,
) -> IngestionSummary:
    """Clean the source CSV and atomically rebuild the SQLite events table."""

    source_hash = _sha256_file(csv_path)
    raw_events = read_source_csv(csv_path)

    try:
        processed = preprocess_events(raw_events, settings)
        initialize_database(database_path)
        inserted_rows = replace_events(
            database_path,
            processed.events,
            batch_size=batch_size,
        )
        validate_event_store(database_path, expected_rows=inserted_rows)
        table_counts = get_table_counts(database_path)
    except (PreprocessingError, DatabaseError, ValueError, TypeError) as exc:
        raise IngestionError(f"Historical event ingestion failed: {exc}") from exc

    start_values = pd.to_datetime(
        processed.events["start_datetime"],
        errors="coerce",
        utc=True,
    )
    date_range_start = (
        start_values.min().isoformat() if start_values.notna().any() else None
    )
    date_range_end = (
        start_values.max().isoformat() if start_values.notna().any() else None
    )

    summary = IngestionSummary(
        source_path=str(csv_path.resolve()),
        source_sha256=source_hash,
        database_path=str(database_path.resolve()),
        source_rows=len(raw_events),
        exact_duplicate_rows_removed=processed.exact_duplicate_rows_removed,
        inserted_rows=inserted_rows,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        table_counts=table_counts,
        quality_flag_counts=processed.quality_flag_counts,
        completed_at=datetime.now(timezone.utc).isoformat(),
        notes=[
            "Handling duration uses valid resolved/closed timestamps and is not "
            "a congestion-duration measurement.",
            "Resource quantities are not present in the historical CSV.",
            "Canonical text excludes vehicle numbers and administrative user IDs.",
        ],
    )
    LOGGER.info(
        "Ingestion completed: %s rows inserted into %s.",
        inserted_rows,
        database_path,
    )
    return summary

