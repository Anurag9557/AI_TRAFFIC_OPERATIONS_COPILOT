"""SQLite schema and atomic event replacement tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Settings
from src.data.preprocessing import preprocess_events
from src.services.database import (
    get_table_counts,
    initialize_database,
    replace_events,
    validate_event_store,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        raw_csv_path=tmp_path / "events.csv",
        database_path=tmp_path / "traffic_ops.db",
        faiss_index_path=tmp_path / "events.faiss",
        faiss_mapping_path=tmp_path / "mapping.json",
    )


def _source_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "EVENT-DB-1",
                "event_type": "unplanned",
                "latitude": 12.97,
                "longitude": 77.59,
                "event_cause": "vehicle_breakdown",
                "requires_road_closure": False,
                "start_datetime": "2024-03-01 10:00:00+00:00",
                "status": "closed",
                "corridor": "Mysore Road",
                "priority": "High",
                "police_station": "Test Station",
                "authenticated": "yes",
                "modified_datetime": "2024-03-01 11:00:00+00:00",
                "created_date": "2024-03-01 10:01:00+00:00",
                "closed_datetime": "2024-03-01 11:00:00+00:00",
            }
        ]
    )


def test_database_contains_only_approved_tables_and_event(tmp_path: Path) -> None:
    """The MVP database should contain one event and empty case/feedback tables."""

    settings = _settings(tmp_path)
    processed = preprocess_events(_source_row(), settings)

    initialize_database(settings.database_path)
    inserted = replace_events(settings.database_path, processed.events)
    validate_event_store(settings.database_path, expected_rows=1)

    assert inserted == 1
    assert get_table_counts(settings.database_path) == {
        "events": 1,
        "cases": 0,
        "feedback": 0,
    }

