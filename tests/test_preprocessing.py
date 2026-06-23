"""Focused tests for EDA-derived deterministic cleaning rules."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import Settings
from src.data.preprocessing import (
    normalize_event_cause,
    preprocess_events,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        raw_csv_path=tmp_path / "events.csv",
        database_path=tmp_path / "events.db",
        faiss_index_path=tmp_path / "events.faiss",
        faiss_mapping_path=tmp_path / "mapping.json",
    )


def test_event_cause_normalization_merges_debris_case() -> None:
    """`Debris` and `debris` must become one retrieval category."""

    assert normalize_event_cause("Debris") == "debris"
    assert normalize_event_cause("debris") == "debris"
    assert normalize_event_cause("Fog / Low Visibility") == "fog_low_visibility"


def test_preprocessing_derives_durations_and_flags(tmp_path: Path) -> None:
    """One representative row should produce deterministic derived fields."""

    raw = pd.DataFrame(
        [
            {
                "id": "EVENT-1",
                "event_type": "planned",
                "latitude": 12.97,
                "longitude": 77.59,
                "endlatitude": 0,
                "endlongitude": 0,
                "address": "Test Road, Bengaluru",
                "end_address": None,
                "event_cause": "Debris",
                "requires_road_closure": "TRUE",
                "start_datetime": "2024-03-01 10:00:00+00:00",
                "end_datetime": "2024-03-01 12:00:00+00:00",
                "status": "closed",
                "authenticated": "yes",
                "modified_datetime": "2024-03-01 13:00:00+00:00",
                "description": None,
                "veh_type": "heavy_vehicle",
                "corridor": "Non-corridor",
                "priority": "High",
                "cargo_material": None,
                "reason_breakdown": None,
                "age_of_truck": 2019,
                "created_date": "2024-03-01 10:05:00+00:00",
                "route_path": "[]",
                "police_station": "Test Station",
                "zone": "Central Zone 1",
                "junction": "Test Junction",
                "closed_datetime": "2024-03-01 13:00:00+00:00",
                "resolved_datetime": None,
            }
        ]
    )

    result = preprocess_events(raw, _settings(tmp_path))
    event = result.events.iloc[0]
    flags = json.loads(event["data_quality_flags"])

    assert event["event_cause"] == "debris"
    assert event["requires_road_closure"] == 1
    assert event["scheduled_duration_hours"] == 2.0
    assert event["handling_duration_hours"] == 3.0
    assert event["truck_age_years"] == 5.0
    assert "missing_description" in flags
    assert "empty_route_path" in flags
    assert "truck_age_derived_from_manufacturing_year" in flags
    assert "Event type: planned" in event["canonical_text"]
    assert "Cause: debris" in event["canonical_text"]

