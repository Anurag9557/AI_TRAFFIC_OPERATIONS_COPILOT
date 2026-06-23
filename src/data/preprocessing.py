"""Deterministic cleaning and feature preparation for historical events.

The rules in this module come directly from the source CSV's observed shape:
mixed timestamp formats, sparse end coordinates, inconsistent cause casing,
mixed truck age/manufacturing-year values, and partially populated outcome
timestamps. No statistical model or LLM is used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.config import Settings
from src.schemas import HistoricalEventRecord

LOGGER = logging.getLogger(__name__)

REQUIRED_SOURCE_COLUMNS = {
    "id",
    "event_type",
    "latitude",
    "longitude",
    "event_cause",
    "requires_road_closure",
    "start_datetime",
    "status",
    "corridor",
    "priority",
    "police_station",
}

TIMESTAMP_COLUMNS = (
    "start_datetime",
    "end_datetime",
    "created_date",
    "modified_datetime",
    "closed_datetime",
    "resolved_datetime",
)

NULL_TEXT_VALUES = {"", "null", "none", "n/a", "na", "nan"}

CAUSE_ALIASES = {
    "debris": "debris",
    "fog_low_visibility": "fog_low_visibility",
}


class PreprocessingError(RuntimeError):
    """Raised when source data cannot be safely transformed."""


@dataclass(frozen=True)
class PreprocessingResult:
    """Output of a successful source-data preprocessing run."""

    events: pd.DataFrame
    exact_duplicate_rows_removed: int
    quality_flag_counts: dict[str, int]


def _is_missing(value: Any) -> bool:
    """Return whether a scalar value should be treated as missing."""

    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_text(value: Any) -> str | None:
    """Trim a text value and convert common literal nulls to `None`."""

    if _is_missing(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return None if text.casefold() in NULL_TEXT_VALUES else text


def normalize_slug(value: Any) -> str | None:
    """Convert a categorical label to a stable lowercase snake-case value."""

    text = normalize_text(value)
    if text is None:
        return None
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or None


def normalize_event_cause(value: Any) -> str:
    """Normalize event-cause spelling and casing."""

    slug = normalize_slug(value)
    if slug is None:
        return "unknown"
    return CAUSE_ALIASES.get(slug, slug)


def normalize_corridor(value: Any) -> str | None:
    """Normalize corridor whitespace while retaining readable source labels."""

    text = normalize_text(value)
    if text is None:
        return None
    if normalize_slug(text) in {"non_corridor", "noncorridor"}:
        return "Non-corridor"
    return text


def normalize_priority(value: Any) -> str | None:
    """Normalize the source's High/Low priority values."""

    slug = normalize_slug(value)
    if slug == "high":
        return "High"
    if slug == "low":
        return "Low"
    return normalize_text(value)


def parse_boolean(value: Any) -> int | None:
    """Convert common boolean representations to SQLite-compatible integers."""

    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return int(value)
    text = str(value).strip().casefold()
    if text in {"true", "yes", "y", "1"}:
        return 1
    if text in {"false", "no", "n", "0"}:
        return 0
    return None


def _to_iso_utc(value: pd.Timestamp | None) -> str | None:
    """Serialize a parsed UTC timestamp to ISO 8601."""

    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _valid_global_coordinate(latitude: float, longitude: float) -> bool:
    """Check whether a coordinate pair falls within global bounds."""

    return -90 <= latitude <= 90 and -180 <= longitude <= 180


def _valid_bengaluru_coordinate(
    latitude: float,
    longitude: float,
    settings: Settings,
) -> bool:
    """Check whether a coordinate pair falls in the MVP operating area."""

    return (
        settings.bengaluru_min_latitude
        <= latitude
        <= settings.bengaluru_max_latitude
        and settings.bengaluru_min_longitude
        <= longitude
        <= settings.bengaluru_max_longitude
    )


def _clean_end_coordinates(
    latitude_value: Any,
    longitude_value: Any,
    settings: Settings,
    flags: list[str],
) -> tuple[float | None, float | None]:
    """Validate sparse end coordinates and replace sentinels with nulls."""

    if _is_missing(latitude_value) or _is_missing(longitude_value):
        return None, None

    latitude = float(latitude_value)
    longitude = float(longitude_value)
    if latitude == 0 and longitude == 0:
        return None, None
    if not _valid_global_coordinate(latitude, longitude):
        flags.append("invalid_end_coordinates")
        return None, None
    if not _valid_bengaluru_coordinate(latitude, longitude, settings):
        flags.append("end_coordinates_outside_bengaluru")
        return None, None
    return latitude, longitude


def _clean_route_path(value: Any, flags: list[str]) -> str | None:
    """Validate and compact the sparse JSON route-path field."""

    text = normalize_text(value)
    if text is None:
        return None
    try:
        route = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        flags.append("invalid_route_path_json")
        return None

    if not isinstance(route, list):
        flags.append("invalid_route_path_shape")
        return None
    if not route:
        flags.append("empty_route_path")
        return None

    cleaned_points: list[list[float]] = []
    for point in route:
        if not isinstance(point, list) or len(point) < 2:
            flags.append("invalid_route_path_point")
            return None
        try:
            latitude = float(point[0])
            longitude = float(point[1])
        except (TypeError, ValueError):
            flags.append("invalid_route_path_point")
            return None
        if not _valid_global_coordinate(latitude, longitude):
            flags.append("invalid_route_path_point")
            return None
        cleaned_points.append([latitude, longitude])

    return json.dumps(cleaned_points, separators=(",", ":"))


def _normalize_truck_age(
    value: Any,
    event_year: int,
    flags: list[str],
) -> float | None:
    """Normalize a truck age field that mixes ages and manufacturing years."""

    if _is_missing(value):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        flags.append("invalid_truck_age")
        return None

    if not math.isfinite(numeric_value):
        flags.append("invalid_truck_age")
        return None
    if 0 <= numeric_value <= 80:
        return numeric_value
    if 1900 <= numeric_value <= event_year:
        flags.append("truck_age_derived_from_manufacturing_year")
        return float(event_year - int(numeric_value))

    flags.append("invalid_truck_age")
    return None


def _time_bucket(hour: int) -> str:
    """Map local event hour to a compact human-readable period."""

    if 0 <= hour < 6:
        return "overnight"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 22:
        return "evening"
    return "night"


def _display_label(slug: str | None) -> str | None:
    """Convert a normalized slug into readable canonical text."""

    return slug.replace("_", " ") if slug else None


def _build_canonical_text(event: dict[str, Any]) -> str:
    """Build privacy-conscious retrieval text from operational fields."""

    fields: list[tuple[str, Any]] = [
        ("Event type", _display_label(event["event_type"])),
        ("Cause", _display_label(event["event_cause"])),
        ("Corridor", event["corridor"]),
        ("Priority", event["priority"]),
        (
            "Road closure required",
            "yes" if event["requires_road_closure"] else "no",
        ),
        ("Police station", event["police_station"]),
        ("Zone", event["zone"]),
        ("Junction", event["junction"]),
        ("Address", event["address"]),
        ("Vehicle type", _display_label(event["vehicle_type"])),
        ("Cargo", event["cargo_material"]),
        ("Breakdown reason", event["breakdown_reason"]),
        ("Weekday", event["event_weekday_ist"]),
        (
            "Time period",
            _time_bucket(event["event_hour_ist"])
            if event["event_hour_ist"] is not None
            else None,
        ),
        (
            "Scheduled duration hours",
            round(event["scheduled_duration_hours"], 2)
            if event["scheduled_duration_hours"] is not None
            else None,
        ),
        ("Description", event["description"]),
    ]
    return ". ".join(f"{label}: {value}" for label, value in fields if value is not None)


def _source_row_hash(row: pd.Series) -> str:
    """Create a stable fingerprint of one original CSV row."""

    serializable = {
        str(key): None if _is_missing(value) else str(value)
        for key, value in row.items()
        if key != "_source_row_number"
    }
    payload = json.dumps(
        serializable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _duration_hours(
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> float | None:
    """Calculate elapsed hours between two parsed timestamps."""

    if end is None or pd.isna(end):
        return None
    return float((end - start).total_seconds() / 3600)


def _parse_timestamps(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Parse all source timestamps using Pandas mixed-format UTC handling."""

    parsed: dict[str, pd.Series] = {}
    for column in TIMESTAMP_COLUMNS:
        if column not in frame.columns:
            parsed[column] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
            continue
        parsed[column] = pd.to_datetime(
            frame[column],
            errors="coerce",
            format="mixed",
            utc=True,
        )
    return parsed


def preprocess_events(raw_events: pd.DataFrame, settings: Settings) -> PreprocessingResult:
    """Clean source events and return database-ready validated records.

    Args:
        raw_events: DataFrame read directly from the source CSV.
        settings: Validated project settings and operating-area bounds.

    Raises:
        PreprocessingError: If required columns, identifiers, coordinates, or
            start timestamps cannot be safely processed.
    """

    missing_columns = sorted(REQUIRED_SOURCE_COLUMNS - set(raw_events.columns))
    if missing_columns:
        raise PreprocessingError(
            f"Source CSV is missing required columns: {', '.join(missing_columns)}"
        )
    if raw_events.empty:
        raise PreprocessingError("Source CSV contains no event rows.")

    frame = raw_events.copy()
    frame["_source_row_number"] = frame.index + 2
    duplicate_mask = frame.drop(columns=["_source_row_number"]).duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        LOGGER.warning("Removing %s exact duplicate source rows.", duplicate_count)
        frame = frame.loc[~duplicate_mask].copy()

    normalized_ids = frame["id"].map(normalize_text)
    if normalized_ids.isna().any():
        rows = frame.loc[normalized_ids.isna(), "_source_row_number"].tolist()
        raise PreprocessingError(f"Missing event IDs at CSV rows: {rows[:10]}")
    if normalized_ids.duplicated().any():
        duplicate_ids = normalized_ids[normalized_ids.duplicated(keep=False)].unique()
        raise PreprocessingError(
            "Duplicate event IDs are not safe to ingest: "
            + ", ".join(map(str, duplicate_ids[:10]))
        )

    parsed_timestamps = _parse_timestamps(frame)
    ingested_at = datetime.now(timezone.utc).isoformat()
    quality_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []

    for index, row in frame.iterrows():
        flags: list[str] = []
        source_row_number = int(row["_source_row_number"])
        event_id = normalize_text(row["id"])
        start_timestamp = parsed_timestamps["start_datetime"].loc[index]
        if pd.isna(start_timestamp):
            raise PreprocessingError(
                f"Event {event_id} at CSV row {source_row_number} has an invalid "
                "start_datetime."
            )

        latitude_value = row.get("latitude")
        longitude_value = row.get("longitude")
        if _is_missing(latitude_value) or _is_missing(longitude_value):
            raise PreprocessingError(
                f"Event {event_id} at CSV row {source_row_number} is missing its "
                "start coordinate."
            )
        latitude = float(latitude_value)
        longitude = float(longitude_value)
        if not _valid_global_coordinate(latitude, longitude):
            raise PreprocessingError(
                f"Event {event_id} has an invalid global start coordinate."
            )
        if not _valid_bengaluru_coordinate(latitude, longitude, settings):
            flags.append("start_coordinates_outside_bengaluru")

        event_type = normalize_slug(row.get("event_type")) or "unknown"
        if event_type not in {"planned", "unplanned"}:
            flags.append("unknown_event_type")

        event_cause = normalize_event_cause(row.get("event_cause"))
        if event_cause == "unknown":
            flags.append("missing_event_cause")

        requires_road_closure = parse_boolean(row.get("requires_road_closure"))
        if requires_road_closure is None:
            flags.append("invalid_road_closure_value")
            requires_road_closure = 0

        priority = normalize_priority(row.get("priority"))
        if priority is None:
            flags.append("missing_priority")

        corridor = normalize_corridor(row.get("corridor"))
        if corridor is None:
            flags.append("missing_corridor")

        description = normalize_text(row.get("description"))
        if description is None:
            flags.append("missing_description")
        elif any("\u0c80" <= character <= "\u0cff" for character in description):
            flags.append("kannada_description")

        end_latitude, end_longitude = _clean_end_coordinates(
            row.get("endlatitude"),
            row.get("endlongitude"),
            settings,
            flags,
        )

        end_timestamp = parsed_timestamps["end_datetime"].loc[index]
        if event_type == "planned" and pd.isna(end_timestamp):
            flags.append("planned_event_missing_end_datetime")

        scheduled_duration = _duration_hours(start_timestamp, end_timestamp)
        if scheduled_duration is not None and scheduled_duration < 0:
            flags.append("negative_scheduled_duration")
            scheduled_duration = None
        elif scheduled_duration is not None and scheduled_duration > 24 * 30:
            flags.append("scheduled_duration_over_30_days")

        resolved_timestamp = parsed_timestamps["resolved_datetime"].loc[index]
        closed_timestamp = parsed_timestamps["closed_datetime"].loc[index]
        handling_duration: float | None = None
        for outcome_name, outcome_timestamp in (
            ("resolved", resolved_timestamp),
            ("closed", closed_timestamp),
        ):
            candidate = _duration_hours(start_timestamp, outcome_timestamp)
            if candidate is None:
                continue
            if candidate < 0:
                flags.append(f"negative_{outcome_name}_duration")
                continue
            handling_duration = candidate
            break
        if handling_duration is not None and handling_duration > 24 * 30:
            flags.append("handling_duration_over_30_days")

        local_start = start_timestamp.tz_convert(settings.local_timezone)
        vehicle_type = normalize_slug(row.get("veh_type"))
        route_path = _clean_route_path(row.get("route_path"), flags)
        truck_age_years = _normalize_truck_age(
            row.get("age_of_truck"),
            local_start.year,
            flags,
        )

        record: dict[str, Any] = {
            "event_id": event_id,
            "event_type": event_type,
            "event_cause": event_cause,
            "latitude": latitude,
            "longitude": longitude,
            "end_latitude": end_latitude,
            "end_longitude": end_longitude,
            "address": normalize_text(row.get("address")),
            "end_address": normalize_text(row.get("end_address")),
            "requires_road_closure": requires_road_closure,
            "start_datetime": _to_iso_utc(start_timestamp),
            "end_datetime": _to_iso_utc(end_timestamp),
            "status": normalize_slug(row.get("status")),
            "authenticated": parse_boolean(row.get("authenticated")),
            "modified_datetime": _to_iso_utc(
                parsed_timestamps["modified_datetime"].loc[index]
            ),
            "description": description,
            "vehicle_type": vehicle_type,
            "corridor": corridor,
            "priority": priority,
            "cargo_material": normalize_text(row.get("cargo_material")),
            "breakdown_reason": normalize_text(row.get("reason_breakdown")),
            "truck_age_years": truck_age_years,
            "created_datetime": _to_iso_utc(
                parsed_timestamps["created_date"].loc[index]
            ),
            "route_path": route_path,
            "police_station": normalize_text(row.get("police_station")),
            "zone": normalize_text(row.get("zone")),
            "junction": normalize_text(row.get("junction")),
            "closed_datetime": _to_iso_utc(closed_timestamp),
            "resolved_datetime": _to_iso_utc(resolved_timestamp),
            "scheduled_duration_hours": scheduled_duration,
            "handling_duration_hours": handling_duration,
            "event_hour_ist": int(local_start.hour),
            "event_weekday_ist": local_start.day_name(),
            "canonical_text": "",
            "data_quality_flags": "",
            "source_row_number": source_row_number,
            "source_hash": _source_row_hash(row),
            "ingested_at": ingested_at,
        }
        record["canonical_text"] = _build_canonical_text(record)
        unique_flags = sorted(set(flags))
        record["data_quality_flags"] = json.dumps(unique_flags, separators=(",", ":"))
        quality_counts.update(unique_flags)

        validated = HistoricalEventRecord.model_validate(record)
        records.append(validated.model_dump(mode="python"))

    events = pd.DataFrame.from_records(records)
    if len(events) != len(frame):
        raise PreprocessingError(
            f"Preprocessing produced {len(events)} rows from {len(frame)} source rows."
        )

    LOGGER.info(
        "Preprocessed %s events with %s distinct quality flags.",
        len(events),
        len(quality_counts),
    )
    return PreprocessingResult(
        events=events,
        exact_duplicate_rows_removed=duplicate_count,
        quality_flag_counts=dict(sorted(quality_counts.items())),
    )

