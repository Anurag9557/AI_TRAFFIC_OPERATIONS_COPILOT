"""Pydantic schemas shared across ingestion and future graph phases."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    """Supported event planning categories."""

    PLANNED = "planned"
    UNPLANNED = "unplanned"


class EventPriority(StrEnum):
    """Priority labels observed in the source dataset."""

    HIGH = "High"
    LOW = "Low"


class RiskBand(StrEnum):
    """Operational Disruption Risk bands reserved for Phase 4."""

    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


class StructuredEvent(BaseModel):
    """Normalized event input that the future understanding node will produce."""

    model_config = ConfigDict(extra="forbid")

    event_type: EventType | None = None
    event_cause: str | None = None
    description: str | None = None
    corridor: str | None = None
    priority: EventPriority | None = None
    requires_road_closure: bool | None = None
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    police_station: str | None = None
    vehicle_type: str | None = None
    operational_notes: list[str] = Field(default_factory=list)


class HistoricalStatistics(BaseModel):
    """Summary statistics produced by future historical retrieval."""

    model_config = ConfigDict(extra="forbid")

    cohort_size: int = 0
    cause_match_count: int = 0
    corridor_match_count: int = 0
    closure_rate: float | None = None
    high_priority_rate: float | None = None
    median_handling_hours: float | None = None
    handling_duration_sample_size: int = 0
    planned_event_rate: float | None = None
    data_quality_notes: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """Explainable risk result reserved for the deterministic analytics phase."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=100)
    band: RiskBand
    components: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    limitations: list[str] = Field(default_factory=list)


class ResourcePlan(BaseModel):
    """Policy-based deployment recommendation reserved for a later phase."""

    model_config = ConfigDict(extra="forbid")

    manpower_min: int = Field(ge=0)
    manpower_max: int = Field(ge=0)
    barricades_min: int = Field(ge=0)
    barricades_max: int = Field(ge=0)
    support_vehicles: list[str] = Field(default_factory=list)
    deployment_notes: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    policy_disclaimer: str


class HistoricalEventRecord(BaseModel):
    """One cleaned historical event ready for SQLite insertion."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: str
    event_cause: str
    latitude: float
    longitude: float
    end_latitude: float | None = None
    end_longitude: float | None = None
    address: str | None = None
    end_address: str | None = None
    requires_road_closure: int
    start_datetime: str
    end_datetime: str | None = None
    status: str | None = None
    authenticated: int | None = None
    modified_datetime: str | None = None
    description: str | None = None
    vehicle_type: str | None = None
    corridor: str | None = None
    priority: str | None = None
    cargo_material: str | None = None
    breakdown_reason: str | None = None
    truck_age_years: float | None = None
    created_datetime: str | None = None
    route_path: str | None = None
    police_station: str | None = None
    zone: str | None = None
    junction: str | None = None
    closed_datetime: str | None = None
    resolved_datetime: str | None = None
    scheduled_duration_hours: float | None = None
    handling_duration_hours: float | None = None
    event_hour_ist: int | None = None
    event_weekday_ist: str | None = None
    canonical_text: str
    data_quality_flags: str
    source_row_number: int
    source_hash: str
    ingested_at: str


class CaseRecord(BaseModel):
    """Database representation of a generated copilot case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    user_input: str
    structured_event_json: str
    similar_event_ids_json: str = "[]"
    historical_stats_json: str = "{}"
    risk_assessment_json: str = "{}"
    resource_plan_json: str = "{}"
    report: str = ""
    created_at: str


class FeedbackCreate(BaseModel):
    """Optional post-event feedback accepted by the MVP database."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    actual_road_closure: bool | None = None
    actual_handling_duration_hours: float | None = Field(default=None, ge=0)
    actual_manpower: int | None = Field(default=None, ge=0)
    actual_barricades: int | None = Field(default=None, ge=0)
    support_vehicles_used: list[str] = Field(default_factory=list)
    observed_operational_impact: str | None = None
    notes: str | None = None


class IngestionSummary(BaseModel):
    """Machine-readable result returned by a complete ingestion run."""

    model_config = ConfigDict(extra="forbid")

    source_path: str
    source_sha256: str
    database_path: str
    source_rows: int
    exact_duplicate_rows_removed: int
    inserted_rows: int
    date_range_start: str | None = None
    date_range_end: str | None = None
    table_counts: dict[str, int]
    quality_flag_counts: dict[str, int]
    completed_at: str
    notes: list[str] = Field(default_factory=list)


JsonObject = dict[str, Any]

