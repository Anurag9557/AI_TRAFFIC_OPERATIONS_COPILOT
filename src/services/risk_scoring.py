"""Explainable Operational Disruption Risk scoring.

This module converts Phase 3 retrieval results into a deterministic 0-100
assessment. It does not use an LLM, classifier, or traffic-flow model.

Formula (maximum contribution):

- historical closure requirement among retrieved matches: 30 points;
- current event priority: 20 points;
- event-cause burden from all matching-cause SQLite records: 20 points;
- median historical handling duration among retrieved matches: 15 points;
- planned/unplanned urgency: 10 points;
- precautionary evidence uncertainty: 5 points.

The result represents operational disruption planning risk. The source dataset
does not contain speeds, vehicle counts, queues, or travel-time delay, so this
score must never be described as a congestion prediction.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from src.data.preprocessing import (
    normalize_corridor,
    normalize_event_cause,
    normalize_priority,
    normalize_slug,
)
from src.schemas import RiskAssessment, RiskBand, StructuredEvent
from src.services.retrieval import RetrievalMatch, RetrievalResult

LOGGER = logging.getLogger(__name__)

CLOSURE_WEIGHT = 30.0
PRIORITY_WEIGHT = 20.0
CAUSE_BURDEN_WEIGHT = 20.0
HANDLING_DURATION_WEIGHT = 15.0
URGENCY_WEIGHT = 10.0
UNCERTAINTY_WEIGHT = 5.0


class RiskScoringError(RuntimeError):
    """Raised when an assessment cannot be calculated safely."""


def _event_dict(event: StructuredEvent | Mapping[str, Any]) -> dict[str, Any]:
    """Convert supported event inputs into normalized dictionary values."""

    if isinstance(event, BaseModel):
        values = event.model_dump(mode="python")
    else:
        values = dict(event)
    return {
        **values,
        "event_cause": (
            normalize_event_cause(values.get("event_cause"))
            if values.get("event_cause")
            else None
        ),
        "corridor": normalize_corridor(values.get("corridor")),
        "event_type": normalize_slug(values.get("event_type")),
        "priority": normalize_priority(values.get("priority")),
    }


def _matches_frame(matches: list[RetrievalMatch]) -> pd.DataFrame:
    """Convert retrieval matches into a compact analytics DataFrame."""

    rows: list[dict[str, Any]] = []
    for match in matches:
        rows.append(
            {
                "event_id": match.event_id,
                "semantic_similarity": float(match.semantic_similarity),
                "rerank_score": float(match.rerank_score),
                "event_cause": match.event.get("event_cause"),
                "corridor": match.event.get("corridor"),
                "event_type": match.event.get("event_type"),
                "priority": match.event.get("priority"),
                "requires_road_closure": match.event.get(
                    "requires_road_closure"
                ),
                "handling_duration_hours": match.event.get(
                    "handling_duration_hours"
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_cause_history(database_path: Path, event_cause: str | None) -> pd.DataFrame:
    """Load only cause-level fields required for deterministic burden scoring."""

    if event_cause is None:
        return pd.DataFrame(
            columns=["requires_road_closure", "handling_duration_hours"]
        )
    if not database_path.exists():
        raise RiskScoringError(f"SQLite database does not exist: {database_path}")

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database_path)
        return pd.read_sql_query(
            """
            SELECT requires_road_closure, handling_duration_hours
            FROM events
            WHERE event_cause = ?
            """,
            connection,
            params=(event_cause,),
        )
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise RiskScoringError(
            f"Unable to load cause history from SQLite: {exc}"
        ) from exc
    finally:
        if connection is not None:
            connection.close()


def _valid_numeric(series: pd.Series) -> pd.Series:
    """Return finite, non-negative numeric observations."""

    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[numeric.notna() & numeric.ge(0) & numeric.map(math.isfinite)]


def _closure_component(frame: pd.DataFrame) -> tuple[float, float | None]:
    """Score historical road-closure requirement among retrieved matches."""

    if frame.empty or "requires_road_closure" not in frame:
        return 0.0, None
    values = pd.to_numeric(
        frame["requires_road_closure"],
        errors="coerce",
    ).dropna()
    values = values[values.isin([0, 1])]
    if values.empty:
        return 0.0, None
    rate = float(values.mean())
    return round(CLOSURE_WEIGHT * rate, 2), rate


def _priority_component(priority: str | None) -> float:
    """Map the source dataset's High/Low priority to a fixed contribution."""

    if priority == "High":
        return PRIORITY_WEIGHT
    if priority == "Low":
        return 5.0
    return 10.0


def _duration_burden_factor(median_hours: float | None) -> float:
    """Convert median handling hours to a transparent 0-1 burden factor."""

    if median_hours is None:
        return 0.0
    if median_hours <= 1:
        return 0.15
    if median_hours <= 6:
        return 0.35
    if median_hours <= 24:
        return 0.60
    if median_hours <= 72:
        return 0.80
    return 1.0


def _cause_burden_component(
    cause_history: pd.DataFrame,
) -> tuple[float, float | None, float | None, int]:
    """Score broad cause history using closure tendency and handling duration.

    Within the 20-point component, closure tendency contributes 65% and
    cause-level median handling duration contributes 35%. A 50% cause closure
    rate reaches the closure sub-score cap to prevent tiny extreme cohorts from
    making the total unbounded.
    """

    if cause_history.empty:
        return 0.0, None, None, 0

    closure_values = pd.to_numeric(
        cause_history["requires_road_closure"],
        errors="coerce",
    ).dropna()
    closure_values = closure_values[closure_values.isin([0, 1])]
    closure_rate = (
        float(closure_values.mean()) if not closure_values.empty else None
    )
    durations = _valid_numeric(cause_history["handling_duration_hours"])
    median_duration = float(durations.median()) if not durations.empty else None

    closure_factor = (
        min(closure_rate / 0.50, 1.0) if closure_rate is not None else 0.0
    )
    duration_factor = _duration_burden_factor(median_duration)
    score = CAUSE_BURDEN_WEIGHT * (
        0.65 * closure_factor + 0.35 * duration_factor
    )
    return (
        round(score, 2),
        closure_rate,
        median_duration,
        int(len(cause_history)),
    )


def _handling_duration_component(
    frame: pd.DataFrame,
) -> tuple[float, float | None, int]:
    """Score retrieved-cohort median recorded handling duration."""

    if frame.empty or "handling_duration_hours" not in frame:
        return 0.0, None, 0
    durations = _valid_numeric(frame["handling_duration_hours"])
    if durations.empty:
        return 0.0, None, 0

    median_hours = float(durations.median())
    factor = _duration_burden_factor(median_hours)
    return (
        round(HANDLING_DURATION_WEIGHT * factor, 2),
        median_hours,
        int(len(durations)),
    )


def _urgency_component(event_type: str | None) -> float:
    """Give unplanned events a larger immediate operational contribution."""

    if event_type == "unplanned":
        return URGENCY_WEIGHT
    if event_type == "planned":
        return 4.0
    return 6.0


def _metadata_match_quality(
    event: dict[str, Any],
    matches: list[RetrievalMatch],
) -> float:
    """Measure exact metadata agreement across the returned retrieval cohort."""

    fields = ("event_cause", "corridor", "event_type", "priority")
    available = [field for field in fields if event.get(field) is not None]
    if not matches:
        return 0.0
    if not available:
        return 0.5

    comparisons = 0
    exact_matches = 0
    for match in matches:
        for field in available:
            comparisons += 1
            if match.event.get(field) == event[field]:
                exact_matches += 1
    return exact_matches / comparisons if comparisons else 0.0


def _similarity_quality(matches: list[RetrievalMatch]) -> float:
    """Average the top three non-negative semantic similarities."""

    if not matches:
        return 0.0
    values = [
        max(0.0, min(1.0, float(match.semantic_similarity)))
        for match in matches[:3]
    ]
    return sum(values) / len(values)


def _confidence_score(
    event: dict[str, Any],
    retrieval: RetrievalResult,
) -> tuple[float, float, float, float]:
    """Calculate confidence from cohort size, metadata, and similarity."""

    visible_cohort = min(len(retrieval.matches) / 5.0, 1.0)
    candidate_cohort = min(retrieval.candidate_count / 20.0, 1.0)
    cohort_quality = 0.5 * visible_cohort + 0.5 * candidate_cohort
    metadata_quality = _metadata_match_quality(event, retrieval.matches)
    similarity_quality = _similarity_quality(retrieval.matches)
    confidence = (
        0.35 * cohort_quality
        + 0.35 * metadata_quality
        + 0.30 * similarity_quality
    )
    return (
        round(max(0.0, min(1.0, confidence)), 4),
        cohort_quality,
        metadata_quality,
        similarity_quality,
    )


def _uncertainty_component(
    retrieval: RetrievalResult,
    similarity_quality: float,
    metadata_quality: float,
    duration_sample_size: int,
) -> float:
    """Add at most five precautionary points for weak historical evidence."""

    cohort_size = len(retrieval.matches)
    if cohort_size < 3:
        score = 3.0
    elif cohort_size < 5:
        score = 2.0
    else:
        score = 0.0

    if similarity_quality < 0.45:
        score += 1.5
    elif similarity_quality < 0.60:
        score += 0.75
    if duration_sample_size == 0:
        score += 1.0
    if metadata_quality < 0.25:
        score += 0.5
    return round(min(UNCERTAINTY_WEIGHT, score), 2)


def _risk_band(score: float) -> RiskBand:
    """Map the numeric score to the approved operational bands."""

    if score < 30:
        return RiskBand.LOW
    if score < 55:
        return RiskBand.MODERATE
    if score < 75:
        return RiskBand.HIGH
    return RiskBand.CRITICAL


def assess_operational_disruption(
    event: StructuredEvent | Mapping[str, Any],
    retrieval: RetrievalResult,
    database_path: Path,
) -> RiskAssessment:
    """Create a fully explainable Operational Disruption Risk assessment."""

    normalized_event = _event_dict(event)
    frame = _matches_frame(retrieval.matches)
    cause_history = _load_cause_history(
        database_path,
        normalized_event.get("event_cause"),
    )

    closure_score, closure_rate = _closure_component(frame)
    priority_score = _priority_component(normalized_event.get("priority"))
    (
        cause_score,
        cause_closure_rate,
        cause_median_duration,
        cause_sample_size,
    ) = _cause_burden_component(cause_history)
    (
        duration_score,
        cohort_median_duration,
        duration_sample_size,
    ) = _handling_duration_component(frame)
    urgency_score = _urgency_component(normalized_event.get("event_type"))
    (
        confidence,
        cohort_quality,
        metadata_quality,
        similarity_quality,
    ) = _confidence_score(normalized_event, retrieval)
    uncertainty_score = _uncertainty_component(
        retrieval,
        similarity_quality,
        metadata_quality,
        duration_sample_size,
    )

    components = {
        "historical_closure_requirement": closure_score,
        "event_priority": priority_score,
        "event_cause_operational_burden": cause_score,
        "historical_handling_duration": duration_score,
        "planned_unplanned_urgency": urgency_score,
        "evidence_uncertainty": uncertainty_score,
    }
    total_score = round(min(100.0, max(0.0, sum(components.values()))), 1)

    reasons: list[str] = []
    if closure_rate is None:
        reasons.append(
            "Retrieved events did not provide a usable road-closure requirement."
        )
    else:
        reasons.append(
            "Similar historical events required road closure "
            f"{closure_rate:.0%} of the time."
        )

    priority = normalized_event.get("priority")
    if priority is None:
        reasons.append(
            "Event priority was unavailable, so a neutral priority contribution "
            "was used."
        )
    else:
        reasons.append(
            f"The current event priority is {priority}, contributing "
            f"{priority_score:.0f} of 20 points."
        )

    cause = normalized_event.get("event_cause") or "unknown cause"
    if cause_sample_size:
        cause_text = cause.replace("_", " ")
        detail = (
            f"Historical {cause_text} records required closure "
            f"{cause_closure_rate:.0%} of the time"
            if cause_closure_rate is not None
            else f"Historical {cause_text} records had no usable closure rate"
        )
        if cause_median_duration is not None:
            detail += (
                f" and had a median recorded handling duration of "
                f"{cause_median_duration:.1f} hours"
            )
        reasons.append(f"{detail} across {cause_sample_size} records.")
    else:
        reasons.append(
            "No same-cause history was available for the cause-burden component."
        )

    if cohort_median_duration is None:
        reasons.append(
            "No valid handling duration was available in the retrieved cohort."
        )
    else:
        reasons.append(
            f"The retrieved cohort's median recorded handling duration was "
            f"{cohort_median_duration:.1f} hours based on "
            f"{duration_sample_size} events."
        )

    event_type = normalized_event.get("event_type")
    if event_type == "unplanned":
        reasons.append(
            "The event is unplanned, increasing immediate operational urgency."
        )
    elif event_type == "planned":
        reasons.append(
            "The event is planned, allowing preparation and reducing the urgency "
            "contribution."
        )
    else:
        reasons.append(
            "Event planning type was unavailable, so a neutral urgency "
            "contribution was used."
        )

    if uncertainty_score:
        reasons.append(
            f"Weak or incomplete evidence added {uncertainty_score:.1f} "
            "precautionary risk points."
        )

    limitations = [
        "No direct traffic-speed, vehicle-count, queue-length, or travel-time "
        "data is available.",
        "This score represents operational disruption risk, not a congestion "
        "prediction.",
        "Recorded handling duration reflects database closure/resolution timing "
        "and may not equal road-clearance time.",
    ]
    if len(retrieval.matches) < 5 or retrieval.candidate_count < 10:
        limitations.append(
            "Limited number of matching historical events reduced confidence."
        )
    if duration_sample_size == 0:
        limitations.append(
            "No valid handling-duration observations were available in the "
            "retrieved cohort."
        )
    if metadata_quality < 0.5:
        limitations.append(
            "Retrieved events had limited exact metadata agreement with the "
            "current event."
        )
    if cause_sample_size < 10:
        limitations.append(
            "The event-cause burden is based on a small same-cause history."
        )

    LOGGER.info(
        "Operational risk score %.1f (%s), confidence %.3f; evidence quality "
        "cohort=%.2f metadata=%.2f similarity=%.2f.",
        total_score,
        _risk_band(total_score).value,
        confidence,
        cohort_quality,
        metadata_quality,
        similarity_quality,
    )
    return RiskAssessment(
        score=total_score,
        band=_risk_band(total_score),
        confidence=confidence,
        components=components,
        reasons=reasons,
        limitations=limitations,
    )
