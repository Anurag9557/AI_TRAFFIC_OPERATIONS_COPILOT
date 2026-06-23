"""Tests for deterministic Operational Disruption Risk assessment."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.schemas import RiskBand
from src.services.retrieval import RetrievalMatch, RetrievalResult
from src.services.risk_scoring import assess_operational_disruption


def _create_cause_history(
    database_path: Path,
    cause: str,
    closure_values: list[int],
    durations: list[float | None],
) -> None:
    """Create the minimal SQLite history required by the scorer."""

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_cause TEXT NOT NULL,
                requires_road_closure INTEGER NOT NULL,
                handling_duration_hours REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO events VALUES (?, ?, ?)",
            [
                (cause, closure, duration)
                for closure, duration in zip(
                    closure_values,
                    durations,
                    strict=True,
                )
            ],
        )
        connection.commit()


def _retrieval(
    *,
    cause: str,
    event_type: str,
    corridor: str,
    priority: str,
    closures: list[int],
    durations: list[float | None],
    similarities: list[float] | None = None,
    candidate_count: int = 30,
) -> RetrievalResult:
    """Build a synthetic Phase 3 result without invoking FAISS."""

    if similarities is None:
        similarities = [0.82] * len(closures)
    matches = [
        RetrievalMatch(
            rank=index + 1,
            event_id=f"E{index + 1}",
            semantic_similarity=similarity,
            rerank_score=similarity,
            reasons=["Synthetic test match"],
            event={
                "event_cause": cause,
                "event_type": event_type,
                "corridor": corridor,
                "priority": priority,
                "requires_road_closure": closure,
                "handling_duration_hours": duration,
            },
        )
        for index, (closure, duration, similarity) in enumerate(
            zip(closures, durations, similarities, strict=True)
        )
    ]
    return RetrievalResult(
        matches=matches,
        confidence=0.8,
        candidate_count=candidate_count,
        model_name="test-model",
    )


def _event(cause: str, event_type: str, priority: str) -> dict[str, str]:
    return {
        "event_cause": cause,
        "event_type": event_type,
        "corridor": "Test Corridor",
        "priority": priority,
    }


def test_low_risk_event(tmp_path: Path) -> None:
    """Low closure, low priority, short planned work should remain Low."""

    database = tmp_path / "events.db"
    _create_cause_history(database, "low_cause", [0] * 20, [0.5] * 20)
    result = assess_operational_disruption(
        _event("low_cause", "planned", "Low"),
        _retrieval(
            cause="low_cause",
            event_type="planned",
            corridor="Test Corridor",
            priority="Low",
            closures=[0] * 5,
            durations=[0.5] * 5,
        ),
        database,
    )

    assert result.band is RiskBand.LOW
    assert 0 <= result.score <= 29


def test_moderate_risk_event(tmp_path: Path) -> None:
    """Mixed closure history and moderate handling should produce Moderate."""

    database = tmp_path / "events.db"
    _create_cause_history(
        database,
        "moderate_cause",
        [1] * 4 + [0] * 16,
        [4.0] * 20,
    )
    result = assess_operational_disruption(
        _event("moderate_cause", "planned", "Low"),
        _retrieval(
            cause="moderate_cause",
            event_type="planned",
            corridor="Test Corridor",
            priority="Low",
            closures=[1, 1, 0, 0, 0],
            durations=[4.0] * 5,
        ),
        database,
    )

    assert result.band is RiskBand.MODERATE
    assert 30 <= result.score <= 54


def test_high_risk_event(tmp_path: Path) -> None:
    """High priority with elevated closure and duration should produce High."""

    database = tmp_path / "events.db"
    _create_cause_history(
        database,
        "high_cause",
        [1] * 10 + [0] * 10,
        [12.0] * 20,
    )
    result = assess_operational_disruption(
        _event("high_cause", "planned", "High"),
        _retrieval(
            cause="high_cause",
            event_type="planned",
            corridor="Test Corridor",
            priority="High",
            closures=[1, 1, 1, 0, 0],
            durations=[12.0] * 5,
        ),
        database,
    )

    assert result.band is RiskBand.HIGH
    assert 55 <= result.score <= 74


def test_critical_risk_event(tmp_path: Path) -> None:
    """Unplanned high-priority evidence with full closure should be Critical."""

    database = tmp_path / "events.db"
    _create_cause_history(database, "critical_cause", [1] * 20, [100.0] * 20)
    result = assess_operational_disruption(
        _event("critical_cause", "unplanned", "High"),
        _retrieval(
            cause="critical_cause",
            event_type="unplanned",
            corridor="Test Corridor",
            priority="High",
            closures=[1] * 5,
            durations=[100.0] * 5,
        ),
        database,
    )

    assert result.band is RiskBand.CRITICAL
    assert 75 <= result.score <= 100


def test_small_cohort_reduces_confidence(tmp_path: Path) -> None:
    """One weak evidence record must be less confident than a full cohort."""

    database = tmp_path / "events.db"
    _create_cause_history(database, "cohort_cause", [0] * 20, [2.0] * 20)
    event = _event("cohort_cause", "planned", "Low")

    small = assess_operational_disruption(
        event,
        _retrieval(
            cause="cohort_cause",
            event_type="planned",
            corridor="Test Corridor",
            priority="Low",
            closures=[0],
            durations=[2.0],
            similarities=[0.55],
            candidate_count=1,
        ),
        database,
    )
    full = assess_operational_disruption(
        event,
        _retrieval(
            cause="cohort_cause",
            event_type="planned",
            corridor="Test Corridor",
            priority="Low",
            closures=[0] * 5,
            durations=[2.0] * 5,
            similarities=[0.8] * 5,
            candidate_count=20,
        ),
        database,
    )

    assert small.confidence < full.confidence
    assert small.components["evidence_uncertainty"] > 0
    assert any("Limited number" in item for item in small.limitations)


def test_missing_duration_is_handled_without_fabrication(tmp_path: Path) -> None:
    """Missing handling duration must score zero and produce a limitation."""

    database = tmp_path / "events.db"
    _create_cause_history(database, "missing_duration", [0] * 20, [None] * 20)
    result = assess_operational_disruption(
        _event("missing_duration", "planned", "Low"),
        _retrieval(
            cause="missing_duration",
            event_type="planned",
            corridor="Test Corridor",
            priority="Low",
            closures=[0] * 5,
            durations=[None] * 5,
        ),
        database,
    )

    assert result.components["historical_handling_duration"] == 0
    assert result.components["evidence_uncertainty"] >= 1
    assert any("No valid handling-duration" in item for item in result.limitations)

