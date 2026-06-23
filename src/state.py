"""Shared state passed through the five-node Traffic Operations graph."""

from __future__ import annotations

from typing import Any, TypedDict

from src.schemas import (
    HistoricalStatistics,
    ResourcePlan,
    RiskAssessment,
    StructuredEvent,
)
from src.services.retrieval import RetrievalResult


class TrafficOpsState(TypedDict, total=False):
    """Compact state passed between the future five LangGraph nodes."""

    case_id: str
    user_input: str
    form_data: dict[str, Any]
    event: StructuredEvent
    validation_errors: list[str]
    similar_events: list[dict[str, Any]]
    evidence_event_ids: list[str]
    historical_stats: HistoricalStatistics
    retrieval_confidence: float
    retrieval_result: RetrievalResult
    risk_assessment: RiskAssessment
    resource_plan: ResourcePlan
    report: str
    case_saved: bool
    error: str | None
