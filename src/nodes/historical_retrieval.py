"""Historical Retrieval Agent that reuses the Phase 3 FAISS engine."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from src.schemas import HistoricalStatistics
from src.services.retrieval import RetrievalQuery, RetrievalResult
from src.state import TrafficOpsState


class Retriever(Protocol):
    """Minimal search contract implemented by HistoricalEventRetriever."""

    def search(
        self,
        query: RetrievalQuery,
        top_k: int = 5,
        candidate_k: int = 30,
    ) -> RetrievalResult:
        """Search historical events."""


def _query_text(event: Any) -> str:
    """Build a short semantic query from normalized event facts."""

    values = event.model_dump(mode="python")
    parts = [
        values.get("description"),
        values.get("event_type"),
        values.get("event_cause"),
        values.get("corridor"),
        values.get("priority"),
        values.get("address"),
        values.get("vehicle_type"),
    ]
    return ". ".join(str(value) for value in parts if value)


def _historical_statistics(result: RetrievalResult) -> HistoricalStatistics:
    """Summarize the returned evidence with deterministic Pandas analytics."""

    rows = [match.event for match in result.matches]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return HistoricalStatistics(
            cohort_size=0,
            data_quality_notes=["No similar historical events were retrieved."],
        )

    closure = pd.to_numeric(
        frame.get("requires_road_closure"),
        errors="coerce",
    ).dropna()
    durations = pd.to_numeric(
        frame.get("handling_duration_hours"),
        errors="coerce",
    ).dropna()
    durations = durations[durations.ge(0)]
    priority = frame.get("priority", pd.Series(dtype=str))
    event_type = frame.get("event_type", pd.Series(dtype=str))

    top = result.matches[0].event if result.matches else {}
    cause = top.get("event_cause")
    corridor = top.get("corridor")
    notes: list[str] = []
    if durations.empty:
        notes.append("Retrieved events contain no valid handling duration.")
    if len(frame) < 5:
        notes.append("Fewer than five historical matches were available.")

    return HistoricalStatistics(
        cohort_size=len(frame),
        cause_match_count=int((frame.get("event_cause") == cause).sum())
        if cause is not None
        else 0,
        corridor_match_count=int((frame.get("corridor") == corridor).sum())
        if corridor is not None
        else 0,
        closure_rate=float(closure.mean()) if not closure.empty else None,
        high_priority_rate=float((priority == "High").mean())
        if len(priority)
        else None,
        median_handling_hours=float(durations.median())
        if not durations.empty
        else None,
        handling_duration_sample_size=int(len(durations)),
        planned_event_rate=float((event_type == "planned").mean())
        if len(event_type)
        else None,
        data_quality_notes=notes,
    )


def create_historical_retrieval_node(retriever: Retriever):
    """Create a retrieval node around the existing Phase 3 service."""

    def historical_retrieval(state: TrafficOpsState) -> dict[str, Any]:
        event = state["event"]
        result = retriever.search(
            RetrievalQuery(
                text=_query_text(event),
                event_cause=event.event_cause,
                corridor=event.corridor,
                event_type=event.event_type.value if event.event_type else None,
                priority=event.priority.value if event.priority else None,
                latitude=event.latitude,
                longitude=event.longitude,
            ),
            top_k=5,
            candidate_k=30,
        )
        return {
            "retrieval_result": result,
            "similar_events": [
                {
                    "rank": match.rank,
                    "event_id": match.event_id,
                    "semantic_similarity": match.semantic_similarity,
                    "rerank_score": match.rerank_score,
                    "reasons": match.reasons,
                    "distance_km": match.distance_km,
                    **match.event,
                }
                for match in result.matches
            ],
            "evidence_event_ids": [match.event_id for match in result.matches],
            "historical_stats": _historical_statistics(result),
            "retrieval_confidence": result.confidence,
        }

    return historical_retrieval
