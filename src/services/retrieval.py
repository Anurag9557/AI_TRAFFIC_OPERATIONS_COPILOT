"""Semantic historical-event retrieval with explainable metadata reranking."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from math import radians
from math import sin
from math import cos
from math import sqrt
from math import atan2



import numpy as np

from src.data.preprocessing import (
    normalize_corridor,
    normalize_event_cause,
    normalize_priority,
    normalize_slug,
)
from src.services.embeddings import (
    EmbeddingError,
    IndexMetadata,
    TextEncoder,
    load_faiss_index,
)

LOGGER = logging.getLogger(__name__)

SEMANTIC_WEIGHT = 0.50
CAUSE_WEIGHT = 0.15
CORRIDOR_WEIGHT = 0.10
EVENT_TYPE_WEIGHT = 0.05
PRIORITY_WEIGHT = 0.05
LOCATION_WEIGHT = 0.15

EVENT_SELECT_COLUMNS = (
    "event_id",
    "event_type",
    "event_cause",
    "description",
    "corridor",
    "priority",
    "requires_road_closure",
    "start_datetime",
    "status",
    "address",
    "police_station",
    "vehicle_type",
    "scheduled_duration_hours",
    "handling_duration_hours",
    "latitude",
    "longitude",
)


class RetrievalError(RuntimeError):
    """Raised when a semantic search cannot be completed."""


@dataclass(frozen=True)
class RetrievalQuery:

    text: str
    event_cause: str | None = None
    corridor: str | None = None
    event_type: str | None = None
    priority: str | None = None

    latitude: float | None = None
    longitude: float | None = None


@dataclass(frozen=True)
class RetrievalMatch:

    rank: int
    event_id: str
    semantic_similarity: float
    rerank_score: float
    distance_km: float | None
    reasons: list[str]
    event: dict[str, Any]


@dataclass(frozen=True)
class RetrievalResult:
    """Complete retrieval response returned to later application phases."""

    matches: list[RetrievalMatch]
    confidence: float
    candidate_count: int
    model_name: str


def _normalized_query(query: RetrievalQuery) -> RetrievalQuery:
    """Normalize query metadata with the same rules used during ingestion."""

    text = query.text.strip()
    if not text:
        raise RetrievalError("Search text cannot be empty.")
    return RetrievalQuery(
        text=text,
        event_cause=(
            normalize_event_cause(query.event_cause)
            if query.event_cause
            else None
        ),
        corridor=normalize_corridor(query.corridor),
        event_type=normalize_slug(query.event_type),
        priority=normalize_priority(query.priority),

        latitude=query.latitude,
        longitude=query.longitude,
    )

def calculate_distance_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    R = 6371

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a),
    )

    return R * c


def calculate_rerank_score(
    semantic_similarity: float,
    event: dict[str, Any],
    query: RetrievalQuery,
) -> tuple[float, list[str]]:
    """Combine cosine similarity and exact metadata matches.

    Weights are fixed and intentionally simple:
    semantic 65%, cause 15%, corridor 10%, event type 5%, priority 5%.
    """

    semantic = max(0.0, min(1.0, float(semantic_similarity)))
    score = SEMANTIC_WEIGHT * semantic
    reasons: list[str] = []
    distance_km = None

    if query.event_cause and event.get("event_cause") == query.event_cause:
        score += CAUSE_WEIGHT
        reasons.append("Same event cause")
    if query.corridor and event.get("corridor") == query.corridor:
        score += CORRIDOR_WEIGHT
        reasons.append("Same corridor")
    if query.event_type and event.get("event_type") == query.event_type:
        score += EVENT_TYPE_WEIGHT
        reasons.append("Same planned/unplanned type")
    if query.priority and event.get("priority") == query.priority:
        score += PRIORITY_WEIGHT
        reasons.append("Similar priority")
    
    if (
    query.latitude is not None
    and query.longitude is not None
    and event.get("latitude") is not None
    and event.get("longitude") is not None
):

        distance = calculate_distance_km(
            query.latitude,
            query.longitude,
            float(event["latitude"]),
            float(event["longitude"]),
        )

        distance_km = distance

        if distance <= 1:
            location_score = 1.0
        elif distance <= 5:
            location_score = 0.8
        elif distance <= 10:
            location_score = 0.5
        elif distance <= 20:
            location_score = 0.2
        else:
            location_score = 0.0

        score += LOCATION_WEIGHT * location_score

        reasons.append(
            f"Located {distance:.1f} km from current event"
        )

    if semantic >= 0.75:
        reasons.append("Highly similar event description and context")
    elif semantic >= 0.50:
        reasons.append("Similar event description and context")
    else:
        reasons.append("Nearest available semantic context")

    return min(1.0, score), reasons, distance_km


def _retrieval_confidence(matches: list[RetrievalMatch]) -> float:
    """Calculate a compact confidence score from strength and consistency."""

    if not matches:
        return 0.0
    scores = [match.rerank_score for match in matches[:3]]
    top_score = scores[0]
    top_three_mean = sum(scores) / len(scores)
    confidence = 0.70 * top_score + 0.30 * top_three_mean
    return round(max(0.0, min(1.0, confidence)), 4)


class HistoricalEventRetriever:
    """Load one local FAISS index and search SQLite-backed event metadata."""

    def __init__(
        self,
        database_path: Path,
        index_path: Path,
        mapping_path: Path,
        encoder: TextEncoder,
    ) -> None:
        self.database_path = database_path
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.encoder = encoder
        try:
            self.index, self.metadata = load_faiss_index(index_path, mapping_path)
        except EmbeddingError as exc:
            raise RetrievalError(str(exc)) from exc
        self._validate_model(self.metadata)

    def _validate_model(self, metadata: IndexMetadata) -> None:
        """Prevent querying an index with a different embedding model."""

        if metadata.model_name != self.encoder.model_name:
            raise RetrievalError(
                f"Index model is {metadata.model_name}, but the query encoder is "
                f"{self.encoder.model_name}. Rebuild the index or use the matching "
                "model."
            )

    def _load_events(self, event_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch candidate event records from SQLite without duplicating them."""

        if not event_ids:
            return {}
        placeholders = ",".join("?" for _ in event_ids)
        columns = ", ".join(EVENT_SELECT_COLUMNS)
        try:
            with sqlite3.connect(self.database_path) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    f"SELECT {columns} FROM events "
                    f"WHERE event_id IN ({placeholders})",
                    event_ids,
                ).fetchall()
        except sqlite3.Error as exc:
            raise RetrievalError(f"Unable to load candidates from SQLite: {exc}") from exc
        return {str(row["event_id"]): dict(row) for row in rows}

    def search(
        self,
        query: RetrievalQuery,
        top_k: int = 5,
        candidate_k: int = 30,
    ) -> RetrievalResult:
        """Return top similar events after exact metadata-aware reranking."""

        if top_k <= 0:
            raise RetrievalError("top_k must be positive.")
        if candidate_k < top_k:
            raise RetrievalError("candidate_k must be greater than or equal to top_k.")

        normalized_query = _normalized_query(query)
        try:
            query_vector = self.encoder.encode([normalized_query.text], batch_size=1)
        except EmbeddingError as exc:
            raise RetrievalError(str(exc)) from exc

        if query_vector.shape != (1, self.metadata.dimension):
            raise RetrievalError(
                f"Query embedding shape {query_vector.shape} does not match index "
                f"dimension {self.metadata.dimension}."
            )
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(query_vector[0]))
        if norm == 0:
            raise RetrievalError("Query embedding has zero length.")
        query_vector /= norm

        search_count = min(candidate_k, self.metadata.event_count)
        similarities, positions = self.index.search(query_vector, search_count)
        candidates: list[tuple[str, float]] = []
        for position, similarity in zip(positions[0], similarities[0], strict=True):
            if position < 0:
                continue
            candidates.append(
                (self.metadata.event_ids[int(position)], float(similarity))
            )

        events = self._load_events([event_id for event_id, _ in candidates])
        scored: list[tuple[float, float, str, list[str], dict[str, Any]]] = []
        for event_id, similarity in candidates:
            event = events.get(event_id)
            if event is None:
                LOGGER.warning("FAISS event %s is missing from SQLite.", event_id)
                continue
            rerank_score, reasons, distance_km = calculate_rerank_score(
                similarity,
                event,
                normalized_query,
            )
            scored.append((rerank_score, similarity, event_id, reasons, event , distance_km))

        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        matches = [
            RetrievalMatch(
                rank=rank,
                event_id=event_id,
                semantic_similarity=round(
                    max(-1.0, min(1.0, similarity)),
                    4,
                ),
                rerank_score=round(score, 4),
                distance_km=(
                    round(distance_km, 2)
                    if distance_km is not None
                    else None
                ),
                reasons=reasons,
                event=event,
            )
            for rank, (
                score,
                similarity,
                event_id,
                reasons,
                event,
                distance_km,
            ) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]
        return RetrievalResult(
            matches=matches,
            confidence=_retrieval_confidence(matches),
            candidate_count=len(scored),
            model_name=self.metadata.model_name,
        )
