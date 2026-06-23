"""Phase 3 tests for FAISS creation, loading, search, and reranking."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from src.services.embeddings import (
    build_faiss_index,
    load_faiss_index,
)
from src.services.retrieval import (
    HistoricalEventRetriever,
    RetrievalQuery,
    calculate_rerank_score,
)


class FakeEncoder:
    """Small deterministic encoder that avoids downloading a model in tests."""

    model_name = "fake-test-model"

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.casefold()
            vectors.append(
                [
                    float("breakdown" in lowered or "truck" in lowered),
                    float("accident" in lowered or "collision" in lowered),
                    float("construction" in lowered or "road work" in lowered),
                    float("mysore" in lowered or "corridor" in lowered),
                ]
            )
        matrix = np.asarray(vectors, dtype=np.float32)
        zero_rows = np.linalg.norm(matrix, axis=1) == 0
        matrix[zero_rows, 0] = 0.5
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix


def _create_test_database(path: Path) -> None:
    """Create a compact SQLite fixture containing all retrieval output fields."""

    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                event_cause TEXT NOT NULL,
                description TEXT,
                corridor TEXT,
                priority TEXT,
                requires_road_closure INTEGER NOT NULL,
                start_datetime TEXT NOT NULL,
                status TEXT,
                address TEXT,
                police_station TEXT,
                vehicle_type TEXT,
                scheduled_duration_hours REAL,
                handling_duration_hours REAL,
                canonical_text TEXT NOT NULL
            );
            """
        )
        rows = [
            (
                "E1",
                "unplanned",
                "vehicle_breakdown",
                "Heavy truck breakdown on Mysore Road",
                "Mysore Road",
                "High",
                0,
                "2024-03-01T10:00:00+00:00",
                "closed",
                "Mysore Road",
                "Test Station",
                "heavy_vehicle",
                None,
                1.0,
                "unplanned heavy truck vehicle breakdown Mysore Road High",
            ),
            (
                "E2",
                "unplanned",
                "vehicle_breakdown",
                "Car breakdown on another corridor",
                "Tumkur Road",
                "Low",
                0,
                "2024-03-02T10:00:00+00:00",
                "closed",
                "Tumkur Road",
                "Test Station",
                "private_car",
                None,
                0.5,
                "unplanned car vehicle breakdown Tumkur Road Low",
            ),
            (
                "E3",
                "unplanned",
                "accident",
                "Collision on Mysore Road",
                "Mysore Road",
                "High",
                0,
                "2024-03-03T10:00:00+00:00",
                "closed",
                "Mysore Road",
                "Test Station",
                "private_car",
                None,
                0.8,
                "unplanned accident collision Mysore Road High",
            ),
            (
                "E4",
                "planned",
                "construction",
                "Planned road work",
                "Mysore Road",
                "High",
                1,
                "2024-03-04T10:00:00+00:00",
                "closed",
                "Mysore Road",
                "Test Station",
                None,
                8.0,
                8.0,
                "planned construction road work Mysore Road High",
            ),
        ]
        connection.executemany(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        connection.commit()


def test_index_creation_and_loading(tmp_path: Path) -> None:
    """Persisted index dimensions and ordered ID mapping must round-trip."""

    database = tmp_path / "events.db"
    index_path = tmp_path / "events.faiss"
    mapping_path = tmp_path / "mapping.json"
    _create_test_database(database)

    summary = build_faiss_index(
        database,
        index_path,
        mapping_path,
        FakeEncoder(),
        batch_size=2,
    )
    index, metadata = load_faiss_index(index_path, mapping_path)

    assert summary.event_count == 4
    assert summary.dimension == 4
    assert index.ntotal == 4
    assert metadata.event_ids == ["E1", "E2", "E3", "E4"]


def test_metadata_reranking_is_explainable() -> None:
    """Exact metadata bonuses should be visible in score and reasons."""

    query = RetrievalQuery(
        text="truck breakdown",
        event_cause="vehicle_breakdown",
        corridor="Mysore Road",
        event_type="unplanned",
        priority="High",
    )
    event = {
        "event_cause": "vehicle_breakdown",
        "corridor": "Mysore Road",
        "event_type": "unplanned",
        "priority": "High",
    }
    score, reasons = calculate_rerank_score(0.8, event, query)

    assert score == pytest.approx(0.87)
    assert "Same event cause" in reasons
    assert "Same corridor" in reasons
    assert "Same planned/unplanned type" in reasons
    assert "Similar priority" in reasons
    assert "Highly similar event description and context" in reasons


def test_retrieval_returns_ranked_matches_and_confidence(tmp_path: Path) -> None:
    """A matching cause/corridor/type/priority event should rank first."""

    database = tmp_path / "events.db"
    index_path = tmp_path / "events.faiss"
    mapping_path = tmp_path / "mapping.json"
    encoder = FakeEncoder()
    _create_test_database(database)
    build_faiss_index(database, index_path, mapping_path, encoder)

    retriever = HistoricalEventRetriever(
        database,
        index_path,
        mapping_path,
        encoder,
    )
    result = retriever.search(
        RetrievalQuery(
            text="heavy truck breakdown on Mysore corridor",
            event_cause="vehicle_breakdown",
            corridor="Mysore Road",
            event_type="unplanned",
            priority="High",
        ),
        top_k=3,
        candidate_k=4,
    )

    assert len(result.matches) == 3
    assert result.matches[0].event_id == "E1"
    assert result.matches[0].rank == 1
    assert result.matches[0].rerank_score >= result.matches[1].rerank_score
    assert "Same event cause" in result.matches[0].reasons
    assert 0 < result.confidence <= 1
