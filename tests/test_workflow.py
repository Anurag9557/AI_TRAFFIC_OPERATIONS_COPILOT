"""Backend workflow tests with no network, model download, or API cost."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.graph import build_copilot_graph, run_copilot
from src.nodes.event_understanding import (
    create_event_understanding_node,
)
from src.nodes.report_generation import create_report_generation_node
from src.nodes.resource_planning import plan_resources
from src.schemas import RiskBand, StructuredEvent
from src.services.retrieval import (
    RetrievalMatch,
    RetrievalQuery,
    RetrievalResult,
)


class FakeLLM:
    """Deterministic OpenAI replacement for graph and report tests."""

    def __init__(self, event: StructuredEvent) -> None:
        self.event = event
        self.report_payload: dict[str, Any] | None = None

    def understand_event(
        self,
        user_input: str,
        form_data: dict[str, Any],
    ) -> StructuredEvent:
        return self.event

    def generate_report(self, grounded_state: dict[str, Any]) -> str:
        self.report_payload = grounded_state
        return (
            "# Traffic Management Plan\n\n"
            "## Event Summary\nGrounded test event.\n\n"
            "## Historical Evidence\nFive similar events reviewed.\n\n"
            "## Operational Disruption Risk\nHigh operational disruption risk.\n\n"
            "## Resource Recommendation\nPolicy-based recommendation, not learned "
            "prediction.\n\n"
            "## Monitoring Checklist\n- Monitor field conditions.\n\n"
            "## Assumptions and Limitations\nNo traffic-speed data was used."
        )


class FakeRetriever:
    """Deterministic Phase 3 replacement returning five matching events."""

    def __init__(self) -> None:
        self.calls = 0

    def search(
        self,
        query: RetrievalQuery,
        top_k: int = 5,
        candidate_k: int = 30,
    ) -> RetrievalResult:
        self.calls += 1
        matches = [
            RetrievalMatch(
                rank=index + 1,
                event_id=f"H{index + 1}",
                semantic_similarity=0.82 - index * 0.02,
                rerank_score=0.90 - index * 0.02,
                reasons=["Same event cause", "Same corridor"],
                event={
                    "event_type": "unplanned",
                    "event_cause": "tree_fall",
                    "description": "Tree fall obstructing the corridor",
                    "corridor": "Mysore Road",
                    "priority": "High",
                    "requires_road_closure": 1 if index < 3 else 0,
                    "start_datetime": "2024-03-01T10:00:00+00:00",
                    "status": "closed",
                    "address": "Mysore Road",
                    "police_station": "Test Station",
                    "vehicle_type": None,
                    "scheduled_duration_hours": None,
                    "handling_duration_hours": 12.0,
                },
            )
            for index in range(5)
        ]
        return RetrievalResult(
            matches=matches,
            confidence=0.82,
            candidate_count=30,
            model_name="fake-model",
        )


def _database(path: Path) -> None:
    """Create only the history and cases columns exercised by the graph."""

    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE events (
                event_cause TEXT NOT NULL,
                requires_road_closure INTEGER NOT NULL,
                handling_duration_hours REAL
            );
            CREATE TABLE cases (
                case_id TEXT PRIMARY KEY,
                user_input TEXT NOT NULL,
                structured_event_json TEXT NOT NULL,
                similar_event_ids_json TEXT NOT NULL DEFAULT '[]',
                historical_stats_json TEXT NOT NULL DEFAULT '{}',
                risk_assessment_json TEXT NOT NULL DEFAULT '{}',
                resource_plan_json TEXT NOT NULL DEFAULT '{}',
                report TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        connection.executemany(
            "INSERT INTO events VALUES (?, ?, ?)",
            [("tree_fall", 1 if index < 8 else 0, 12.0) for index in range(20)],
        )
        connection.commit()


def _valid_event() -> StructuredEvent:
    return StructuredEvent(
        event_type="unplanned",
        event_cause="tree_fall",
        description="A large tree has fallen across Mysore Road.",
        corridor="Mysore Road",
        priority="High",
        requires_road_closure=True,
    )


def test_complete_graph_execution_saves_case(tmp_path: Path) -> None:
    """The linear graph should create a report and persist one completed case."""

    database = tmp_path / "traffic_ops.db"
    _database(database)
    llm = FakeLLM(_valid_event())
    retriever = FakeRetriever()
    graph = build_copilot_graph(
        llm=llm,
        retriever=retriever,
        database_path=database,
    )

    result = run_copilot(
        "Tree fall blocking Mysore Road",
        {},
        graph=graph,
        case_id="CASE-1",
    )

    assert result["case_saved"] is True
    assert result["case_id"] == "CASE-1"
    assert result["risk_assessment"].score >= 55
    assert result["resource_plan"].policy_disclaimer == (
        "Policy-based recommendation, not learned prediction."
    )
    assert "# Traffic Management Plan" in result["report"]
    assert retriever.calls == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0] == 1


def test_invalid_event_returns_validation_error_without_retrieval(
    tmp_path: Path,
) -> None:
    """The only conditional path must stop immediately on invalid input."""

    database = tmp_path / "traffic_ops.db"
    _database(database)
    llm = FakeLLM(StructuredEvent(description="Vague traffic issue"))
    retriever = FakeRetriever()
    graph = build_copilot_graph(
        llm=llm,
        retriever=retriever,
        database_path=database,
    )

    result = run_copilot("Vague traffic issue", {}, graph=graph)

    assert result["validation_errors"]
    assert result["case_saved"] is False
    assert retriever.calls == 0
    assert "report" not in result


def test_form_fields_override_extracted_event() -> None:
    """Explicit form facts should override OpenAI extraction and normalize."""

    llm = FakeLLM(
        StructuredEvent(
            event_type="planned",
            event_cause="others",
            description="Original",
            corridor="Non-corridor",
            priority="Low",
            start_datetime="2026-06-22T10:00:00+05:30",
        )
    )
    node = create_event_understanding_node(llm)
    result = node(
        {
            "user_input": "Tree fall reported",
            "form_data": {
                "event_type": "unplanned",
                "event_cause": "Tree Fall",
                "corridor": "Mysore Road",
                "priority": "High",
            },
        }
    )

    assert result["validation_errors"] == []
    assert result["event"].event_cause == "tree_fall"
    assert result["event"].event_type == "unplanned"
    assert result["event"].priority == "High"


def test_report_generation_passes_only_grounded_state() -> None:
    """Report generation should pass approved state fields to the LLM mock."""

    llm = FakeLLM(_valid_event())
    node = create_report_generation_node(llm)
    state = {
        "event": _valid_event(),
        "historical_stats": {
            "cohort_size": 5,
        },
    }

    # Use real schemas by taking them from a short graph-compatible fixture.
    from src.schemas import HistoricalStatistics, ResourcePlan, RiskAssessment

    state["historical_stats"] = HistoricalStatistics(cohort_size=5)
    state["risk_assessment"] = RiskAssessment(
        score=60,
        band="High",
        confidence=0.8,
    )
    state["resource_plan"] = ResourcePlan(
        manpower_min=6,
        manpower_max=10,
        barricades_min=8,
        barricades_max=14,
        policy_disclaimer="Policy-based recommendation, not learned prediction.",
    )
    state["similar_events"] = []
    state["retrieval_confidence"] = 0.8

    result = node(state)  # type: ignore[arg-type]

    assert "Traffic Management Plan" in result["report"]
    assert llm.report_payload is not None
    assert set(llm.report_payload) == {
        "event",
        "historical_evidence",
        "similar_events",
        "retrieval_confidence",
        "risk_assessment",
        "resource_recommendation",
    }


def test_resource_planning_applies_closure_and_cause_rules() -> None:
    """Critical public events should receive visible deterministic modifiers."""

    plan = plan_resources(
        RiskBand.CRITICAL,
        "public_event",
        closure_tendency=0.50,
        requires_road_closure=True,
    )

    assert plan.manpower_min == 14
    assert plan.manpower_max == 22
    assert plan.barricades_min == 18
    assert plan.barricades_max == 28
    assert "Patrol vehicle" in plan.support_vehicles
    assert plan.policy_disclaimer == (
        "Policy-based recommendation, not learned prediction."
    )

