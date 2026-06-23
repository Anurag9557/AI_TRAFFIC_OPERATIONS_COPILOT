"""LangGraph workflow for the backend Traffic Operations Copilot."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.config import Settings, get_settings
from src.nodes.event_understanding import create_event_understanding_node
from src.nodes.historical_retrieval import (
    Retriever,
    create_historical_retrieval_node,
)
from src.nodes.impact_assessment import create_impact_assessment_node
from src.nodes.report_generation import create_report_generation_node
from src.nodes.resource_planning import resource_planning
from src.services.database import save_case
from src.services.embeddings import SentenceTransformerEncoder
from src.services.openai_service import CopilotLLM, OpenAICopilotService
from src.services.retrieval import HistoricalEventRetriever
from src.state import TrafficOpsState


def _route_after_understanding(state: TrafficOpsState) -> str:
    """End early when structured event validation fails."""

    return "invalid" if state.get("validation_errors") else "valid"


def _create_save_case_node(database_path: Path):
    """Create the final persistence node for successful graph executions."""

    def save_completed_case(state: TrafficOpsState) -> dict[str, Any]:
        case_id = state.get("case_id") or str(uuid.uuid4())
        save_case(
            database_path,
            case_id=case_id,
            user_input=state.get("user_input", ""),
            structured_event_json=json.dumps(
                state["event"].model_dump(mode="json"),
                ensure_ascii=False,
            ),
            similar_event_ids_json=json.dumps(
                state.get("evidence_event_ids", []),
                ensure_ascii=False,
            ),
            historical_stats_json=state["historical_stats"].model_dump_json(),
            risk_assessment_json=state["risk_assessment"].model_dump_json(),
            resource_plan_json=state["resource_plan"].model_dump_json(),
            report=state["report"],
        )
        return {"case_id": case_id, "case_saved": True}

    return save_completed_case


def build_copilot_graph(
    *,
    llm: CopilotLLM,
    retriever: Retriever,
    database_path: Path,
):
    """Compile the linear five-agent workflow with one validation branch."""

    builder = StateGraph(TrafficOpsState)
    builder.add_node(
        "event_understanding",
        create_event_understanding_node(llm),
    )
    builder.add_node(
        "historical_retrieval",
        create_historical_retrieval_node(retriever),
    )
    builder.add_node(
        "impact_assessment",
        create_impact_assessment_node(database_path),
    )
    builder.add_node("resource_planning", resource_planning)
    builder.add_node(
        "report_generation",
        create_report_generation_node(llm),
    )
    builder.add_node("save_case", _create_save_case_node(database_path))

    builder.add_edge(START, "event_understanding")
    builder.add_conditional_edges(
        "event_understanding",
        _route_after_understanding,
        {"invalid": END, "valid": "historical_retrieval"},
    )
    builder.add_edge("historical_retrieval", "impact_assessment")
    builder.add_edge("impact_assessment", "resource_planning")
    builder.add_edge("resource_planning", "report_generation")
    builder.add_edge("report_generation", "save_case")
    builder.add_edge("save_case", END)
    return builder.compile()


def build_default_copilot_graph(settings: Settings | None = None):
    """Construct the real OpenAI + FAISS graph from environment settings."""

    settings = settings or get_settings()
    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is required to build the default graph.")
    llm = OpenAICopilotService(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
    )
    encoder = SentenceTransformerEncoder(settings.embedding_model)
    retriever = HistoricalEventRetriever(
        settings.database_path,
        settings.faiss_index_path,
        settings.faiss_mapping_path,
        encoder,
    )
    return build_copilot_graph(
        llm=llm,
        retriever=retriever,
        database_path=settings.database_path,
    )


def run_copilot(
    user_input: str,
    form_data: dict[str, Any],
    *,
    graph: Any | None = None,
    case_id: str | None = None,
) -> TrafficOpsState:
    """Invoke a compiled graph and return its final state."""

    compiled = graph or build_default_copilot_graph()
    initial_state: TrafficOpsState = {
        "case_id": case_id or str(uuid.uuid4()),
        "user_input": user_input,
        "form_data": form_data,
        "validation_errors": [],
        "case_saved": False,
        "error": None,
    }
    return compiled.invoke(initial_state)
