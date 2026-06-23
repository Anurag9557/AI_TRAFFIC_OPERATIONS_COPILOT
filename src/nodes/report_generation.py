"""Report Generation Agent using a grounded OpenAI prompt."""

from __future__ import annotations

from typing import Any

from src.services.openai_service import CopilotLLM
from src.state import TrafficOpsState


def create_report_generation_node(llm: CopilotLLM):
    """Create a report node with an injected OpenAI-compatible service."""

    def report_generation(state: TrafficOpsState) -> dict[str, Any]:
        grounded_state = {
            "event": state["event"].model_dump(mode="json"),
            "historical_evidence": state["historical_stats"].model_dump(mode="json"),
            "similar_events": state.get("similar_events", [])[:5],
            "retrieval_confidence": state.get("retrieval_confidence"),
            "risk_assessment": state["risk_assessment"].model_dump(mode="json"),
            "resource_recommendation": state["resource_plan"].model_dump(mode="json"),
        }
        return {"report": llm.generate_report(grounded_state)}

    return report_generation
