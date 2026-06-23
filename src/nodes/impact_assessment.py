"""Impact Assessment Agent reusing the Phase 4 deterministic scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.risk_scoring import assess_operational_disruption
from src.state import TrafficOpsState


def create_impact_assessment_node(database_path: Path):
    """Create a node bound to the existing SQLite event history."""

    def impact_assessment(state: TrafficOpsState) -> dict[str, Any]:
        assessment = assess_operational_disruption(
            state["event"],
            state["retrieval_result"],
            database_path,
        )
        return {"risk_assessment": assessment}

    return impact_assessment
