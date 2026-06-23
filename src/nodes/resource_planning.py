"""Deterministic policy-based resource recommendations."""

from __future__ import annotations

from typing import Any

from src.schemas import ResourcePlan, RiskBand
from src.state import TrafficOpsState

POLICY_DISCLAIMER = "Policy-based recommendation, not learned prediction."

BASE_POLICY = {
    RiskBand.LOW: (2, 4, 0, 4),
    RiskBand.MODERATE: (4, 6, 4, 8),
    RiskBand.HIGH: (6, 10, 8, 14),
    RiskBand.CRITICAL: (10, 16, 14, 24),
}

SUPPORT_VEHICLES = {
    "vehicle_breakdown": ["Tow/recovery vehicle"],
    "accident": ["Ambulance/rescue coordination", "Patrol vehicle"],
    "tree_fall": ["Debris/utility clearance vehicle"],
    "debris": ["Debris clearance vehicle"],
    "water_logging": ["Drainage/utility support vehicle"],
    "pot_holes": ["Road maintenance vehicle"],
    "road_conditions": ["Road maintenance vehicle"],
    "construction": ["Work-zone support vehicle"],
    "public_event": ["Patrol vehicle"],
    "procession": ["Patrol vehicle"],
    "protest": ["Patrol vehicle"],
    "vip_movement": ["Patrol vehicle"],
}


def plan_resources(
    risk_band: RiskBand,
    event_cause: str | None,
    closure_tendency: float | None,
    requires_road_closure: bool | None = None,
) -> ResourcePlan:
    """Apply visible resource ranges and simple cause/closure modifiers."""

    manpower_min, manpower_max, barricades_min, barricades_max = BASE_POLICY[
        risk_band
    ]
    rationale = [f"{risk_band.value} risk uses the {risk_band.value.lower()} base policy."]
    closure_trigger = requires_road_closure is True or (
        closure_tendency is not None and closure_tendency >= 0.40
    )
    if closure_trigger:
        manpower_min += 2
        manpower_max += 2
        barricades_min += 4
        barricades_max += 4
        rationale.append(
            "Road-closure requirement/tendency adds traffic-control posts and "
            "barricades."
        )

    cause = event_cause or "unknown"
    support = list(SUPPORT_VEHICLES.get(cause, []))
    if risk_band in {RiskBand.HIGH, RiskBand.CRITICAL} and "Patrol vehicle" not in support:
        support.append("Patrol vehicle")
    if not support:
        support.append("No dedicated support vehicle; keep patrol support available")

    if cause in {"public_event", "procession", "protest", "vip_movement"}:
        manpower_min += 2
        manpower_max += 4
        rationale.append(
            "Public-gathering control adds entry/exit and perimeter staffing."
        )
    rationale.append(f"Support vehicles follow the event-cause rule for {cause}.")

    return ResourcePlan(
        manpower_min=manpower_min,
        manpower_max=manpower_max,
        barricades_min=barricades_min,
        barricades_max=barricades_max,
        support_vehicles=support,
        deployment_notes=[
            "Final deployment remains subject to traffic-police judgement and "
            "current field conditions."
        ],
        rationale=rationale,
        policy_disclaimer=POLICY_DISCLAIMER,
    )


def resource_planning(state: TrafficOpsState) -> dict[str, Any]:
    """LangGraph node for deterministic resource planning."""

    event = state["event"]
    stats = state["historical_stats"]
    plan = plan_resources(
        state["risk_assessment"].band,
        event.event_cause,
        stats.closure_rate,
        event.requires_road_closure,
    )
    return {"resource_plan": plan}
