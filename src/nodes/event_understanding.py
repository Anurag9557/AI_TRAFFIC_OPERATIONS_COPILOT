"""Event Understanding Agent using OpenAI structured output."""

from __future__ import annotations

from typing import Any

from src.data.preprocessing import (
    normalize_corridor,
    normalize_event_cause,
    normalize_priority,
    normalize_slug,
)
from src.schemas import StructuredEvent
from src.services.openai_service import CopilotLLM
from src.state import TrafficOpsState


def _present(value: Any) -> bool:
    """Return whether a form value should override extracted content."""

    return value is not None and value != "" and value != "Auto"


def normalize_event(event: StructuredEvent) -> StructuredEvent:
    """Apply the same categorical normalization used by historical ingestion."""

    values = event.model_dump(mode="python")
    if values.get("event_cause"):
        values["event_cause"] = normalize_event_cause(values["event_cause"])
    values["corridor"] = normalize_corridor(values.get("corridor"))
    values["priority"] = normalize_priority(values.get("priority"))
    values["event_type"] = normalize_slug(values.get("event_type"))
    return StructuredEvent.model_validate(values)


def validate_event(event: StructuredEvent) -> list[str]:
    """Return concise validation messages for required operational facts."""

    errors: list[str] = []
    if not event.description:
        errors.append("Event description is required.")
    if event.event_type is None:
        errors.append("Planned or unplanned event type is required.")
    if not event.event_cause:
        errors.append("Event cause is required.")
    if not event.corridor:
        errors.append("Corridor is required.")
    if event.priority is None:
        errors.append("Priority is required.")
    if event.event_type == "planned" and event.start_datetime is None:
        errors.append("Start date/time is required for a planned event.")
    if (
        event.start_datetime is not None
        and event.end_datetime is not None
        and event.end_datetime < event.start_datetime
    ):
        errors.append("End date/time cannot be earlier than start date/time.")
    return errors


def create_event_understanding_node(llm: CopilotLLM):
    """Create the graph node with an injected OpenAI-compatible service."""

    def event_understanding(state: TrafficOpsState) -> dict[str, Any]:
        user_input = state.get("user_input", "").strip()
        form_data = state.get("form_data", {})
        extracted = llm.understand_event(user_input, form_data)
        values = extracted.model_dump(mode="python")

        field_aliases = {"vehicle_type": ("vehicle_type", "veh_type")}
        for field in StructuredEvent.model_fields:
            keys = field_aliases.get(field, (field,))
            for key in keys:
                value = form_data.get(key)
                if _present(value):
                    values[field] = value
                    break

        if not values.get("description") and user_input:
            values["description"] = user_input
        event = normalize_event(StructuredEvent.model_validate(values))
        errors = validate_event(event)
        return {
            "event": event,
            "validation_errors": errors,
            "error": "; ".join(errors) if errors else None,
        }

    return event_understanding
