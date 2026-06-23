"""Minimal OpenAI adapter for structured event extraction and grounded reports."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel

from src.schemas import StructuredEvent


class OpenAIServiceError(RuntimeError):
    """Raised when an OpenAI request fails or returns no usable content."""


class CopilotLLM(Protocol):
    """Dependency contract used by graph nodes and lightweight tests."""

    def understand_event(
        self,
        user_input: str,
        form_data: dict[str, Any],
    ) -> StructuredEvent:
        """Extract a structured traffic event."""

    def generate_report(self, grounded_state: dict[str, Any]) -> str:
        """Generate a Traffic Management Plan from supplied state only."""


class OpenAICopilotService:
    """OpenAI Responses API implementation using native Pydantic parsing."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise OpenAIServiceError("OPENAI_API_KEY is required.")
        if not model:
            raise OpenAIServiceError("OPENAI_MODEL is required.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise OpenAIServiceError(
                "The openai package is not installed."
            ) from exc
        self.client = OpenAI(api_key=api_key, timeout=45.0, max_retries=1)
        self.model = model

    def understand_event(
        self,
        user_input: str,
        form_data: dict[str, Any],
    ) -> StructuredEvent:
        """Extract event facts with OpenAI Structured Outputs."""

        prompt = json.dumps(
            {"description": user_input, "form_fields": form_data},
            default=str,
            ensure_ascii=False,
        )
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Extract a Bengaluru traffic event. Use only supplied "
                            "facts. Infer planned/unplanned and a concise normalized "
                            "cause when reasonable. Return null for unknown values. "
                            "Do not invent locations, times, priority, or closure."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                text_format=StructuredEvent,
            )
        except Exception as exc:
            raise OpenAIServiceError(f"Event extraction failed: {exc}") from exc

        event = response.output_parsed
        if event is None:
            raise OpenAIServiceError(
                "OpenAI returned no structured event. The input may have been "
                "refused or incomplete."
            )
        return event

    def generate_report(self, grounded_state: dict[str, Any]) -> str:
        """Generate a concise report while prohibiting unsupported claims."""

        payload = json.dumps(grounded_state, default=str, ensure_ascii=False)
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=(
                    "Write a concise Traffic Management Plan grounded only in the "
                    "provided JSON. Use headings: Event Summary, Historical "
                    "Evidence, Operational Disruption Risk, Resource "
                    "Recommendation, Monitoring Checklist, Assumptions and "
                    "Limitations. Do not invent traffic speeds, congestion levels, "
                    "routes, crowd counts, resources, or facts. Clearly preserve "
                    "the policy-based recommendation disclaimer."
                ),
                input=payload,
                max_output_tokens=900,
            )
        except Exception as exc:
            raise OpenAIServiceError(f"Report generation failed: {exc}") from exc

        report = response.output_text.strip()
        if not report:
            raise OpenAIServiceError("OpenAI returned an empty report.")
        return report


def model_to_jsonable(value: BaseModel | Any) -> Any:
    """Convert Pydantic values to JSON-compatible data for prompts/storage."""

    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value
