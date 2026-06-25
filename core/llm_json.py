"""Robust structured-output helper.

Big hosted models (Gemini) do structured output reliably, so we use it. But it
can fail (schema quirks, transient errors), so we fall back to a manual JSON
prompt + Pydantic validation — the M6 "deterministic parse" lesson. The small
local critic model uses _parse_json directly because it is less reliable with
the structured-output API.
"""
from __future__ import annotations

import json
from typing import TypeVar

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def parse_json(raw: str) -> dict | None:
    """LLMs wrap JSON in code fences sometimes. Strip and parse defensively."""
    raw = (raw or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # Grab the outermost JSON object/array if there's surrounding prose.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        if open_c in raw and close_c in raw:
            raw = raw[raw.index(open_c): raw.rindex(close_c) + 1]
            break
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def structured(model, schema: type[T], messages: list) -> T:
    """Try native structured output; fall back to JSON-prompt + validation."""
    try:
        return model.with_structured_output(schema).invoke(messages)
    except Exception:
        raw = model.invoke(
            messages + [HumanMessage(content="Return ONLY valid JSON for the schema. No prose.")]
        )
        data = parse_json(getattr(raw, "content", "")) or {}
        return schema.model_validate(data)
