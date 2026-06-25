"""Robust async structured-output helper + defensive JSON parsing.

Native structured output for big hosted models, with a manual JSON fallback
(M6 deterministic-parse lesson). include_raw=True lets us capture token usage
even on structured calls and feed it to obs.record_usage.
"""
from __future__ import annotations

import json
from typing import TypeVar

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from core.obs import record_usage

T = TypeVar("T", bound=BaseModel)


def parse_json(raw: str) -> dict | None:
    raw = (raw or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    for open_c, close_c in (("{", "}"), ("[", "]")):
        if open_c in raw and close_c in raw:
            raw = raw[raw.index(open_c): raw.rindex(close_c) + 1]
            break
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def astructured(model, schema: type[T], messages: list, model_name: str = "?") -> T:
    """Try native structured output (capturing usage); fall back to JSON parse."""
    try:
        runnable = model.with_structured_output(schema, include_raw=True)
        res = await runnable.ainvoke(messages)
        if isinstance(res, dict):  # include_raw shape (real models, matching fakes)
            if res.get("raw") is not None:
                record_usage(model_name, res["raw"])
            if res.get("parsed") is not None:
                return res["parsed"]
            raise ValueError("structured parsing returned no object")
        return res  # a simple fake returned the parsed object directly
    except Exception:
        raw = await model.ainvoke(
            messages + [HumanMessage(content="Return ONLY valid JSON for the schema. No prose.")]
        )
        record_usage(model_name, raw)
        data = parse_json(getattr(raw, "content", "")) or {}
        return schema.model_validate(data)
