"""Observability — per-request token/latency/cost telemetry + a UI trace event.

Port of learning_AgenticAI/module5_production/03_telemetry.py, plus two small
additions Prism needs downstream:
  - cost accounting (PRICING table; Ollama = $0 — the local-critic cost lever).
  - TraceEvent: the per-agent event the Phase-6 Streamlit panel will render.
    ONE source of truth — the same events feed logs and the live UI.

Everything emits as JSON lines to stdout, pipeable to any log aggregator later.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

# Approximate USD per 1M tokens. VERIFY against current provider pricing before
# you quote these as fact — they drift. The architecture point (local critic is
# free, cost attributed per request) does not depend on the exact cents.
PRICING: dict[str, tuple[float, float]] = {
    # model substring : (input_per_1M, output_per_1M)
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "claude-haiku": (1.00, 5.00),
    "claude-sonnet": (3.00, 15.00),
    "qwen2.5": (0.0, 0.0),  # local Ollama — the cost lever
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    for key, (cin, cout) in PRICING.items():
        if key in (model or ""):
            return (input_tokens / 1_000_000) * cin + (output_tokens / 1_000_000) * cout
    return 0.0


@dataclass
class TraceEvent:
    """A single agent state change. Streamed to the UI trace panel (Phase 6)."""
    agent: str                      # "planner" | "researcher" | "critic" | ...
    status: str                     # "running" | "complete" | "flagged" | "error"
    detail: str = ""                # human-readable, e.g. "validating 7 claims, 2 flagged"
    ts: float = field(default_factory=time.time)

    def emit(self) -> None:
        print("[trace] " + json.dumps(asdict(self)))


@dataclass
class RequestTelemetry:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_query: str = ""
    started_at: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None

    def add_llm_usage(self, model: str, input_t: int, output_t: int) -> None:
        self.llm_calls += 1
        self.input_tokens += input_t
        self.output_tokens += output_t
        self.cost_usd += estimate_cost(model, input_t, output_t)

    def record_response(self, model: str, response) -> None:
        """Pull token usage off a LangChain response if the provider reported it."""
        usage = getattr(response, "usage_metadata", None) or {}
        self.add_llm_usage(
            model,
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
        )

    def finalize(self) -> None:
        self.duration_ms = round((time.time() - self.started_at) * 1000, 1)

    def emit(self) -> None:
        print("[telemetry] " + json.dumps(asdict(self)))


@contextmanager
def telemetry(query: str):
    """Use in a `with` block to guarantee finalize/emit runs even on error."""
    t = RequestTelemetry(user_query=query[:200])
    try:
        yield t
    except Exception as e:
        t.error = f"{type(e).__name__}: {str(e)[:200]}"
        raise
    finally:
        t.finalize()
        t.emit()
