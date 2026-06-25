"""Observability — per-request token/latency/cost telemetry + UI trace events.

Port of learning_AgenticAI/module5_production/03_telemetry.py, extended for Prism:
  - cost accounting (PRICING; Ollama = $0 — the local-critic cost lever).
  - TraceEvent: the per-agent event the Streamlit panel will render (Phase 6).
  - A contextvar so async agent nodes roll their LLM usage up into ONE request
    telemetry line, even across asyncio.gather fan-out. (Fixes the Phase-2 gap
    where node calls logged zero tokens.)
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field

PRICING: dict[str, tuple[float, float]] = {
    # model substring : (USD per 1M input, USD per 1M output). Approximate —
    # verify against current provider pricing. Architecture point (local=free,
    # cost attributed per request) does not depend on exact cents.
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
    agent: str
    status: str            # running | complete | flagged | error
    detail: str = ""
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
    by_model: dict = field(default_factory=dict)   # per-provider breakdown
    error: str | None = None

    def add_llm_usage(self, model: str, input_t: int, output_t: int) -> None:
        self.llm_calls += 1
        self.input_tokens += input_t
        self.output_tokens += output_t
        cost = estimate_cost(model, input_t, output_t)
        self.cost_usd = round(self.cost_usd + cost, 8)
        m = self.by_model.setdefault(model, {"calls": 0, "in": 0, "out": 0, "usd": 0.0})
        m["calls"] += 1
        m["in"] += input_t
        m["out"] += output_t
        m["usd"] = round(m["usd"] + cost, 8)

    def finalize(self) -> None:
        self.duration_ms = round((time.time() - self.started_at) * 1000, 1)

    def emit(self) -> None:
        print("[telemetry] " + json.dumps(asdict(self)))


# Active request telemetry for the current async context. Set by telemetry();
# read by record_usage() inside agent nodes (propagates into gather tasks).
_current: ContextVar[RequestTelemetry | None] = ContextVar("prism_telemetry", default=None)


def record_usage(model_name: str, response) -> None:
    """Roll a LangChain response's token usage into the active request, if any."""
    t = _current.get()
    if t is None:
        return
    usage = getattr(response, "usage_metadata", None) or {}
    t.add_llm_usage(
        model_name,
        int(usage.get("input_tokens", 0) or 0),
        int(usage.get("output_tokens", 0) or 0),
    )


@contextmanager
def telemetry(query: str):
    t = RequestTelemetry(user_query=query[:200])
    token = _current.set(t)
    try:
        yield t
    except Exception as e:
        t.error = f"{type(e).__name__}: {str(e)[:200]}"
        raise
    finally:
        _current.reset(token)
        t.finalize()
        t.emit()
