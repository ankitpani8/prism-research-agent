"""Shared async pipeline runner — ONE code path for the CLI, Streamlit, and FastAPI.

Streams per-agent trace events as they happen (via LangGraph astream updates) and
calls on_event for each, so any front end can render a live trace panel. Returns
the final brief, the per-request telemetry, and the re-plan count.

`app` is injectable so tests can drive a fake-model graph with no API keys.
"""
from __future__ import annotations

from collections.abc import Callable

from core.graph import build_graph, initial_state
from core.obs import telemetry


async def run_research(
    question: str,
    image_path: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    thread_id: str = "prism-demo",
    request_id: str | None = None,
    app=None,
) -> dict:
    app = app or build_graph()
    cfg = {"configurable": {"thread_id": thread_id}}
    final = None
    replan_count = 0

    with telemetry(question, request_id=request_id) as tel:
        async for chunk in app.astream(
            initial_state(question, image_path), config=cfg, stream_mode="updates"
        ):
            for _node, delta in chunk.items():
                if not isinstance(delta, dict):
                    continue
                for ev in (delta.get("trace") or []):
                    if on_event:
                        on_event(ev)
                if delta.get("replan_count") is not None:
                    replan_count = delta["replan_count"]
                if delta.get("final") is not None:
                    final = delta["final"]

    return {"final": final, "telemetry": tel, "replan_count": replan_count}
