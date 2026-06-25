"""Planner / Orchestrator — decomposes the question into typed subtasks and
drives the re-plan loop. Evaluator-optimizer pattern from M3/02, rewired so the
planner (not a fixed sequence) owns the loop. Uses the `light` role.

Factory pattern: make_planner_node(model) -> node. Lets the graph inject real or
fake models (the latter makes the wiring testable with no API keys).
"""
from __future__ import annotations

from dataclasses import asdict

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_json import structured
from core.obs import TraceEvent
from core.schemas import Plan, ResearchState

_SYS = (
    "You are a research planner. Decompose the user's question into 2-3 focused, "
    "non-overlapping subtasks. For each, pick the single best tool: 'web' (external "
    "facts/benchmarks), 'rag' (internal documents), 'vision' (read an uploaded "
    "chart/dashboard image), or 'none'. Be concise."
)


def make_planner_node(model):
    def planner_node(state: ResearchState) -> dict:
        replanning = state.get("critic_report") is not None
        replan_count = state.get("replan_count", 0) + (1 if replanning else 0)

        ev = TraceEvent("planner", "running",
                        "re-planning around flagged claims" if replanning else "decomposing question")
        ev.emit()

        user = f"Question: {state['question']}"
        if state.get("image_path"):
            user += "\n(Note: an image/dashboard is attached — consider a 'vision' subtask.)"
        if replanning:
            flagged = state["critic_report"].unsupported_claims
            user += (
                "\n\nThe previous attempt produced claims the critic could NOT ground:\n- "
                + "\n- ".join(flagged)
                + "\n\nReplan to find solid evidence for these, or drop them."
            )

        plan = structured(model, Plan, [SystemMessage(content=_SYS), HumanMessage(content=user)])
        subtasks = plan.subtasks
        done = TraceEvent("planner", "complete", f"{len(subtasks)} subtasks")
        done.emit()
        return {
            "subtasks": subtasks,
            "replan_count": replan_count,
            "trace": [asdict(ev), asdict(done)],
        }

    return planner_node
