"""Planner / Orchestrator (async) — decompose into typed subtasks; drive re-plan.
Uses the `light` role. Evaluator-optimizer loop from M3/02.
"""
from __future__ import annotations

from dataclasses import asdict

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_json import astructured
from core.obs import TraceEvent
from core.schemas import Plan, ResearchState

_SYS = (
    "You are a research planner. Decompose the question into 2-3 focused, "
    "non-overlapping subtasks. For each, choose ONE tool: 'web' (external "
    "facts/benchmarks), 'rag' (internal CX documents), 'vision' (read an uploaded "
    "dashboard image), or 'none'. Prefer 'rag' for internal/company specifics and "
    "'web' for external benchmarks. Be concise."
)


def make_planner_node(model, model_name: str = "light"):
    async def planner_node(state: ResearchState) -> dict:
        replanning = state.get("critic_report") is not None
        replan_count = state.get("replan_count", 0) + (1 if replanning else 0)
        ev = TraceEvent("planner", "running",
                        "re-planning around flagged claims" if replanning else "decomposing question")
        ev.emit()

        user = f"Question: {state['question']}"
        if state.get("image_path"):
            user += "\n(An image/dashboard is attached — include a 'vision' subtask.)"
        if replanning:
            flagged = state["critic_report"].unsupported_claims
            user += ("\n\nThe previous attempt produced claims the critic could NOT ground:\n- "
                     + "\n- ".join(flagged)
                     + "\n\nReplan to find solid evidence for these, or drop them.")

        plan = await astructured(model, Plan,
                                 [SystemMessage(content=_SYS), HumanMessage(content=user)],
                                 model_name=model_name)
        done = TraceEvent("planner", "complete", f"{len(plan.subtasks)} subtasks")
        done.emit()
        return {"subtasks": plan.subtasks, "replan_count": replan_count,
                "trace": [asdict(ev), asdict(done)]}

    return planner_node
