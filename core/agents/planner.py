"""Planner / Orchestrator (async) — decompose into EVIDENCE-GATHERING subtasks.

Each subtask must be answerable by a tool: 'rag' (internal CX corpus), 'web'
(external benchmarks), or 'vision' (only if an image is attached). Analysis and
synthesis are NOT subtasks — that's the summariser's job. Defensive code drops
any untooled subtask and guarantees at least two evidence subtasks so the graph
always has something real to ground. Evaluator-optimizer loop from M3/02.
"""
from __future__ import annotations

from dataclasses import asdict

from langchain_core.messages import HumanMessage, SystemMessage

from core.guardrails import screen_injection
from core.llm_json import astructured
from core.obs import TraceEvent
from core.schemas import Plan, ResearchState, SubTask

_SYS = (
    "You are a research planner. Output 2-3 EVIDENCE-GATHERING subtasks that each "
    "retrieve information with a tool. Rules:\n"
    "- tool MUST be 'rag' (internal CX documents/corpus), 'web' (external "
    "benchmarks/best-practices), or 'vision' (ONLY if an image is attached).\n"
    "- Use 'rag' for company/internal specifics (complaint categories, verbatims) "
    "and 'web' for external benchmarks.\n"
    "- NEVER use 'none'. NEVER create analysis/summarise/identify-drivers subtasks "
    "— synthesis happens later. Each subtask is a retrieval, phrased as a search.\n"
    "Example for complaint drivers: rag='internal prepaid complaint category "
    "definitions and frequencies'; rag='representative customer verbatims by "
    "category'; web='telecom prepaid complaint benchmarks and churn drivers'."
)

_VALID = {"rag", "web", "vision"}


def _fallback_subtasks(question: str, has_image: bool) -> list[SubTask]:
    subs = [
        SubTask(id="rag1", description=f"internal CX documents relevant to: {question}", tool="rag"),
        SubTask(id="web1", description=f"external telecom CX benchmarks for: {question}", tool="web"),
    ]
    if has_image:
        subs.insert(0, SubTask(id="vis1", description="read the attached dashboard image", tool="vision"))
    return subs


def make_planner_node(model, model_name: str = "light"):
    async def planner_node(state: ResearchState) -> dict:
        replanning = state.get("critic_report") is not None
        replan_count = state.get("replan_count", 0) + (1 if replanning else 0)
        has_image = bool(state.get("image_path"))
        # Input guardrail (OWASP-LLM01): screen the user question before it
        # enters any prompt. Detected -> flag on the trace; we neutralise by
        # treating the question as data, not instructions.
        hits = screen_injection(state["question"])
        if hits:
            TraceEvent("guardrail", "flagged",
                       f"prompt-injection screen matched {len(hits)} pattern(s)").emit()

        ev = TraceEvent("planner", "running",
                        "re-planning around flagged claims" if replanning else "decomposing question")
        ev.emit()

        user = f"Question: {state['question']}"
        if has_image:
            user += "\n(An image/dashboard IS attached — include exactly one 'vision' subtask.)"
        if replanning:
            flagged = state["critic_report"].unsupported_claims
            user += ("\n\nThe previous attempt produced claims the critic could NOT ground:\n- "
                     + "\n- ".join(flagged)
                     + "\n\nReplan with retrieval subtasks to find solid evidence, or drop them.")

        plan = await astructured(model, Plan,
                                 [SystemMessage(content=_SYS), HumanMessage(content=user)],
                                 model_name=model_name)
        # Defensive: keep only tool-actionable subtasks; vision only if image present.
        kept = [s for s in plan.subtasks
                if s.tool in _VALID and not (s.tool == "vision" and not has_image)]
        if len(kept) < 2:
            kept = _fallback_subtasks(state["question"], has_image)

        done = TraceEvent("planner", "complete",
                          f"{len(kept)} subtasks [{', '.join(s.tool for s in kept)}]")
        done.emit()
        return {"subtasks": kept, "replan_count": replan_count,
                "trace": [asdict(ev), asdict(done)]}

    return planner_node
