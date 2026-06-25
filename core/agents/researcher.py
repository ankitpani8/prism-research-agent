"""Research dispatcher (async) — real fan-out over subtasks via asyncio.gather.

Each subtask is routed to its tool (web/rag), top snippets retrieved, and the
`light` model synthesises ONE claim grounded in those snippets (source_ref =
the retrieved text). vision subtasks are deferred to Phase 4. If a tool returns
nothing, the finding is left ungrounded (empty source_ref) so the critic flags
it and the planner re-plans — graceful degradation, visible.

Tools and model are injected (build_graph supplies real ones; tests pass fakes).
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict

from langchain_core.messages import HumanMessage, SystemMessage

from core.obs import TraceEvent, record_usage
from core.schemas import Finding, ResearchState, SubTask

_SYS = (
    "You are a research analyst. From the SOURCE SNIPPETS, state the single most "
    "relevant finding for the SUBTASK as ONE precise sentence grounded ONLY in the "
    "snippets. If the snippets do not address the subtask, reply exactly INSUFFICIENT."
)


async def _research_one(model, model_name, tools, subtask: SubTask) -> Finding:
    tool = tools.get(subtask.tool)
    if subtask.tool == "vision":
        return Finding(claim=f"[vision pending Phase 4] {subtask.description}",
                       source_type="vision", source_ref="", subtask_id=subtask.id)
    if tool is None:
        return Finding(claim=f"[no tool] {subtask.description}",
                       source_type="none", source_ref="", subtask_id=subtask.id)

    snippets = await tool(subtask.description)
    if not snippets:
        return Finding(claim=f"[unresolved] {subtask.description}",
                       source_type=subtask.tool, source_ref="", subtask_id=subtask.id)

    joined = "\n---\n".join(s.get("text") or s.get("content", "") for s in snippets[:4])
    sources = ", ".join(s.get("source") or s.get("url", "") for s in snippets[:4] if (s.get("source") or s.get("url")))
    resp = await model.ainvoke([
        SystemMessage(content=_SYS),
        HumanMessage(content=f"SUBTASK: {subtask.description}\n\nSOURCE SNIPPETS:\n{joined}"),
    ])
    record_usage(model_name, resp)
    claim = str(getattr(resp, "content", "")).strip()
    cited = f"Sources: {sources}\n{joined}" if sources else joined
    grounded_ref = "" if claim.upper().startswith("INSUFFICIENT") else cited
    return Finding(claim=claim or f"[empty] {subtask.description}",
                   source_type=subtask.tool, source_ref=grounded_ref,
                   subtask_id=subtask.id)


def make_researcher_node(model, tools: dict, model_name: str = "light"):
    async def researcher_node(state: ResearchState) -> dict:
        subtasks = state["subtasks"]
        ev = TraceEvent("researcher", "running",
                        f"{len(subtasks)} subtasks fanning out (async)")
        ev.emit()
        findings = await asyncio.gather(
            *(_research_one(model, model_name, tools, st) for st in subtasks)
        )
        done = TraceEvent("researcher", "complete", f"{len(findings)} findings")
        done.emit()
        return {"findings": list(findings), "trace": [asdict(ev), asdict(done)]}

    return researcher_node
