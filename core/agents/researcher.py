"""Research dispatcher (async) — real fan-out over subtasks via asyncio.gather.

Each subtask routes to its tool; top snippets retrieved; the `light` model
synthesises ONE claim grounded ONLY in those snippets. Per-subtask trace events
show the tool and snippet count, so an empty tool result is VISIBLE (not a silent
ungrounded finding). Empty result -> ungrounded finding -> critic flags -> replan.
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
    if subtask.tool == "vision":
        TraceEvent("researcher", "running", f"{subtask.tool}: deferred to Phase 4").emit()
        return Finding(claim=f"[vision pending Phase 4] {subtask.description}",
                       source_type="vision", source_ref="", subtask_id=subtask.id)
    tool = tools.get(subtask.tool)
    if tool is None:
        return Finding(claim=f"[no tool:{subtask.tool}] {subtask.description}",
                       source_type="none", source_ref="", subtask_id=subtask.id)

    snippets = await tool(subtask.description)
    TraceEvent("researcher", "running",
               f"{subtask.tool}: {len(snippets)} snippets for '{subtask.description[:48]}'").emit()
    if not snippets:
        return Finding(claim=f"[unresolved] {subtask.description}",
                       source_type=subtask.tool, source_ref="", subtask_id=subtask.id)

    joined = "\n---\n".join(s.get("text") or s.get("content", "") for s in snippets[:4])
    sources = ", ".join(s.get("source") or s.get("url", "")
                        for s in snippets[:4] if (s.get("source") or s.get("url")))
    resp = await model.ainvoke([
        SystemMessage(content=_SYS),
        HumanMessage(content=f"SUBTASK: {subtask.description}\n\nSOURCE SNIPPETS:\n{joined}"),
    ])
    record_usage(model_name, resp)
    claim = str(getattr(resp, "content", "")).strip()
    cited = f"Sources: {sources}\n{joined}" if sources else joined
    grounded_ref = "" if "INSUFFICIENT" in claim.upper()[:40] else cited
    return Finding(claim=claim or f"[empty] {subtask.description}",
                   source_type=subtask.tool, source_ref=grounded_ref, subtask_id=subtask.id)


def make_researcher_node(model, tools: dict, model_name: str = "light"):
    async def researcher_node(state: ResearchState) -> dict:
        subtasks = state["subtasks"]
        ev = TraceEvent("researcher", "running", f"{len(subtasks)} subtasks fanning out (async)")
        ev.emit()
        findings = await asyncio.gather(
            *(_research_one(model, model_name, tools, st) for st in subtasks))
        grounded_n = sum(1 for f in findings if f.source_ref)
        done = TraceEvent("researcher", "complete",
                          f"{len(findings)} findings ({grounded_n} with sources)")
        done.emit()
        return {"findings": list(findings), "trace": [asdict(ev), asdict(done)]}

    return researcher_node
