"""Research dispatcher.

PHASE 2: returns STUB findings so the graph runs end-to-end with no real tools.
Real fan-out (web/rag/vision via asyncio.gather) lands in Phase 3 — the node
signature and the Finding contract do NOT change when tools are swapped in.

To make the re-plan loop demonstrable on a text-only run (option 1), the FIRST
finding is deliberately left ungrounded (empty source_ref) on the first pass, so
the critic flags it; on re-plan it gains a source and resolves.
"""
from __future__ import annotations

from dataclasses import asdict

from core.obs import TraceEvent
from core.schemas import Finding, ResearchState


def make_researcher_node():
    def researcher_node(state: ResearchState) -> dict:
        subtasks = state["subtasks"]
        replan_count = state.get("replan_count", 0)
        ev = TraceEvent("researcher", "running", f"gathering evidence for {len(subtasks)} subtasks")
        ev.emit()

        findings: list[Finding] = []
        for i, st in enumerate(subtasks):
            claim = f"[stub] Key finding addressing: {st.description}"
            if i == 0 and replan_count == 0:
                source_ref = ""  # ungrounded on first pass -> critic flags -> re-plan
            else:
                source_ref = f"[stub source] Evidence snippet that supports: {st.description}"
            findings.append(Finding(claim=claim, source_type="stub",
                                    source_ref=source_ref, subtask_id=st.id))

        done = TraceEvent("researcher", "complete", f"{len(findings)} findings")
        done.emit()
        return {"findings": findings, "trace": [asdict(ev), asdict(done)]}

    return researcher_node
