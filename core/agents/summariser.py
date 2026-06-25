"""Summariser — produces the final Pydantic-validated ResearchBrief.

Uses ONLY findings the critic grounded (refuses to assert what couldn't be
grounded — the M6 refusal lesson). If nothing is grounded, it returns an explicit
"insufficient evidence" brief rather than hallucinating. Uses the `heavy` role.
"""
from __future__ import annotations

from dataclasses import asdict

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_json import structured
from core.obs import TraceEvent
from core.schemas import EvidenceItem, ResearchBrief, ResearchState

_SYS = (
    "You are a research summariser. Using ONLY the grounded findings provided, write a "
    "structured brief: themes, evidence (each claim with its source citation and "
    "confidence), an overall confidence, flagged uncertainties, and recommended actions. "
    "Never assert anything beyond the grounded findings."
)


def make_summariser_node(model):
    def summariser_node(state: ResearchState) -> dict:
        ev = TraceEvent("summariser", "running", "composing structured brief")
        ev.emit()

        report = state["critic_report"]
        findings = state["findings"]
        conf_by_claim = {v.claim: v.confidence for v in report.verdicts}
        grounded = {v.claim for v in report.verdicts if v.grounded}
        kept = [f for f in findings if f.claim in grounded]

        if not kept:
            brief = ResearchBrief(
                question=state["question"], themes=[], evidence=[], confidence=0.0,
                flagged_uncertainties=(report.unsupported_claims
                                       or ["Insufficient grounded evidence to answer."]),
                recommended_actions=["Gather stronger sources before deciding."],
            )
            done = TraceEvent("summariser", "complete", "insufficient evidence")
            done.emit()
            return {"final": brief, "trace": [asdict(ev), asdict(done)]}

        evidence_block = "\n".join(
            f"- CLAIM: {f.claim}\n  SOURCE: {f.source_ref}\n  CONF: {conf_by_claim.get(f.claim, 0.0)}"
            for f in kept
        )
        user = (f"Question: {state['question']}\n\nGrounded findings:\n{evidence_block}\n\n"
                f"Flagged (exclude from claims, list under uncertainties): "
                f"{report.unsupported_claims or 'none'}")

        brief = structured(model, ResearchBrief,
                           [SystemMessage(content=_SYS), HumanMessage(content=user)])

        # Enforce invariants deterministically rather than trusting the model:
        brief.question = state["question"]
        brief.confidence = report.mean_confidence
        for uc in report.unsupported_claims:
            if uc not in brief.flagged_uncertainties:
                brief.flagged_uncertainties.append(uc)
        if not brief.evidence:
            brief.evidence = [EvidenceItem(claim=f.claim, source=f.source_ref,
                                           confidence=conf_by_claim.get(f.claim, 0.0)) for f in kept]

        done = TraceEvent("summariser", "complete", f"{len(brief.evidence)} evidence items")
        done.emit()
        return {"final": brief, "trace": [asdict(ev), asdict(done)]}

    return summariser_node
