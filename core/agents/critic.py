"""Critic / Validator (async) — LOCAL Ollama model + DETERMINISTIC guardrails.

Three layers, deterministic first (see core/guardrails.py):
  1. cites_source  -> no source_ref => UNSUPPORTED, no LLM call.
  2. lexical_overlap -> a claim that restates its source is grounded by
     construction (robust to the 1.5B critic's noise).
  3. LLM judgement -> catches semantic drift the lexical check would miss.
Final: grounded = (LLM grounded) OR (deterministic grounded). A fabricated claim
has low overlap AND fails the LLM check -> stays flagged.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import settings
from core.guardrails import LEX_THRESHOLD, cites_source, lexical_overlap
from core.llm_json import parse_json
from core.obs import TraceEvent, record_usage
from core.schemas import ClaimVerdict, CriticReport, ResearchState

_SYS = (
    "You are a grounding critic. Decide whether the CLAIM is SUPPORTED BY or "
    "CONSISTENT WITH the SOURCE text (do not require identical wording; a faithful "
    "paraphrase IS supported). Use only the SOURCE, not outside knowledge. Respond "
    'JSON only: {\"grounded\": true|false, \"confidence\": 0.0-1.0, \"reason\": \"<short>\"}'
)


async def _judge_one(model, model_name, claim: str, source: str) -> ClaimVerdict:
    if not cites_source(source):
        return ClaimVerdict(claim=claim, grounded=False, confidence=0.0,
                            reason="no source reference (deterministic cites_source)")
    overlap = lexical_overlap(claim, source)
    det_grounded = overlap >= LEX_THRESHOLD

    llm_grounded, llm_conf, reason = det_grounded, 0.0, ""
    try:
        resp = await model.ainvoke([SystemMessage(content=_SYS),
                                    HumanMessage(content=f"CLAIM:\n{claim}\n\nSOURCE:\n{source[:1200]}")])
        record_usage(model_name, resp)
        data = parse_json(getattr(resp, "content", "")) or {}
        llm_grounded = bool(data.get("grounded", False))
        llm_conf = float(data.get("confidence", 0.0) or 0.0)
        reason = str(data.get("reason", ""))[:140]
    except Exception as e:
        reason = f"critic exec failed ({type(e).__name__})"

    grounded = llm_grounded or det_grounded
    confidence = round(max(llm_conf, overlap), 3) if grounded else round(min(llm_conf, 0.3), 3)
    note = f"{reason} [lexical={overlap:.2f}{'/det-grounded' if det_grounded else ''}]"
    return ClaimVerdict(claim=claim, grounded=grounded, confidence=confidence, reason=note[:200])


def make_critic_node(model, model_name: str = "critic"):
    async def critic_node(state: ResearchState) -> dict:
        findings = state["findings"]
        ev = TraceEvent("critic", "running",
                        f"validating {len(findings)} claims (local model + lexical check)")
        ev.emit()
        verdicts = list(await asyncio.gather(
            *(_judge_one(model, model_name, f.claim, f.source_ref) for f in findings)))
        unsupported = [v.claim for v in verdicts if not v.grounded]
        mean_conf = round(sum(v.confidence for v in verdicts) / len(verdicts), 3) if verdicts else 0.0
        replan_count = state.get("replan_count", 0)
        needs_replan = bool(unsupported) and replan_count < settings.max_replans
        report = CriticReport(verdicts=verdicts, unsupported_claims=unsupported,
                              mean_confidence=mean_conf, needs_replan=needs_replan)
        done = TraceEvent("critic", "flagged" if unsupported else "complete",
                          f"validated {len(verdicts)} claims, {len(unsupported)} flagged "
                          f"(conf {mean_conf}){' -> re-plan' if needs_replan else ''}")
        done.emit()
        return {"critic_report": report, "trace": [asdict(ev), asdict(done)]}

    return critic_node
