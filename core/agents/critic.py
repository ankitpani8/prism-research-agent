"""Critic / Validator (async) — LOCAL Ollama model; concurrent per-claim checks.

Deterministic cites_source check first (empty source_ref -> UNSUPPORTED, no LLM).
Findings WITH a source are judged concurrently via asyncio.gather. Grounding is
claim-vs-its-own-source (option 1).
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import settings
from core.llm_json import parse_json
from core.obs import TraceEvent, record_usage
from core.schemas import ClaimVerdict, CriticReport, ResearchState

_SYS = (
    "You are a strict grounding critic. Decide whether a CLAIM is supported by the "
    "SOURCE text ALONE (no outside knowledge). Respond JSON only: "
    '{\"grounded\": true|false, \"confidence\": 0.0-1.0, \"reason\": \"<short>\"}'
)


async def _judge_one(model, model_name, claim: str, source: str) -> ClaimVerdict:
    if not (source or "").strip():
        return ClaimVerdict(claim=claim, grounded=False, confidence=0.0,
                            reason="no source reference (deterministic)")
    try:
        resp = await model.ainvoke([SystemMessage(content=_SYS),
                                    HumanMessage(content=f"CLAIM:\n{claim}\n\nSOURCE:\n{source}")])
        record_usage(model_name, resp)
        data = parse_json(getattr(resp, "content", "")) or {}
        return ClaimVerdict(claim=claim, grounded=bool(data.get("grounded", False)),
                            confidence=float(data.get("confidence", 0.0) or 0.0),
                            reason=str(data.get("reason", ""))[:200])
    except Exception as e:
        return ClaimVerdict(claim=claim, grounded=False, confidence=0.3,
                            reason=f"critic failed ({type(e).__name__}) - flagged")


def make_critic_node(model, model_name: str = "critic"):
    async def critic_node(state: ResearchState) -> dict:
        findings = state["findings"]
        ev = TraceEvent("critic", "running", f"validating {len(findings)} claims (local model)")
        ev.emit()
        verdicts = await asyncio.gather(
            *(_judge_one(model, model_name, f.claim, f.source_ref) for f in findings)
        )
        verdicts = list(verdicts)
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
