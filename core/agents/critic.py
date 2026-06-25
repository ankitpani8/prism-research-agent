"""Critic / Validator — runs on the LOCAL Ollama model (decorrelated errors).

Grounding model (option 1): each claim is judged against ITS OWN source_ref. The
deterministic cites_source check runs FIRST — a finding with no source_ref is
flagged UNSUPPORTED with zero LLM cost (M6: prefer deterministic when checkable).
Only findings that HAVE a source incur an LLM judgement.
"""
from __future__ import annotations

from dataclasses import asdict

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import settings
from core.llm_json import parse_json
from core.obs import TraceEvent
from core.schemas import ClaimVerdict, CriticReport, ResearchState

_SYS = (
    "You are a strict grounding critic. Decide whether a CLAIM is supported by the "
    "provided SOURCE text alone. Do not use outside knowledge. Respond with JSON only: "
    '{\"grounded\": true|false, \"confidence\": 0.0-1.0, \"reason\": \"<short>\"}'
)


def _judge_one(model, claim: str, source: str) -> ClaimVerdict:
    prompt = f"CLAIM:\n{claim}\n\nSOURCE:\n{source}"
    try:
        raw = model.invoke([SystemMessage(content=_SYS), HumanMessage(content=prompt)])
        data = parse_json(getattr(raw, "content", "")) or {}
        return ClaimVerdict(
            claim=claim,
            grounded=bool(data.get("grounded", False)),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            reason=str(data.get("reason", ""))[:200],
        )
    except Exception as e:
        # Conservative: if the critic can't verify, FLAG it (safer than passing).
        return ClaimVerdict(claim=claim, grounded=False, confidence=0.3,
                            reason=f"critic parse/exec failed ({type(e).__name__}) - flagged")


def make_critic_node(model):
    def critic_node(state: ResearchState) -> dict:
        findings = state["findings"]
        ev = TraceEvent("critic", "running", f"validating {len(findings)} claims (local model)")
        ev.emit()

        verdicts: list[ClaimVerdict] = []
        for f in findings:
            src = (f.source_ref or "").strip()
            if not src:
                # deterministic cites_source check — no LLM call
                verdicts.append(ClaimVerdict(claim=f.claim, grounded=False, confidence=0.0,
                                             reason="no source reference (deterministic)"))
            else:
                verdicts.append(_judge_one(model, f.claim, src))

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
