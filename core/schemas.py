"""Pydantic models for every agent boundary + the LangGraph state.

Deterministic schema validation at each handoff is the first guardrail (req 5):
no LLM call needed to reject a malformed agent output. Internal models (Finding,
ClaimVerdict) are separate from the presentation model (EvidenceItem) so the
final brief stays clean and structured-output-friendly.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

ToolName = Literal["web", "rag", "vision", "none"]


class SubTask(BaseModel):
    id: str
    description: str
    tool: ToolName = "none"
    rationale: str = ""


class Plan(BaseModel):
    """Planner structured output."""
    subtasks: list[SubTask] = Field(default_factory=list)


class Finding(BaseModel):
    """One piece of evidence a researcher produced. Internal."""
    claim: str
    source_type: Literal["web", "rag", "vision", "stub", "none"] = "stub"
    source_ref: str = ""           # the text the claim is grounded in (option 1)
    subtask_id: str | None = None


class ClaimVerdict(BaseModel):
    """Critic's per-claim judgement."""
    claim: str
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class CriticReport(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    mean_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_replan: bool = False


class EvidenceItem(BaseModel):
    """Presentation: a supported claim + its citation, for the final brief."""
    claim: str
    source: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ResearchBrief(BaseModel):
    question: str
    themes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    flagged_uncertainties: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


class ResearchState(TypedDict):
    """LangGraph state. `trace` accumulates across nodes (operator.add); every
    other field is overwritten by the node that owns it."""
    question: str
    image_path: str | None
    subtasks: list[SubTask]
    findings: list[Finding]
    critic_report: CriticReport | None
    replan_count: int
    final: ResearchBrief | None
    trace: Annotated[list, operator.add]
