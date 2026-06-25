"""Phase 2: Pydantic boundary validation — the deterministic first guardrail.

No API keys, no models. Pure schema contracts.
"""
import pytest
from pydantic import ValidationError

from core.schemas import (
    ClaimVerdict,
    EvidenceItem,
    Finding,
    Plan,
    ResearchBrief,
    SubTask,
)


def test_subtask_tool_enum_enforced():
    SubTask(id="s1", description="d", tool="web")
    with pytest.raises(ValidationError):
        SubTask(id="s1", description="d", tool="telepathy")  # not in ToolName


def test_confidence_bounds_enforced():
    ClaimVerdict(claim="c", grounded=True, confidence=1.0)
    with pytest.raises(ValidationError):
        ClaimVerdict(claim="c", grounded=True, confidence=1.5)
    with pytest.raises(ValidationError):
        EvidenceItem(claim="c", source="s", confidence=-0.1)


def test_finding_defaults():
    f = Finding(claim="x")
    assert f.source_type == "stub" and f.source_ref == ""


def test_brief_roundtrips_through_json():
    brief = ResearchBrief(
        question="q",
        themes=["t1"],
        evidence=[EvidenceItem(claim="c", source="s", confidence=0.8)],
        confidence=0.8,
    )
    again = ResearchBrief.model_validate(brief.model_dump())
    assert again.evidence[0].confidence == 0.8


def test_plan_defaults_empty():
    assert Plan().subtasks == []
