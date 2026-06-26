"""core.runner.run_research dispatches trace events and returns the final brief,
driven by a fake graph (no API keys)."""
import asyncio

from core.runner import run_research
from core.schemas import EvidenceItem, ResearchBrief


class _FakeApp:
    async def astream(self, state, config=None, stream_mode="updates"):
        yield {"planner": {"trace": [{"agent": "planner", "status": "running", "detail": "decomposing"}],
                           "replan_count": 0}}
        yield {"critic": {"trace": [{"agent": "critic", "status": "flagged", "detail": "1 flagged"}]}}
        yield {"summariser": {"trace": [{"agent": "summariser", "status": "complete", "detail": "done"}],
                              "final": ResearchBrief(question="q", themes=["billing"],
                                                     evidence=[EvidenceItem(claim="c", source="s.md",
                                                                            confidence=0.9)],
                                                     confidence=0.9)}}


def test_run_research_streams_events_and_returns_brief():
    events = []
    result = asyncio.run(run_research("q", on_event=events.append, app=_FakeApp()))
    agents = [e["agent"] for e in events]
    assert agents == ["planner", "critic", "summariser"]
    assert result["final"].evidence[0].confidence == 0.9
    assert result["replan_count"] == 0
    assert result["telemetry"] is not None
