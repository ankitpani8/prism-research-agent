"""Phase 2: end-to-end graph wiring with FAKE models — runs in CI with NO keys.

Proves: the four-node graph executes start->finish; the re-plan loop fires when
the critic flags a claim and RESOLVES on re-plan; the MAX_REPLANS breaker stops
the loop and yields an honest "insufficient evidence" brief.
"""
from types import SimpleNamespace

from core.config import settings
from core.graph import build_graph, initial_state
from core.schemas import Plan, ResearchBrief, SubTask


class _FakeStructured:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, messages):
        return self._fn(messages)


class FakeModel:
    """Duck-typed stand-in for a LangChain chat model."""
    def __init__(self, structured_map=None, content="{}"):
        self._sm = structured_map or {}
        self._content = content

    def with_structured_output(self, schema):
        return _FakeStructured(self._sm[schema])

    def invoke(self, messages):
        # used by the critic's _judge_one
        c = self._content(messages) if callable(self._content) else self._content
        return SimpleNamespace(content=c)


def _planner_model():
    return FakeModel(structured_map={
        Plan: lambda _m: Plan(subtasks=[
            SubTask(id="s1", description="read the dashboard", tool="vision"),
            SubTask(id="s2", description="retrieve internal complaint categories", tool="rag"),
        ])
    })


def _summariser_model():
    return FakeModel(structured_map={
        ResearchBrief: lambda _m: ResearchBrief(
            question="q", themes=["network", "billing"],
            evidence=[], confidence=0.0,
            recommended_actions=["prioritise network fixes"],
        )
    })


def _models(critic_content):
    return {
        "light": _planner_model(),
        "critic": FakeModel(content=critic_content),
        "heavy": _summariser_model(),
    }


def test_replan_loop_fires_then_resolves():
    # Critic approves anything WITH a source; the deterministic empty-source check
    # flags finding #0 on pass 1, forcing exactly one re-plan that then resolves.
    models = _models('{"grounded": true, "confidence": 0.9, "reason": "ok"}')
    app = build_graph(models=models)
    out = app.invoke(initial_state("q"), config={"configurable": {"thread_id": "t1"}})

    assert out["replan_count"] == 1, "expected exactly one re-plan"
    assert out["final"] is not None
    assert out["final"].evidence, "resolved run should carry grounded evidence"
    assert out["final"].confidence > 0


def test_breaker_stops_at_max_replans_with_honest_brief():
    # Critic rejects everything -> loop can never resolve -> breaker must stop it.
    models = _models('{"grounded": false, "confidence": 0.1, "reason": "nope"}')
    app = build_graph(models=models)
    out = app.invoke(initial_state("q"), config={"configurable": {"thread_id": "t2"}})

    assert out["replan_count"] == settings.max_replans, "must stop at MAX_REPLANS"
    assert out["final"].evidence == [], "no grounded evidence -> empty"
    assert out["final"].flagged_uncertainties, "must honestly flag the gap"
    assert out["final"].confidence == 0.0
