"""End-to-end graph wiring with FAKE models + FAKE tools (async) — keyless CI.

Proves: the four-node async graph runs start->finish; the re-plan loop fires when
the critic flags a claim and RESOLVES on re-plan; the MAX_REPLANS breaker stops
the loop and yields an honest "insufficient evidence" brief.

Scenarios control claim<->source lexical overlap explicitly, since the critic now
combines a deterministic overlap signal with the (fake) LLM verdict.
"""
from types import SimpleNamespace

from core.config import settings
from core.graph import build_graph, initial_state
from core.schemas import Plan, ResearchBrief, SubTask

USAGE = {"input_tokens": 5, "output_tokens": 5}


class _Structured:
    def __init__(self, fn):
        self._fn = fn

    async def ainvoke(self, messages):
        return {"raw": SimpleNamespace(content="{}", usage_metadata=USAGE),
                "parsed": self._fn(messages), "parsing_error": None}


class FakeModel:
    def __init__(self, structured_map=None, content="{}"):
        self._sm = structured_map or {}
        self._content = content

    def with_structured_output(self, schema, include_raw=False):
        return _Structured(self._sm[schema])

    async def ainvoke(self, messages):
        c = self._content(messages) if callable(self._content) else self._content
        return SimpleNamespace(content=c, usage_metadata=USAGE)


def _planner():
    return FakeModel(structured_map={Plan: lambda _m: Plan(subtasks=[
        SubTask(id="s1", description="internal complaint categories", tool="rag"),
        SubTask(id="s2", description="external benchmarks", tool="web"),
    ])})


def _summariser():
    return FakeModel(structured_map={ResearchBrief: lambda _m: ResearchBrief(
        question="q", themes=["billing"], evidence=[], confidence=0.0,
        recommended_actions=["prioritise billing fixes"])})


def _models(critic_content, claim_text):
    light = _planner()
    light._content = claim_text  # researcher synthesis claim
    return {"light": light, "critic": FakeModel(content=critic_content), "heavy": _summariser()}


class _FlakyTool:
    """Empty on the first call, returns an overlapping hit afterwards -> forces
    exactly one re-plan that then resolves."""
    def __init__(self, text):
        self.calls = 0
        self.text = text

    async def __call__(self, _q):
        self.calls += 1
        return [] if self.calls == 1 else [{"text": self.text, "source": "s.md"}]


def _hits(text):
    async def _t(_q):
        return [{"text": text, "source": "s.md"}]
    return _t


async def test_replan_loop_fires_then_resolves():
    claim = "billing is the single largest driver of prepaid complaints"
    models = _models('{"grounded": true, "confidence": 0.9, "reason": "ok"}', claim)
    # rag overlaps the claim (grounds); web is flaky (empty pass1 -> replan -> resolves)
    tools = {"rag": _hits(claim), "web": _FlakyTool(claim)}
    app = build_graph(models=models, tools=tools,
                      model_names={"light": "fake", "critic": "fake", "heavy": "fake"})
    out = await app.ainvoke(initial_state("q"), config={"configurable": {"thread_id": "t1"}})
    assert out["replan_count"] == 1, "expected exactly one re-plan"
    assert out["final"] is not None
    assert out["final"].evidence, "resolved run should carry grounded evidence"


async def test_breaker_stops_at_max_replans():
    # Claim shares NO content words with the source, and the critic rejects -> the
    # deterministic overlap can't rescue it -> loop hits the breaker.
    claim = "zzz alpha beta gamma delta"
    models = _models('{"grounded": false, "confidence": 0.1, "reason": "nope"}', claim)
    tools = {"rag": _hits("completely unrelated wording here"),
             "web": _hits("more unrelated wording entirely")}
    app = build_graph(models=models, tools=tools,
                      model_names={"light": "fake", "critic": "fake", "heavy": "fake"})
    out = await app.ainvoke(initial_state("q"), config={"configurable": {"thread_id": "t2"}})
    assert out["replan_count"] == settings.max_replans
    assert out["final"].evidence == []
    assert out["final"].flagged_uncertainties
