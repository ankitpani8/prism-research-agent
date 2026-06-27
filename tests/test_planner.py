"""Planner — an attached image MUST yield a vision subtask, even when the planner
LLM (e.g. a small local model) omits it. Regression guard for the bug where
qwen2.5:1.5b returned only [rag, web] and vision was silently dropped."""
import asyncio
from types import SimpleNamespace

from core.agents.planner import make_planner_node
from core.schemas import Plan, SubTask

USAGE = {"input_tokens": 1, "output_tokens": 1}


class _Structured:
    def __init__(self, plan):
        self._plan = plan

    async def ainvoke(self, messages):
        return {"raw": SimpleNamespace(content="{}", usage_metadata=USAGE),
                "parsed": self._plan, "parsing_error": None}


class FakePlanner:
    """Always returns rag+web only — i.e. it 'forgets' the vision subtask."""
    def __init__(self):
        self._plan = Plan(subtasks=[
            SubTask(id="s1", description="internal complaint categories", tool="rag"),
            SubTask(id="s2", description="external benchmarks", tool="web")])

    def with_structured_output(self, schema, include_raw=False):
        return _Structured(self._plan)


def _state(image_path):
    return {"question": "top prepaid complaint drivers", "image_path": image_path}


def test_vision_injected_when_image_present_even_if_llm_omits_it():
    node = make_planner_node(FakePlanner(), "light")
    out = asyncio.run(node(_state("data/sample_dashboard.png")))
    tools = [s.tool for s in out["subtasks"]]
    assert "vision" in tools, "an attached image must always produce a vision subtask"


def test_no_vision_subtask_without_image():
    node = make_planner_node(FakePlanner(), "light")
    out = asyncio.run(node(_state(None)))
    tools = [s.tool for s in out["subtasks"]]
    assert "vision" not in tools
    assert len(out["subtasks"]) >= 2
