"""Vision agent — offline / keyless.

A fake multimodal model returns structured VisionInsights; we assert the image
insight flows into a grounded Finding (source_type='vision', readout in
source_ref). Also covers the missing-image and no-model graceful paths.
"""
import asyncio
from types import SimpleNamespace

from core.agents.vision import vision_finding
from core.schemas import SubTask, VisionInsights

USAGE = {"input_tokens": 5, "output_tokens": 5}


class _Structured:
    def __init__(self, value):
        self._value = value

    async def ainvoke(self, messages):
        return {"raw": SimpleNamespace(content="{}", usage_metadata=USAGE),
                "parsed": self._value, "parsing_error": None}


class FakeVisionModel:
    def __init__(self, insights):
        self._insights = insights

    def with_structured_output(self, schema, include_raw=False):
        return _Structured(self._insights)


def _img(tmp_path):
    p = tmp_path / "dash.png"
    p.write_bytes(b"\\x89PNG\\r\\n\\x1a\\n fake image bytes")
    return str(p)


def test_vision_insight_becomes_grounded_finding(tmp_path):
    insights = VisionInsights(
        summary="Billing is the top prepaid complaint driver at 27%, NPS down to 22",
        top_categories=["Billing 27%", "Network 24%", "Recharge 18%"],
        nps="NPS down from 28 to 22 (-6 over the year)",
        anomalies=["Billing rose 4 pts QoQ"])
    model = FakeVisionModel(insights)
    st = SubTask(id="v1", description="read the dashboard", tool="vision")
    f = asyncio.run(vision_finding(model, "fake-vision", _img(tmp_path), st))
    assert f.source_type == "vision"
    assert "Billing 27%" in f.source_ref
    assert "22" in f.source_ref
    assert f.source_ref  # grounded (non-empty source)
    assert f.claim


def test_vision_missing_image_is_graceful(tmp_path):
    st = SubTask(id="v1", description="read the dashboard", tool="vision")
    f = asyncio.run(vision_finding(FakeVisionModel(VisionInsights()),
                                   "fake", str(tmp_path / "nope.png"), st))
    assert f.source_type == "vision"
    assert f.source_ref == ""  # ungrounded -> critic will flag -> re-plan


def test_vision_no_model_is_graceful(tmp_path):
    st = SubTask(id="v1", description="read the dashboard", tool="vision")
    f = asyncio.run(vision_finding(None, "fake", _img(tmp_path), st))
    assert f.source_ref == ""
