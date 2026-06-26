"""FastAPI surface — keyless. /health, /ready structure, and /research SSE with a
monkeypatched runner (no models needed)."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.app as apimod
from api.app import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ready_reports_checks():
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    body = r.json()
    assert "checks" in body and "corpus" in body["checks"] and "provider" in body["checks"]
    assert body["checks"]["corpus"] is True  # corpus ships in the repo


@pytest.fixture
def _fake_runner(monkeypatch):
    async def fake_run_research(question, image_path=None, on_event=None,
                                thread_id="x", request_id=None, app=None):
        if on_event:
            on_event({"agent": "planner", "status": "running", "detail": "decomposing"})
            on_event({"agent": "summariser", "status": "complete", "detail": "done"})

        tel = SimpleNamespace(request_id=request_id or "tid", llm_calls=2, cost_usd=0.0,
                              duration_ms=12.0,
                              by_model={"qwen2.5:1.5b": {"calls": 2, "in": 1, "out": 1, "usd": 0.0}})
        brief = SimpleNamespace(model_dump=lambda: {
            "question": question, "themes": ["billing"], "evidence": [],
            "confidence": 0.0, "flagged_uncertainties": [], "recommended_actions": []})
        return {"final": brief, "replan_count": 0, "telemetry": tel}

    monkeypatch.setattr(apimod, "run_research", fake_run_research)


def test_research_sse_streams_trace_then_final(_fake_runner):
    r = client.post("/research", json={"question": "q"})
    assert r.status_code == 200
    assert "X-Trace-Id" in r.headers
    body = r.text
    assert "event: trace" in body
    assert "event: final" in body
    # the final event carries a serialisable brief payload
    final_line = [ln for ln in body.splitlines() if ln.startswith("data:") and "themes" in ln][0]
    payload = json.loads(final_line[len("data:"):].strip())
    assert payload["brief"]["themes"] == ["billing"]
    assert payload["telemetry"]["cost_usd"] == 0.0
