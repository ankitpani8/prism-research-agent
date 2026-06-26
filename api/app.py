"""FastAPI production surface — the deployment story (NOT the demo UI).

  GET  /health   liveness — process is up.
  GET  /ready    readiness — corpus reachable AND a provider is reachable.
  POST /research SSE stream of per-agent trace events, then the final brief.

Thin wrapper over the SAME core.runner.run_research that the CLI and Streamlit
use. The per-request trace_id is threaded into telemetry and returned as the
X-Trace-Id header so a caller can correlate logs.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from core.config import settings
from core.logging_config import configure_logging, get_logger
from core.runner import run_research

configure_logging()
log = get_logger("prism.api")
app = FastAPI(title="Prism", version="1.0",
              description="Multi-agent research assistant — production surface.")


class ResearchRequest(BaseModel):
    question: str
    image_path: str | None = None


def _ollama_up() -> bool:
    try:
        base = settings.ollama_host.rstrip("/")
        return httpx.get(f"{base}/api/tags", timeout=1.0).status_code == 200
    except Exception:
        return False


def readiness_checks() -> dict:
    corpus = Path(settings.corpus_dir)
    return {
        "corpus": corpus.exists() and any(corpus.glob("*.md")),
        "provider": _ollama_up() or bool(
            settings.gemini_api_key or settings.anthropic_api_key or settings.openai_api_key),
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    checks = readiness_checks()
    ok = all(checks.values())
    return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)


async def _research_stream(question: str, image_path: str | None, trace_id: str):
    log.info(json.dumps({"event": "research_start", "trace_id": trace_id,
                         "question": question[:120]}))
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(ev: dict) -> None:
        queue.put_nowait(("trace", ev))

    async def run() -> None:
        try:
            result = await run_research(question, image_path, on_event=on_event,
                                        thread_id=trace_id, request_id=trace_id)
            brief = result["final"]
            tel = result["telemetry"]
            queue.put_nowait(("final", {
                "brief": brief.model_dump() if brief else None,
                "replan_count": result["replan_count"],
                "telemetry": {"request_id": tel.request_id, "llm_calls": tel.llm_calls,
                              "cost_usd": tel.cost_usd, "duration_ms": tel.duration_ms,
                              "by_model": tel.by_model}}))
        except Exception as e:
            log.exception(json.dumps({"event": "research_error", "trace_id": trace_id}))
            queue.put_nowait(("error", {"message": f"{type(e).__name__}: {str(e)[:200]}"}))
        finally:
            queue.put_nowait(("__done__", None))

    task = asyncio.create_task(run())
    try:
        while True:
            kind, data = await queue.get()
            if kind == "__done__":
                break
            yield {"event": kind, "data": json.dumps(data)}
    finally:
        await task
        log.info(json.dumps({"event": "research_end", "trace_id": trace_id}))


@app.post("/research")
async def research(req: ResearchRequest):
    trace_id = uuid4().hex[:8]
    return EventSourceResponse(_research_stream(req.question, req.image_path, trace_id),
                               headers={"X-Trace-Id": trace_id})
