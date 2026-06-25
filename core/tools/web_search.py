"""Web search tool — Tavily primary, keyless DuckDuckGo (ddgs) fallback.

M1 tool pattern: tools never raise into the graph. On any failure this returns
an EMPTY list; the researcher then produces an ungrounded finding, which the
critic flags and the planner re-plans around (graceful degradation, visible).
Async: Tavily via its async client; ddgs (sync) via asyncio.to_thread.
"""
from __future__ import annotations

import asyncio

from core.config import settings


def _ddgs_sync(query: str, max_results: int) -> list[dict]:
    try:
        from ddgs import DDGS
        with DDGS() as d:
            hits = list(d.text(query, max_results=max_results))
        return [{"title": h.get("title", ""), "url": h.get("href", ""),
                 "content": h.get("body", "")} for h in hits]
    except Exception:
        return []


async def _tavily(query: str, max_results: int) -> list[dict]:
    try:
        from tavily import AsyncTavilyClient
        client = AsyncTavilyClient(api_key=settings.tavily_api_key)
        resp = await client.search(query, max_results=max_results)
        return [{"title": r.get("title", ""), "url": r.get("url", ""),
                 "content": r.get("content", "")} for r in resp.get("results", [])]
    except Exception:
        return []


async def web_search(query: str, max_results: int = 4) -> list[dict]:
    """Return [{title, url, content}] (possibly empty). Never raises."""
    if settings.tavily_api_key:
        results = await _tavily(query, max_results)
        if results:
            return results
        # fall through to keyless if Tavily errored/empty
    return await asyncio.to_thread(_ddgs_sync, query, max_results)
