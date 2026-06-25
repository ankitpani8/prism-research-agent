"""Web search tool — Tavily primary, keyless DuckDuckGo (ddgs) fallback.

M1 tool pattern: never raises into the graph. On failure returns [] AND prints a
visible [tool-error] line to stderr (so an empty result is diagnosable, not
silent). ddgs API verified: DDGS().text(query, max_results=N) -> [{title,href,body}].
"""
from __future__ import annotations

import asyncio
import sys

from core.config import settings


def _warn(msg: str) -> None:
    print(f"[tool-error] web_search: {msg}", file=sys.stderr)


def _ddgs_sync(query: str, max_results: int) -> list[dict]:
    try:
        from ddgs import DDGS
        hits = DDGS().text(query, max_results=max_results)
        return [{"title": h.get("title", ""), "url": h.get("href", ""),
                 "content": h.get("body", "")} for h in hits]
    except Exception as e:
        _warn(f"ddgs {type(e).__name__}: {str(e)[:120]}")
        return []


async def _tavily(query: str, max_results: int) -> list[dict]:
    try:
        from tavily import AsyncTavilyClient
        client = AsyncTavilyClient(api_key=settings.tavily_api_key)
        resp = await client.search(query, max_results=max_results)
        return [{"title": r.get("title", ""), "url": r.get("url", ""),
                 "content": r.get("content", "")} for r in resp.get("results", [])]
    except Exception as e:
        _warn(f"tavily {type(e).__name__}: {str(e)[:120]}")
        return []


async def web_search(query: str, max_results: int = 4) -> list[dict]:
    """Return [{title, url, content}] (possibly empty). Never raises."""
    if settings.tavily_api_key:
        results = await _tavily(query, max_results)
        if results:
            return results
        _warn("tavily returned no results; falling back to ddgs")
    return await asyncio.to_thread(_ddgs_sync, query, max_results)
