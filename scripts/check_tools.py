"""Diagnostic — exercise the tools directly, independent of the graph.

    python scripts/check_tools.py

Shows exactly what rag_retrieve and web_search return, so an empty result in the
pipeline can be pinned to the tool (rate limit, key, index) rather than guessed.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tools.vector_store import rag_retrieve  # noqa: E402
from core.tools.web_search import web_search  # noqa: E402


async def main() -> int:
    print("=== RAG (internal corpus) ===")
    rag = await rag_retrieve("internal prepaid complaint categories and frequencies", k=3)
    print(f"  returned {len(rag)} snippets")
    for r in rag[:3]:
        print(f"   - {r['source']}: {r['text'][:70].strip()}...")

    print("\n=== WEB (Tavily -> ddgs) ===")
    web = await web_search("telecom prepaid customer complaint top drivers benchmark", max_results=3)
    print(f"  returned {len(web)} results")
    for r in web[:3]:
        print(f"   - {r.get('url','')[:60]}: {r.get('content','')[:70].strip()}...")

    ok = bool(rag) and bool(web)
    print(f"\n[check] rag={'OK' if rag else 'EMPTY'}  web={'OK' if web else 'EMPTY'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
