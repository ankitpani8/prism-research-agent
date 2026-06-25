"""Diagnostic — exercise tools directly and isolate RAG failure modes.

    python scripts/check_tools.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402
from core.tools.vector_store import default_embed_fn, get_collection, rag_retrieve  # noqa: E402
from core.tools.web_search import web_search  # noqa: E402


def _embedding_selftest() -> bool:
    print("--- embedding self-test (all-MiniLM-L6-v2) ---")
    try:
        ef = default_embed_fn()
        vec = ef(["billing complaint about incorrect charges"])
        print(f"  OK — embedding dim={len(vec[0])}")
        return True
    except Exception as e:
        import traceback
        print(f"  FAIL — {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return False


def _corpus_facts():
    cdir = Path(settings.corpus_dir)
    files = sorted(p.name for p in cdir.glob("*") if p.suffix.lower() in (".md", ".txt"))
    print(f"--- corpus: {cdir} (exists={cdir.exists()}) — {len(files)} files ---")
    for f in files:
        print(f"   - {f}")
    try:
        coll = get_collection()
        print(f"  chroma collection '{coll.name}' count={coll.count()}")
    except Exception as e:
        print(f"  collection check FAIL — {type(e).__name__}: {e}")


async def main() -> int:
    _corpus_facts()
    emb_ok = _embedding_selftest()

    print("\n=== RAG (internal corpus) ===")
    rag = await rag_retrieve("internal prepaid complaint categories and frequencies", k=3)
    print(f"  returned {len(rag)} snippets")
    for r in rag[:3]:
        print(f"   - {r['source']}: {r['text'][:70].strip()}...")

    print("\n=== WEB (Tavily -> ddgs) ===")
    web = await web_search("telecom prepaid customer complaint top drivers benchmark", max_results=3)
    print(f"  returned {len(web)} results")

    print(f"\n[check] embedding={'OK' if emb_ok else 'FAIL'}  "
          f"rag={'OK' if rag else 'EMPTY'}  web={'OK' if web else 'EMPTY'}")
    return 0 if (rag and web) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
