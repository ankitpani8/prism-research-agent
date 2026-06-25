"""Phase 3 manual run (async) — real tools, real grounding.

    python scripts/run_graph.py
    python scripts/run_graph.py "Your research question"
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Cosmetic: silence google-genai's async client teardown noise that fires AFTER
# results are produced (a known upstream cleanup wart, harmless to the output).
warnings.filterwarnings("ignore", message=r".*BaseApiClient.aclose.*")
logging.getLogger("asyncio").setLevel(logging.ERROR)

from core.graph import build_graph, initial_state  # noqa: E402
from core.obs import telemetry  # noqa: E402

DEFAULT_Q = "What are the top drivers of prepaid customer complaints this quarter?"


async def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_Q
    print(f"\nQUESTION: {question}\n" + "=" * 60)
    app = build_graph()
    cfg = {"configurable": {"thread_id": "phase3-demo"}}
    with telemetry(question):
        final_state = await app.ainvoke(initial_state(question), config=cfg)
    brief = final_state["final"]
    print("\n" + "=" * 60 + "\nFINAL BRIEF\n" + "=" * 60)
    print(json.dumps(brief.model_dump(), indent=2))
    print(f"\nre-plans used: {final_state['replan_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
