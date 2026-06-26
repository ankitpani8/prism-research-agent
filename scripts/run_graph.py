"""Manual run (async) — real tools, real grounding, optional dashboard image.

    python scripts/run_graph.py
    python scripts/run_graph.py "Your question"
    python scripts/run_graph.py "Your question" --image data/sample_dashboard.png
    python scripts/run_graph.py --image data/sample_dashboard.png
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", message=r".*BaseApiClient.aclose.*")
logging.getLogger("asyncio").setLevel(logging.ERROR)

from core.graph import build_graph, initial_state  # noqa: E402
from core.obs import telemetry  # noqa: E402

DEFAULT_Q = "What are the top drivers of prepaid customer complaints this quarter?"


def parse_args(argv: list[str]) -> tuple[str, str | None]:
    """Return (question, image_path). --image takes the next token; the first
    non-flag token is the question. Order-independent."""
    image, positionals = None, []
    i = 0
    while i < len(argv):
        if argv[i] == "--image" and i + 1 < len(argv):
            image = argv[i + 1]
            i += 2
        else:
            positionals.append(argv[i])
            i += 1
    question = positionals[0] if positionals else DEFAULT_Q
    return question, image


async def main() -> int:
    question, image = parse_args(sys.argv[1:])
    print(f"\nQUESTION: {question}" + (f"\nIMAGE: {image}" if image else "") + "\n" + "=" * 60)
    app = build_graph()
    cfg = {"configurable": {"thread_id": "prism-demo"}}
    with telemetry(question):
        final_state = await app.ainvoke(initial_state(question, image_path=image), config=cfg)
    brief = final_state["final"]
    print("\n" + "=" * 60 + "\nFINAL BRIEF\n" + "=" * 60)
    print(json.dumps(brief.model_dump(), indent=2))
    print(f"\nre-plans used: {final_state['replan_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
