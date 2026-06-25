"""Phase 2 manual run — text-only end-to-end pass with REAL models.

    python scripts/run_graph.py
    python scripts/run_graph.py "Your research question here"

Prints the live agent trace (planner -> research -> critic -> summariser, with
the re-plan loop) then the final structured brief. No image yet (Phase 4).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.graph import build_graph, initial_state  # noqa: E402
from core.obs import telemetry  # noqa: E402

DEFAULT_Q = "What are the top drivers of prepaid customer complaints this quarter?"


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_Q
    print(f"\nQUESTION: {question}\n" + "=" * 60)

    app = build_graph()
    cfg = {"configurable": {"thread_id": "phase2-demo"}}

    with telemetry(question):
        final_state = app.invoke(initial_state(question), config=cfg)

    brief = final_state["final"]
    print("\n" + "=" * 60 + "\nFINAL BRIEF\n" + "=" * 60)
    print(json.dumps(brief.model_dump(), indent=2))
    print(f"\nre-plans used: {final_state['replan_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
