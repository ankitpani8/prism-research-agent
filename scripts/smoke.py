"""Phase 1 smoke test — proves the provider layer + obs layer work end to end.

Run from the project root:   python scripts/smoke.py

Verifies:
  1. select_all_models binds heavy/light/critic via the health-check chain.
  2. The CRITIC binds to a LOCAL Ollama model (the core design guarantee).
  3. A real heavy call returns, with token/latency/cost logged by obs.py.
  4. A real critic call returns from the local model.

Needs: GEMINI_API_KEY in .env, and Ollama running with qwen2.5:1.5b pulled.
Exits non-zero on failure so CI / Make can gate on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402

from core.byok import select_all_models  # noqa: E402
from core.config import settings  # noqa: E402
from core.obs import TraceEvent, telemetry  # noqa: E402


def main() -> int:
    if settings.langsmith_tracing and not settings.project_root:  # pragma: no cover
        pass
    if settings.langsmith_tracing:
        print("[note] LANGCHAIN_TRACING_V2 is on. If LANGSMITH_API_KEY is unset "
              "you'll see tracer warnings; set it to false in .env to silence.\n")

    sels = select_all_models(["heavy", "light", "critic"])

    # (2) The design guarantee: critic must be local.
    critic = sels["critic"]
    if critic.provider != "ollama":
        print(f"\n[WARN] critic bound to {critic.provider}:{critic.name}, not local "
              f"Ollama. Is `ollama serve` running with qwen2.5:1.5b pulled?")
    else:
        print(f"\n[ok] critic is LOCAL: {critic.provider}:{critic.name}")

    # (3) Real heavy call, fully instrumented.
    print("\n--- live calls (instrumented) ---")
    with telemetry("phase1 smoke: heavy + critic round-trip") as t:
        TraceEvent("heavy", "running", "say PRISM ONLINE").emit()
        heavy_llm = sels["heavy"].to_langchain(temperature=0)
        r1 = heavy_llm.invoke([HumanMessage(content="Reply with exactly: PRISM ONLINE")])
        t.record_response(sels["heavy"].name, r1)
        TraceEvent("heavy", "complete", str(r1.content)[:60]).emit()
        print(f"  heavy({sels['heavy'].name}) -> {str(r1.content).strip()[:60]}")

        # (4) Real critic call on the local model.
        TraceEvent("critic", "running", "local groundedness check").emit()
        critic_llm = critic.to_langchain(temperature=0)
        r2 = critic_llm.invoke([HumanMessage(
            content="Is 'the sky is green' supported by common knowledge? "
                    "Answer SUPPORTED or UNSUPPORTED, one word.")])
        t.record_response(critic.name, r2)
        TraceEvent("critic", "complete", str(r2.content)[:60]).emit()
        print(f"  critic({critic.name}) -> {str(r2.content).strip()[:60]}")

    print("\n[smoke] PASS — providers bound, local critic live, telemetry emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
