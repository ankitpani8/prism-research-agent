"""Prism CLI — the recordable "code executing" path. Prints the live agent trace
to the terminal, then the structured brief.

    python run_demo.py
    python run_demo.py "Your question"
    python run_demo.py "Your question" --image data/sample_dashboard.png
"""
from __future__ import annotations

import asyncio
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", message=r".*BaseApiClient.aclose.*")
logging.getLogger("asyncio").setLevel(logging.ERROR)

from core.format import confidence_tier  # noqa: E402
from core.runner import run_research  # noqa: E402

DEFAULT_Q = "What are the top drivers of prepaid customer complaints this quarter?"

# ANSI colours (fall back to plain if not a TTY)
_TTY = sys.stdout.isatty()
def _c(code, s): return f"\033[{code}m{s}\033[0m" if _TTY else s
DIM, BOLD, GREEN, YELLOW, RED, CYAN = "2", "1", "32", "33", "31", "36"

_ICON = {"planner": "🧭", "researcher": "🔎", "critic": "⚖️", "summariser": "📝",
         "vision": "🖼️", "guardrail": "🛡️"}
_COLOR = {"running": CYAN, "complete": GREEN, "flagged": YELLOW, "error": RED}


def _on_event(ev: dict) -> None:
    agent = ev.get("agent", "?")
    status = ev.get("status", "")
    detail = ev.get("detail", "")
    icon = _ICON.get(agent, "•")
    tag = _c(_COLOR.get(status, DIM), f"{status:<8}")
    print(f"  {icon} {_c(BOLD, agent.ljust(11))} {tag} {_c(DIM, detail)}")


def _badge(conf: float) -> str:
    code = {"high": GREEN, "medium": YELLOW, "low": RED}[confidence_tier(conf)]
    return _c(code, f"[{conf:.2f}]")


def parse_args(argv):
    image, pos, i = None, [], 0
    while i < len(argv):
        if argv[i] == "--image" and i + 1 < len(argv):
            image = argv[i + 1]
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    return (pos[0] if pos else DEFAULT_Q), image


async def main() -> int:
    question, image = parse_args(sys.argv[1:])
    print(_c(BOLD, "\n┌─ PRISM ─ multi-agent research assistant"))
    print(f"│ Q: {question}")
    if image:
        print(f"│ image: {image}")
    print(_c(DIM, "└" + "─" * 50) + "\n" + _c(BOLD, "AGENT TRACE"))

    result = await run_research(question, image, on_event=_on_event)
    brief = result["final"]
    tel = result["telemetry"]

    print("\n" + _c(BOLD, "BRIEF"))
    if not brief or not brief.evidence:
        print(_c(YELLOW, "  Insufficient grounded evidence."))
    else:
        if brief.themes:
            print("  " + _c(BOLD, "Themes: ") + ", ".join(brief.themes))
        print("  " + _c(BOLD, "Evidence:"))
        for e in brief.evidence:
            print(f"    {_badge(e.confidence)} {e.claim}")
            print(f"        {_c(DIM, 'source: ' + (e.source or 'n/a')[:80])}")
        if brief.recommended_actions:
            print("  " + _c(BOLD, "Recommended actions:"))
            for a in brief.recommended_actions:
                print(f"    • {a}")
    if brief and brief.flagged_uncertainties:
        print("  " + _c(YELLOW, "Flagged / uncertain:"))
        for u in brief.flagged_uncertainties:
            print(f"    {_c(YELLOW, '⚠')}  {u[:100]}")

    print("\n" + _c(DIM, f"overall confidence {brief.confidence if brief else 0.0} · "
                       f"re-plans {result['replan_count']} · "
                       f"{tel.llm_calls} calls · {tel.input_tokens + tel.output_tokens} tokens · "
                       f"${tel.cost_usd:.4f} · {tel.duration_ms:.0f}ms"))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
