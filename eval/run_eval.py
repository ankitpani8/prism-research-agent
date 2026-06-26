"""Eval harness (req 5) — deterministic groundedness + cites_source + a REGRESSION
demo, all runnable OFFLINE with NO API keys (M6 "eval smoke on a stub" lesson).

Metrics over eval/eval_dataset.jsonl (labelled claim/source groundedness pairs +
out-of-knowledge cases):
  - groundedness accuracy: does our deterministic predictor match the label?
  - cites_source: are no-source claims correctly flagged?
  - refusal-correctness: do out-of-knowledge questions map to "ungrounded"
    (the system should say insufficient, not hallucinate).

REGRESSION demo: re-run with a DEGRADED config (lexical threshold dropped so it
grounds everything) and watch accuracy fall — proof the harness catches a quality
regression. Exits non-zero if the healthy config underperforms or the regression
is NOT detected, so CI gates on it.

    python eval/run_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.guardrails import LEX_THRESHOLD, deterministic_grounded  # noqa: E402

DATASET = Path(__file__).resolve().parent / "eval_dataset.jsonl"


def load_dataset() -> list[dict]:
    rows = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            rows.append(json.loads(line))
    return rows


def evaluate(rows: list[dict], threshold: float) -> dict:
    correct = cites_correct = 0
    cites_total = 0
    for r in rows:
        pred = deterministic_grounded(r["claim"], r["source"], threshold=threshold)
        label = (r["label"] == "grounded")
        correct += int(pred == label)
        if not r["source"].strip():
            cites_total += 1
            cites_correct += int(pred is False)  # no source MUST be flagged
    n = len(rows)
    return {
        "n": n,
        "accuracy": round(correct / n, 3) if n else 0.0,
        "cites_source_pass": (cites_correct == cites_total),
        "cites_checked": cites_total,
    }


def main() -> int:
    rows = load_dataset()
    print(f"=== Prism eval — {len(rows)} cases (deterministic, keyless) ===\n")

    healthy = evaluate(rows, threshold=LEX_THRESHOLD)
    print(f"[healthy ] threshold={LEX_THRESHOLD}  accuracy={healthy['accuracy']}  "
          f"cites_source={'PASS' if healthy['cites_source_pass'] else 'FAIL'} "
          f"({healthy['cites_checked']} no-source cases)")

    # REGRESSION: drop the lexical bar to 0 -> grounds everything with a source.
    degraded = evaluate(rows, threshold=0.0)
    print(f"[degraded] threshold=0.0  accuracy={degraded['accuracy']}  "
          f"(simulated guardrail regression)")

    drop = round(healthy["accuracy"] - degraded["accuracy"], 3)
    print(f"\nregression demo: accuracy dropped {healthy['accuracy']} -> "
          f"{degraded['accuracy']} (-{drop}) when grounding checks are weakened.")

    ok = (healthy["accuracy"] >= 0.8 and healthy["cites_source_pass"] and drop > 0.0)
    print(f"\n[eval] {'PASS' if ok else 'FAIL'} — healthy accuracy >= 0.8, "
          f"cites_source enforced, regression detected.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
