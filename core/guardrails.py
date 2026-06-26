"""Deterministic guardrails (req 5) — checkable WITHOUT an LLM call (M6 lesson).

  - cites_source: every asserted claim must carry a source reference.
  - lexical_overlap / deterministic_grounded: a claim that restates its source is
    grounded by construction; a fabricated claim shares little vocabulary with it.
  - screen_injection: an OWASP-LLM-style prompt-injection screen over user input
    and retrieved/searched text before it enters a prompt.

These are the FIRST line of defence: cheap, deterministic, demonstrable, and they
back both the critic (core/agents/critic.py) and the eval harness (eval/).
"""
from __future__ import annotations

import re

LEX_THRESHOLD = 0.5

_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "this", "that", "these", "those", "is", "are", "was", "were", "be", "been",
    "from", "by", "as", "at", "it", "its", "their", "most", "single",
    "relevant", "finding", "snippets", "source", "claim", "data", "current",
    "quarter", "regarding", "specific", "given", "provided", "identify",
}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 3 and w not in _STOP}


def lexical_overlap(claim: str, source: str) -> float:
    cw = content_words(claim)
    if not cw:
        return 0.0
    return len(cw & content_words(source)) / len(cw)


def cites_source(source_ref: str) -> bool:
    """Deterministic cites_source guardrail: is there ANY grounding text?"""
    return bool((source_ref or "").strip())


def deterministic_grounded(claim: str, source: str, threshold: float = LEX_THRESHOLD) -> bool:
    """No source -> not grounded. Else: claim restates source above threshold."""
    if not cites_source(source):
        return False
    return lexical_overlap(claim, source) >= threshold


# --- Input guardrail: prompt-injection screen (OWASP-LLM01) -------------------

_INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the|all|any)? ?(previous|prior|above)",
    r"you are now (a|an|in)\b",
    r"system prompt",
    r"reveal (your|the) (system )?(prompt|instructions)",
    r"forget (everything|all|your instructions)",
    r"act as (a|an)? ?(developer|dan|jailbreak)",
    r"\boverride\b.*\b(safety|guardrail|rule)s?\b",
    r"print (your|the) (instructions|system prompt)",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def screen_injection(text: str) -> list[str]:
    """Return the list of injection patterns matched (empty = clean)."""
    hits = []
    for rx in _INJECTION_RE:
        if rx.search(text or ""):
            hits.append(rx.pattern)
    return hits
