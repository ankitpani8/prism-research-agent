"""Pure presentation helpers shared by the CLI and Streamlit UIs (testable)."""
from __future__ import annotations

TIER_COLOR = {"high": "#1a7f37", "medium": "#9a6700", "low": "#cf222e"}


def confidence_tier(c: float) -> str:
    if c >= 0.7:
        return "high"
    if c >= 0.4:
        return "medium"
    return "low"


def badge_html(c: float) -> str:
    color = TIER_COLOR[confidence_tier(c)]
    return (f'<span style="background:{color};color:#fff;padding:1px 8px;'
            f'border-radius:10px;font-size:0.78em;font-weight:700">{c:.2f}</span>')
