"""Prism — Streamlit demo UI.

Thin presentation layer over the SAME core (core.runner.run_research). A live
per-agent trace panel (st.status) + the structured, cited, confidence-scored
brief. The agent core is framework-agnostic; this file only renders.

    streamlit run streamlit_app.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import warnings
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", message=r".*BaseApiClient.aclose.*")

from core.format import badge_html  # noqa: E402
from core.runner import run_research  # noqa: E402

DEFAULT_Q = "What are the top drivers of prepaid customer complaints this quarter?"
ICON = {"planner": "🧭", "researcher": "🔎", "critic": "⚖️", "summariser": "📝",
        "vision": "🖼️", "guardrail": "🛡️"}

st.set_page_config(page_title="Prism", page_icon="🔎", layout="wide")


class TracePanel:
    """Renders one st.status per agent, updated as trace events arrive."""

    def __init__(self):
        self._status = {}

    def handle(self, ev: dict) -> None:
        agent, status, detail = ev.get("agent", "?"), ev.get("status", ""), ev.get("detail", "")
        if agent not in self._status:
            self._status[agent] = st.status(f"{ICON.get(agent, '•')} {agent.title()}",
                                            state="running", expanded=(agent != "guardrail"))
        box = self._status[agent]
        box.write(f"`{status}` — {detail}")
        if status == "complete":
            box.update(state="complete")
        elif status in ("flagged", "error"):
            box.update(state="error",
                       label=f"{ICON.get(agent, '•')} {agent.title()} — {status}")


def _render_brief(result: dict) -> None:
    brief = result["final"]
    tel = result["telemetry"]

    st.subheader("Structured brief")
    if not brief or not brief.evidence:
        st.warning("Insufficient grounded evidence — the critic could not ground enough "
                   "claims. (The summariser refuses to assert what it cannot ground.)")
    if brief:
        if brief.themes:
            st.markdown("**Themes:** " + "  ".join(f"`{t}`" for t in brief.themes))
        if brief.evidence:
            st.markdown("**Evidence (grounded & cited):**")
            for e in brief.evidence:
                st.markdown(f"{badge_html(e.confidence)}&nbsp; {e.claim}",
                            unsafe_allow_html=True)
                st.caption(f"source: {(e.source or 'n/a')[:160]}")
        if brief.recommended_actions:
            st.markdown("**Recommended actions:**")
            for a in brief.recommended_actions:
                st.markdown(f"- {a}")
        if brief.flagged_uncertainties:
            with st.expander(f"⚠ Flagged / insufficient evidence ({len(brief.flagged_uncertainties)})"):
                for u in brief.flagged_uncertainties:
                    st.markdown(f"- {u}")

    st.divider()
    cols = st.columns(6)
    cols[0].metric("Confidence", f"{brief.confidence:.2f}" if brief else "0.00")
    cols[1].metric("Re-plans", result["replan_count"])
    cols[2].metric("LLM calls", tel.llm_calls)
    cols[3].metric("Tokens", tel.input_tokens + tel.output_tokens)
    cols[4].metric("Cost", f"${tel.cost_usd:.4f}")
    cols[5].metric("Latency", f"{tel.duration_ms/1000:.1f}s")
    if tel.by_model:
        st.caption("per-model: " + " · ".join(
            f"{m}: {d['calls']} calls, ${d['usd']:.4f}" for m, d in tel.by_model.items()))


# ---------------- layout ----------------
st.title("🔎 Prism")
st.caption("A multi-agent research assistant: it decomposes a question, gathers and "
           "cross-validates evidence across the web, internal documents, and a dashboard "
           "image, and returns a structured, cited, confidence-scored brief.")

left, right = st.columns([2, 1])
with left:
    question = st.text_area("Research question", value=DEFAULT_Q, height=90)
with right:
    uploaded = st.file_uploader("Dashboard image (optional)", type=["png", "jpg", "jpeg"])

run = st.button("Run research", type="primary")

if run:
    image_path = None
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.getvalue())
        tmp.close()
        image_path = tmp.name
        st.image(uploaded, caption="Input dashboard", width=440)

    st.subheader("Agent trace")
    panel = TracePanel()
    with st.spinner("Running the multi-agent pipeline…"):
        result = asyncio.run(run_research(question, image_path, on_event=panel.handle))
    _render_brief(result)
