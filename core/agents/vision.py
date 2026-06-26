"""Vision agent (async, multimodal) — reads a dashboard image and extracts
structured insights that flow into the research state as a grounded Finding.

Sends the image (base64) to a MULTIMODAL model (Gemini/Claude — never the local
1.5B model). The extracted readout becomes the Finding's source_ref, so the
image insight is grounded through the SAME critic as every other finding
(claim-vs-its-own-source). Failures are visible and never crash the graph.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from core.llm_json import astructured
from core.obs import TraceEvent
from core.schemas import Finding, SubTask, VisionInsights

_PROMPT = (
    "You are a dashboard analyst. Read ONLY what is visible in this image. Extract: "
    "the complaint categories with their percentages (highest first), the NPS trend "
    "(values and direction), and any notable anomalies. Do not invent numbers."
)

# Providers disagree on the multimodal content-block shape. Try the known-good
# forms in order; the first that the bound model accepts wins. (gemini via
# langchain-google-genai usually accepts the first two; the third is the
# langchain-core standard block.)
def _image_blocks(b64: str):
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        {"type": "image", "source_type": "base64", "data": b64, "mime_type": "image/png"},
    ]


def _warn(msg: str) -> None:
    print(f"[vision-error] {msg}", file=sys.stderr)


async def vision_finding(model, model_name: str, image_path: str, subtask: SubTask) -> Finding:
    p = Path(image_path)
    if not p.exists():
        _warn(f"image not found: {image_path}")
        return Finding(claim=f"[vision: image not found] {image_path}",
                       source_type="vision", source_ref="", subtask_id=subtask.id)
    if model is None:
        _warn("no multimodal model bound for the vision role")
        return Finding(claim="[vision: no multimodal model available]",
                       source_type="vision", source_ref="", subtask_id=subtask.id)
    try:
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        text_block = {"type": "text", "text": f"{_PROMPT}\n\nTask: {subtask.description}"}
        insights = None
        last_err: Exception | None = None
        for img_block in _image_blocks(b64):
            try:
                msg = HumanMessage(content=[text_block, img_block])
                insights = await astructured(model, VisionInsights, [msg], model_name=model_name)
                break
            except Exception as fmt_err:  # noqa: BLE001 - try the next content shape
                last_err = fmt_err
                continue
        if insights is None:
            raise last_err or RuntimeError("no accepted image format")
        readout = (f"Dashboard readout — Top complaint categories: "
                   f"{', '.join(insights.top_categories) or 'n/a'}. "
                   f"NPS: {insights.nps or 'n/a'}. "
                   f"Anomalies: {', '.join(insights.anomalies) or 'none'}.")
        claim = insights.summary.strip() or readout
        TraceEvent("vision", "complete",
                   f"{len(insights.top_categories)} categories, nps='{insights.nps[:30]}'").emit()
        return Finding(claim=claim, source_type="vision", source_ref=readout, subtask_id=subtask.id)
    except Exception as e:
        _warn(f"{type(e).__name__}: {str(e)[:160]}")
        return Finding(claim=f"[vision failed] {subtask.description}",
                       source_type="vision", source_ref="", subtask_id=subtask.id)
