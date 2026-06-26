# Screen-recording runsheet (3–5 min, MANDATORY deliverable)

The recording is graded as the proof-of-work. It must validate three things:
the solution is **functional and reproducible**, the candidate **understands the
implementation**, and the outputs **reflect actual system behaviour**. It must
show **code executing, not just the UI**, the agents collaborating end-to-end,
and a guardrail catching something — on real input.

## Which take to record

- **Primary take — with `GEMINI_API_KEY` working** (record when quota resets).
  Hosted models make each query ~10–20s (fits the time budget) and the **vision
  agent reads the dashboard**, so the brief carries image-derived numbers.
- **Backup take — fully local (`qwen2.5:1.5b` only).** Slower (~80s/query) but
  it proves graceful degradation at $0. Keep this as a second clip or a fallback
  if quota is unavailable; mention the resilience explicitly while it runs.

## Pre-flight (do BEFORE hitting record)

```bash
cd ~/projects/prism-research-agent && source .venv/bin/activate
python scripts/check_tools.py          # confirm rag=OK web=OK (warms the index)
ollama list                            # qwen2.5:1.5b present
# if recording the hosted take, confirm Gemini is live:
python -c "from core.byok import select_model_for_role as s; print(s('vision').name)"
```
Warming the index first means Chroma init doesn't eat recording time. Have two
terminals ready (one for the API/Streamlit server, one for curl), font size up.

## The script — timed beats (~4 min)

| # | Time | Show | Say (narration cue) |
|---|------|------|---------------------|
| 1 | 0:00–0:25 | `README.md` setup section; `.env` (keys redacted) | "Keyless-by-default: local critic on Ollama, ddgs fallback for search. One env var, `GEMINI_API_KEY`, lights up hosted + vision." |
| 2 | 0:25–0:55 | `docs/diagram.mmd` rendered + `core/graph.py` (nodes + conditional edges) | "Four LangGraph nodes, real control flow — Planner, Research fan-out, Critic, Summariser — with a bounded re-plan edge. `MAX_REPLANS` is the circuit breaker." |
| 3 | 0:55–1:45 | **`python run_demo.py "..." --image data/sample_dashboard.png`** — terminal trace | "This is the code executing. Watch the handoff: planner emits subtasks → researcher fans out async → **vision reads the dashboard** → critic validates each claim with a local model + a deterministic lexical check → summariser composes." |
| 4 | 1:45–2:00 | the printed brief (themes, evidence + confidence badges, sources) | "Every claim is cited and confidence-scored; the dashboard's categories/NPS appear as grounded evidence — multimodal feeding the output." |
| 5 | 2:00–3:00 | **`streamlit run streamlit_app.py`**, same query + upload | "Same core, different surface. The live `st.status` panel shows each agent's state — lots of current states beats a spinner. Here the **critic flags an unsupported claim** and it re-plans." *(If no claim flags on this query, use the planted-claim query below.)* |
| 6 | 3:00–3:40 | **`python eval/run_eval.py`** | "Evaluation is demonstrated, not described. Healthy groundedness ~0.91; then I weaken the grounding bar and **accuracy drops to ~0.55** — the harness catches a quality regression. This gates CI." |
| 7 | 3:40–4:00 | `pytest -q` (38 passed) and/or `docker compose up` | "38 keyless tests — including a planted-bad-claim guardrail test and the async-fan-out concurrency guard — plus a containerised FastAPI surface. Reproducible from the README." |

### Guaranteed guardrail-catch shot (beat 5)

If you want a query that *reliably* makes the critic flag + re-plan on camera,
ask for something the corpus can't ground (forces "insufficient evidence"):

```bash
python run_demo.py "What was the exact rand value of prepaid refunds issued last March?"
```
The synthesiser refuses to fabricate, the critic flags `UNSUPPORTED`, and you see
the re-plan loop — the anti-hallucination story, visible.

## Closing line (understanding)

"The senior decisions here: the critic is a *different model family* than the
generators — local Ollama vs hosted — so errors are decorrelated; deterministic
checks run before any LLM judge; and the one core is reused across CLI, Streamlit,
and a FastAPI surface. The constraint that I only had a low-RAM CPU box became the
design: local critic for $0 and decorrelated validation, hosted models for
planning, synthesis, and vision."

## Don't-forget checklist

- [ ] Terminal font large; window focused; no secrets on screen (`.env` values redacted).
- [ ] Index pre-warmed (`check_tools.py`) so Chroma init isn't on the clock.
- [ ] Show the **code running** (beats 3 & 6), not only the Streamlit UI.
- [ ] A guardrail visibly catches something (beat 5 or the planted-claim query).
- [ ] Real output on screen (the cited brief), matching what the code produced.
- [ ] Keep it 3–5 min — rehearse once; cut beat 7's docker if tight.
