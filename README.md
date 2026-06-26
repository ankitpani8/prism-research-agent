# Prism — Multi-Agent Research Assistant

> Prism is a multi-agent research assistant that decomposes a question, gathers
> and cross-validates evidence across the web, documents, and a dashboard image,
> and returns a structured, cited, confidence-scored brief. Like a prism
> splitting light, the orchestrator splits one question into a spectrum of
> sub-tasks.

*Senior Data Scientist (CX + Agentic AI) assessment — built on **LangGraph**.*

A four-agent LangGraph state machine (**Planner → Research fan-out → Critic →
Summariser**) with a bounded re-plan loop, deterministic-first guardrails, real
tools (web search, vector RAG, multimodal vision), per-request token/cost/latency
telemetry, and one framework-agnostic core driven from a **CLI**, a **Streamlit**
demo, and a **FastAPI** surface.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full overview
(roles, state, tiered model routing, guardrails, production design, CX framing)
and [`docs/diagram.mmd`](docs/diagram.mmd) for the diagram exported from the
compiled graph.

```
Planner → Research (Vision ‖ Web ‖ RAG, async fan-out) → Critic → Summariser
              ▲                                              │
              └───────────── re-plan (MAX_REPLANS=2) ────────┘
```

- **Planner** decomposes the question into typed subtasks and runs the re-plan loop.
- **Research** fans out concurrently: **Vision** (reads a dashboard image),
  **Web** (Tavily → keyless ddgs fallback), **RAG** (Chroma over a synthetic CX corpus).
- **Critic** runs on a **local Ollama model** + deterministic checks (decorrelated
  errors, $0), scores per-claim groundedness/confidence, bounces unsupported claims.
- **Summariser** emits a Pydantic-validated brief and refuses to assert anything
  the critic could not ground.

## Setup

Requires **Python 3.11** and (for the local critic) **Ollama**.

```bash
# 1. clone + venv
git clone https://github.com/ankitpani8/prism-research-agent.git
cd prism-research-agent
python3.11 -m venv .venv && source .venv/bin/activate

# 2. dependencies (no torch — embeddings via Chroma ONNX)
pip install -r requirements.txt        # or requirements-dev.txt for tests/CI

# 3. config
cp .env.example .env                   # then edit (all keys optional — see below)

# 4. local critic model
ollama pull qwen2.5:1.5b

# 5. build the RAG index + the sample dashboard
python scripts/build_index.py
python scripts/make_dashboard.py
```

### Environment variables (all optional — the repo runs keyless)

| Var | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Primary hosted model (multimodal vision, planning, synthesis) | — |
| `TAVILY_API_KEY` | Web search; **ddgs keyless fallback** if unset | — |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Optional hosted fallbacks | — |
| `OLLAMA_HOST` | Local critic endpoint | `http://localhost:11434` |
| `PRISM_MAX_REPLANS` | Re-plan circuit breaker | `2` |

With **no keys at all**, Prism runs fully locally: the critic and (as a last
resort) planning/synthesis use `qwen2.5:1.5b`, web search uses keyless ddgs, and
vision degrades gracefully to "unavailable". With `GEMINI_API_KEY` set, vision
and the hosted roles activate and the brief is richer.

## Run

```bash
# CLI — prints the live agent trace, then the brief (the "code executing" view)
python run_demo.py "What are the top drivers of prepaid customer complaints this quarter?"
python run_demo.py "..." --image data/sample_dashboard.png      # multimodal

# Streamlit demo UI — live st.status trace panel + brief
streamlit run streamlit_app.py

# FastAPI surface
uvicorn api.app:app --reload
curl localhost:8000/health
curl -N -X POST localhost:8000/research -H 'content-type: application/json' \
  -d '{"question":"What are the top drivers of prepaid customer complaints this quarter?"}'

# Containerised (Ollama stays on the host)
docker compose up --build
```

See [`examples/sample_query.md`](examples/sample_query.md) and
[`examples/sample_output.md`](examples/sample_output.md) for a real run.

## Tests & evaluation

```bash
ruff check . && pytest -q        # 38 tests, keyless (fake models, offline Chroma, stubbed search)
python eval/run_eval.py          # groundedness + cites_source + regression demo
```

CI (`.github/workflows/ci.yml`) runs lint + the full rail + the eval smoke on
every push — **green with no live keys**.

## Project layout

```
core/        framework-agnostic product: graph, agents, tools, byok, obs, guardrails, schemas
api/         FastAPI surface (/health, /ready, /research SSE)
streamlit_app.py · run_demo.py    presentation layers over core.runner
data/        synthetic CX corpus (+ SOURCES.md) and the sample dashboard
eval/        labelled dataset + groundedness/regression harness
tests/       tool contracts · schema validation · guardrail catch · concurrency · API
docs/        architecture.md + exported diagram.mmd
```

## Notes

- **Synthetic data only** — no customer PII. Category figures are grounded in
  public regulator disclosures (CCTS / FCC / Ofcom), cited in `data/corpus/SOURCES.md`.
- **Embeddings** use Chroma's ONNX `DefaultEmbeddingFunction` (all-MiniLM-L6-v2)
  — no torch, low RAM. `get_collection()` is the swap seam for a Chroma server /
  Pinecone / Weaviate.
