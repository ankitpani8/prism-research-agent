# Prism — Multi-Agent Research Assistant

> Prism is a multi-agent research assistant that decomposes a question, gathers
> and cross-validates evidence across the web, documents, and visual inputs, and
> returns a structured, cited, confidence-scored brief. Like a prism splitting
> light, the orchestrator splits one question into a spectrum of sub-tasks.

*Senior Data Scientist (CX + Agentic AI) assessment — built on LangGraph.*

---

## Status

🚧 Phase 0 — scaffold. Implementation lands phase by phase (see the build plan).

## What it does (target)

A LangGraph state machine with four agents in clear role separation:

- **Planner / Orchestrator** — decomposes the question, routes tools, runs the re-plan loop.
- **Research agents (fan-out)** — Vision (reads a dashboard image), Web search (Tavily/DDG), RAG (vector DB over a CX corpus).
- **Critic / Validator** — runs on a *local* Ollama model (decorrelated errors), scores per-claim groundedness and confidence, bounces unsupported claims back.
- **Summariser** — emits a Pydantic-validated structured brief: themes → evidence + citations → confidence → flagged uncertainties → recommended actions.

The same framework-agnostic `core/` is driven from a CLI, a Streamlit demo, and a FastAPI surface.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your GEMINI_API_KEY
# (optional) local critic model:  ollama pull qwen2.5:1.5b
```

Run the demo:

```bash
make run            # Streamlit UI
make demo Q="What are the top drivers of prepaid complaints this quarter?"   # CLI trace
make test           # the test rail
make eval           # eval harness + regression demo
```

## Architecture

See [`docs/architecture.md`](docs/architecture.md). The diagram in
[`docs/diagram.mmd`](docs/diagram.mmd) is exported directly from the compiled
LangGraph graph (`make diagram`).

## CX application

The same architecture serves complaint triage, call-centre insight mining, and
NPS root-cause analysis — swap the corpus and the dashboard, keep the agents.
All demo data is synthetic (no real customer PII).

## Configuration

All config via environment (`.env`). Required: `GEMINI_API_KEY`. Optional:
`TAVILY_API_KEY` (search falls back to keyless DuckDuckGo if unset),
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` (BYOK fallbacks), `OLLAMA_HOST`,
`LANGSMITH_*` (tracing).

## License

MIT — see [LICENSE](LICENSE).
