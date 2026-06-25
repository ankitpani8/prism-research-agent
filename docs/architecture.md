# Prism — Architecture

🚧 Filled in Phase 8 (written overview + the Mermaid diagram exported from the
compiled graph + the §9 production write-up: scaling, and monitoring of
quality / cost / latency).

## Agents & interaction (target)

Planner → Research fan-out (Vision · Web · RAG) → Critic (local Ollama) →
Summariser, with a bounded re-plan loop when the Critic rejects claims.
