"""Provider layer — role-based model selection with critic-on-Ollama policy.

Port of learning_AgenticAI/lib/providers.py (select_model_for_role + health
checks). Phase 1 ports it verbatim and adds environment-aware critic routing
(local Ollama on laptop -> hosted small model in Cloud Run).
"""
