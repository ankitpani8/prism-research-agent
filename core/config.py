"""Central, env-driven configuration. 12-factor: all config via environment.

This is the ONE place that reads os.environ. Everything else imports `settings`.
Reused convention from learning_AgenticAI: load_dotenv at import, fail loud on
missing required keys only when a feature that needs them is actually used.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Providers
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or None
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY") or None
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Tracing
    langsmith_tracing: bool = _flag("LANGCHAIN_TRACING_V2", False)
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "prism-research-agent")

    # Loop circuit breakers
    max_replans: int = int(os.getenv("PRISM_MAX_REPLANS", "2"))

    # Paths
    project_root: Path = PROJECT_ROOT
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    chroma_dir: Path = PROJECT_ROOT / "data" / "local" / "chroma"

    @property
    def has_web_search(self) -> bool:
        """Tavily if keyed, else keyless DDG — search is always available."""
        return True


settings = Settings()
