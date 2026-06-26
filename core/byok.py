"""Provider layer — role-based startup model selection with a health-check chain.

Near-verbatim port of learning_AgenticAI/lib/providers.py. Changes from the
original, all minor:
  - Reads keys/host from core.config.settings (one place reads env — 12-factor)
    instead of calling load_dotenv() itself.
  - ChatOllama is given base_url from OLLAMA_HOST so the critic honors the env.
  - Dropped the to_crewai() builder (Prism is LangGraph-only; dead code removed).

The role policy is unchanged and is the whole point: the *critic* role is
local-first (Ollama qwen2.5:1.5b). A different model family than the generators
gives decorrelated errors (the anti-hallucination story) AND costs $0 (the cost
lever). Adding a provider later means editing this one table — no agent code.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from core.config import settings

# ---------------------------------------------------------------------------
# Role policy — the one place that knows which models fit which jobs.
# ---------------------------------------------------------------------------

ROLE_PREFERENCES: dict[str, list[tuple[str, str]]] = {
    # Strong synthesis / writing (Summariser, re-plan reasoning). Degrade through
    # hosted-small models for QUALITY before ever touching the tiny local model.
    "heavy": [
        ("qwen2.5:1.5b", "ollama"),
        ("gemini-2.5-flash", "gemini"),
        ("gemini-2.5-flash-lite", "gemini"),
        ("claude-sonnet-4-6", "anthropic"),
        ("claude-haiku-4-5", "anthropic"),
        # ("qwen2.5:1.5b", "ollama"),          # last resort only
    ],
    # Bounded reasoning: research, planning, claim synthesis. Same quality ladder;
    # local qwen is the LAST resort, not the second option (a 1.5B model plans/
    # synthesises poorly — see the 429-fallback lesson).
    "light": [
        ("qwen2.5:1.5b", "ollama"),
        ("gemini-2.5-flash", "gemini"),
        ("gemini-2.5-flash-lite", "gemini"),
        ("claude-haiku-4-5", "anthropic"),
                  # last resort only
    ],
    # Vision / multimodal: MUST be a model that can see images — NEVER qwen
    # (the local 1.5B model is text-only). Falls back across hosted multimodal.
    "vision": [
        ("gemini-2.5-flash", "gemini"),
        ("gemini-2.5-flash-lite", "gemini"),
        ("claude-sonnet-4-6", "anthropic"),
        ("claude-haiku-4-5", "anthropic"),
        # NOTE: no local vision — small local VLMs (moondream) don't fit a low-RAM
        # CPU box and read charts poorly. Vision routes to hosted multimodal only;
        # if none is reachable the vision agent degrades to "unavailable" (non-fatal).
    ],
        # Critic / evaluator: LOCAL-FIRST by design — decorrelated errors + $0 cost.
    # (Robustness against the small model's noise is handled in critic.py via a
    # deterministic lexical-groundedness signal combined with this LLM verdict.)
    "critic": [
        ("qwen2.5:1.5b", "ollama"),
        ("gemini-2.5-flash-lite", "gemini"),
        ("gemini-2.5-flash", "gemini"),
        ("claude-haiku-4-5", "anthropic"),
    ],
}


@dataclass(frozen=True)
class ModelSelection:
    """The result of running the selection protocol for one role."""
    role: str
    name: str
    provider: str

    def to_langchain(self, temperature: float = 0.3) -> BaseChatModel:
        if self.provider == "gemini":
            if not settings.gemini_api_key:
                raise KeyError("GEMINI_API_KEY")
            return ChatGoogleGenerativeAI(
                model=self.name,
                google_api_key=settings.gemini_api_key,
                temperature=temperature,
            )
        if self.provider == "ollama":
            return ChatOllama(
                model=self.name,
                base_url=settings.ollama_host,
                temperature=temperature,
            )
        if self.provider == "anthropic":
            if not settings.anthropic_api_key:
                raise KeyError("ANTHROPIC_API_KEY")
            # Lazy import so users without langchain-anthropic can still run the chain.
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.name,
                api_key=settings.anthropic_api_key,
                temperature=temperature,
            )
        raise ValueError(f"Unknown provider: {self.provider}")


# ---------------------------------------------------------------------------
# The protocol itself.
# ---------------------------------------------------------------------------

def _health_check(model: BaseChatModel, label: str) -> bool:
    """Send a trivial 'hi' and confirm we get a non-empty response."""
    try:
        response = model.invoke([HumanMessage(content="hi")])
        ok = bool(getattr(response, "content", "").strip())
        print(f"  [health] {label}: {'OK' if ok else 'EMPTY'}")
        return ok
    except Exception as e:
        print(f"  [health] {label}: FAIL ({type(e).__name__}: {str(e)[:100]})")
        return False


def select_model_for_role(role: str) -> ModelSelection:
    """Try each option in the role's preference chain; return the first that responds."""
    if role not in ROLE_PREFERENCES:
        raise ValueError(f"Unknown role '{role}'. Known: {list(ROLE_PREFERENCES)}")

    print(f"\n[selection] role={role}")
    for name, provider in ROLE_PREFERENCES[role]:
        candidate = ModelSelection(role=role, name=name, provider=provider)
        try:
            model = candidate.to_langchain(temperature=0)
        except KeyError as e:
            print(f"  [build]  {provider}:{name}: SKIP (missing env var {e})")
            continue
        except ImportError as e:
            print(f"  [build]  {provider}:{name}: SKIP (missing package: {e.name})")
            continue
        except Exception as e:
            print(f"  [build]  {provider}:{name}: FAIL ({type(e).__name__}: {str(e)[:80]})")
            continue
        if _health_check(model, f"{provider}:{name}"):
            print(f"  -> selected {provider}:{name}")
            return candidate

    raise RuntimeError(
        f"No working provider for role={role}. Tried: {ROLE_PREFERENCES[role]}. "
        f"Check API keys, quota, and Ollama service status."
    )


def select_all_models(roles: list[str]) -> dict[str, ModelSelection]:
    """Run the protocol for multiple roles. Call once at startup; fail fast."""
    print("=" * 60)
    print("MODEL SELECTION PROTOCOL")
    print("=" * 60)
    selections = {role: select_model_for_role(role) for role in roles}
    print("=" * 60)
    print("BOUND MODELS:")
    for role, sel in selections.items():
        print(f"  {role:8s} -> {sel.provider}:{sel.name}")
    print("=" * 60)
    return selections
