"""LangGraph state machine: planner -> research -> critic -> summariser, with a
bounded re-plan loop. M2.1 patterns: typed state, MemorySaver, draw_mermaid.

build_graph(models=None) injects the bound LLMs. With models=None it runs the
provider selection protocol (real run). Tests pass fake models, so the wiring,
the re-plan loop, and the MAX_REPLANS breaker are verifiable with NO API keys.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from core.agents.critic import make_critic_node
from core.agents.planner import make_planner_node
from core.agents.researcher import make_researcher_node
from core.agents.summariser import make_summariser_node
from core.schemas import ResearchState


def _build_models() -> dict:
    """Real path: run the selection protocol and bind one model per role."""
    from core.byok import select_all_models  # lazy: keeps this module importable w/o providers
    sels = select_all_models(["light", "critic", "heavy"])
    return {role: sels[role].to_langchain(temperature=0) for role in ("light", "critic", "heavy")}


def route_after_critic(state: ResearchState) -> str:
    report = state["critic_report"]
    return "replan" if (report and report.needs_replan) else "summarise"


def build_graph(models: dict | None = None):
    models = models or _build_models()

    g = StateGraph(ResearchState)
    g.add_node("planner", make_planner_node(models["light"]))
    g.add_node("research", make_researcher_node())
    g.add_node("critic", make_critic_node(models["critic"]))
    g.add_node("summariser", make_summariser_node(models["heavy"]))

    g.add_edge(START, "planner")
    g.add_edge("planner", "research")
    g.add_edge("research", "critic")
    g.add_conditional_edges("critic", route_after_critic, {
        "replan": "planner",
        "summarise": "summariser",
    })
    g.add_edge("summariser", END)

    return g.compile(checkpointer=MemorySaver())


def initial_state(question: str, image_path: str | None = None) -> dict:
    return {
        "question": question,
        "image_path": image_path,
        "subtasks": [],
        "findings": [],
        "critic_report": None,
        "replan_count": 0,
        "final": None,
        "trace": [],
    }


def export_diagram(path: str = "docs/diagram.mmd") -> None:
    """Write the compiled graph as Mermaid — this IS the architecture diagram."""
    app = build_graph(models={"light": None, "critic": None, "heavy": None})
    # node factories don't touch the model at build time, so None is fine here.
    src = app.get_graph().draw_mermaid()
    with open(path, "w") as f:
        f.write(src)
    print(f"Mermaid written to {path}\n")
    print(src)
