"""LangGraph state machine (async): planner -> research -> critic -> summariser,
with a bounded re-plan loop. M2.1 patterns: typed state, MemorySaver, draw_mermaid.

build_graph(models, tools, model_names) injects everything, so the wiring, the
re-plan loop, and the MAX_REPLANS breaker are testable with NO keys and NO network
(fake models + fake tools). models=None runs the real provider selection.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from core.agents.critic import make_critic_node
from core.agents.planner import make_planner_node
from core.agents.researcher import make_researcher_node
from core.agents.summariser import make_summariser_node
from core.schemas import ResearchState


def _real_models():
    from core.byok import select_all_models
    sels = select_all_models(["light", "critic", "heavy"])
    models = {r: sels[r].to_langchain(temperature=0) for r in ("light", "critic", "heavy")}
    names = {r: sels[r].name for r in ("light", "critic", "heavy")}
    return models, names


def _real_tools():
    from core.tools.vector_store import rag_retrieve
    from core.tools.web_search import web_search
    return {"web": web_search, "rag": rag_retrieve}


def route_after_critic(state: ResearchState) -> str:
    report = state["critic_report"]
    return "replan" if (report and report.needs_replan) else "summarise"


def build_graph(models: dict | None = None, tools: dict | None = None,
                model_names: dict | None = None):
    if models is None:
        models, auto_names = _real_models()
        model_names = model_names or auto_names
    model_names = model_names or {"light": "light", "critic": "critic", "heavy": "heavy"}
    tools = tools if tools is not None else _real_tools()

    g = StateGraph(ResearchState)
    g.add_node("planner", make_planner_node(models["light"], model_names["light"]))
    g.add_node("research", make_researcher_node(models["light"], tools, model_names["light"]))
    g.add_node("critic", make_critic_node(models["critic"], model_names["critic"]))
    g.add_node("summariser", make_summariser_node(models["heavy"], model_names["heavy"]))

    g.add_edge(START, "planner")
    g.add_edge("planner", "research")
    g.add_edge("research", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"replan": "planner", "summarise": "summariser"})
    g.add_edge("summariser", END)
    return g.compile(checkpointer=MemorySaver())


def initial_state(question: str, image_path: str | None = None) -> dict:
    return {"question": question, "image_path": image_path, "subtasks": [],
            "findings": [], "critic_report": None, "replan_count": 0,
            "final": None, "trace": []}


def export_diagram(path: str = "docs/diagram.mmd") -> None:
    app = build_graph(models={"light": None, "critic": None, "heavy": None}, tools={})
    src = app.get_graph().draw_mermaid()
    with open(path, "w") as f:
        f.write(src)
    print(f"Mermaid written to {path}\n")
    print(src)
