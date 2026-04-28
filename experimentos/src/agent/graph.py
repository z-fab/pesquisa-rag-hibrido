import time

from langgraph.graph import END, StateGraph

from src.agent.ablation import AblationMode
from src.agent.nodes.consolidator import consolidator_node
from src.agent.nodes.consolidator_lite import consolidator_lite_node
from src.agent.nodes.planner import planner_node
from src.agent.nodes.simple_router import simple_router_node
from src.agent.nodes.simple_synthesizer import simple_synthesizer_node
from src.agent.nodes.sql_planner_executor import sql_planner_executor_node
from src.agent.nodes.synthesizer import synthesizer_node
from src.agent.nodes.text_retriever import text_retriever_node
from src.agent.nodes.verifier import verifier_node
from src.agent.state import AgentState
from src.config.settings import SETTINGS


def route_by_plan(state: AgentState) -> str | list[str]:
    """Routes based on planner output."""
    if not state.get("needs_retrieval", True):
        return "synthesizer"

    searches = state["planner_output"].searches
    has_sql = any(t.type == "sql" for t in searches)
    has_text = any(t.type == "text" for t in searches)

    if has_sql and has_text:
        return ["sql_planner_executor", "text_retriever"]
    elif has_sql:
        return "sql_planner_executor"
    elif has_text:
        return "text_retriever"
    else:
        return "synthesizer"


def route_after_verification(state: AgentState) -> str:
    """Routes based on verifier output."""
    verifier_out = state.get("verifier_output")
    if not verifier_out or verifier_out.overall_pass:
        return "pass"
    if state.get("retry_count", 0) >= SETTINGS.VERIFIER_MAX_RETRIES:
        return "max_retries"
    return "retry"


def _build_full_graph() -> StateGraph:
    """Builds the full pipeline: Planner -> SQL/Text -> Synthesizer -> Verifier -> Consolidator."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("text_retriever", text_retriever_node)
    graph.add_node("sql_planner_executor", sql_planner_executor_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("consolidator", consolidator_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_by_plan,
        ["sql_planner_executor", "text_retriever", "synthesizer"],
    )

    graph.add_edge("text_retriever", "synthesizer")
    graph.add_edge("sql_planner_executor", "synthesizer")
    graph.add_edge("synthesizer", "verifier")

    graph.add_conditional_edges(
        "verifier",
        route_after_verification,
        {
            "pass": "consolidator",
            "retry": "planner",
            "max_retries": "consolidator",
        },
    )

    graph.add_edge("consolidator", END)
    return graph


def _build_no_verifier_graph() -> StateGraph:
    """Builds pipeline without verifier: Planner -> SQL/Text -> Synthesizer -> Consolidator."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("text_retriever", text_retriever_node)
    graph.add_node("sql_planner_executor", sql_planner_executor_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("consolidator", consolidator_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_by_plan,
        ["sql_planner_executor", "text_retriever", "synthesizer"],
    )

    graph.add_edge("text_retriever", "synthesizer")
    graph.add_edge("sql_planner_executor", "synthesizer")
    graph.add_edge("synthesizer", "consolidator")
    graph.add_edge("consolidator", END)
    return graph


def _build_no_synthesizer_graph() -> StateGraph:
    """Builds pipeline without synthesizer: Planner -> SQL/Text -> ConsolidatorLite."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("text_retriever", text_retriever_node)
    graph.add_node("sql_planner_executor", sql_planner_executor_node)
    graph.add_node("consolidator_lite", consolidator_lite_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_by_plan,
        ["sql_planner_executor", "text_retriever"],
    )

    graph.add_edge("text_retriever", "consolidator_lite")
    graph.add_edge("sql_planner_executor", "consolidator_lite")
    graph.add_edge("consolidator_lite", END)
    return graph


def _build_poc_graph() -> StateGraph:
    """Builds POC pipeline: SimpleRouter -> SQL/Text -> SimpleSynthesizer."""
    graph = StateGraph(AgentState)

    graph.add_node("simple_router", simple_router_node)
    graph.add_node("text_retriever", text_retriever_node)
    graph.add_node("sql_planner_executor", sql_planner_executor_node)
    graph.add_node("simple_synthesizer", simple_synthesizer_node)

    graph.set_entry_point("simple_router")

    graph.add_conditional_edges(
        "simple_router",
        route_by_plan,
        ["sql_planner_executor", "text_retriever"],
    )

    graph.add_edge("text_retriever", "simple_synthesizer")
    graph.add_edge("sql_planner_executor", "simple_synthesizer")
    graph.add_edge("simple_synthesizer", END)
    return graph


_GRAPH_BUILDERS = {
    AblationMode.FULL: _build_full_graph,
    AblationMode.NO_VERIFIER: _build_no_verifier_graph,
    AblationMode.NO_SYNTHESIZER: _build_no_synthesizer_graph,
    AblationMode.POC: _build_poc_graph,
}


def build_graph(mode: AblationMode = AblationMode.FULL) -> StateGraph:
    """Builds and returns the compiled LangGraph for the given ablation mode."""
    builder = _GRAPH_BUILDERS[mode]
    return builder().compile()


def run_graph(question: str, mode: AblationMode = AblationMode.FULL) -> dict:
    """Runs the agent graph with a question and returns the final state."""
    compiled = build_graph(mode)
    initial_state: AgentState = {
        "question": question,
        "trace": [],
        "executed_agents": [],
        "token_usage": {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0},
        "total_start": time.perf_counter(),
        "retry_count": 0,
    }
    return compiled.invoke(initial_state)


def get_graph_mermaid(mode: AblationMode = AblationMode.FULL) -> str:
    """Returns the mermaid diagram of the graph."""
    compiled = build_graph(mode)
    return compiled.get_graph().draw_mermaid()
