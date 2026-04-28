import time

from agent.nodes.rag import RAGAgent
from agent.nodes.router import RouterAgent
from agent.nodes.sql import SQLAgent
from agent.nodes.synthesizer import SynthesizerAgent
from agent.state import AgentState
from langgraph.graph import END, StateGraph


def route_logic(state):
    d = state["router_decision"]
    if d == "structured":
        return "sql_agent"
    elif d == "non_structured":
        return "rag_agent"
    else:
        return ("sql_agent", "rag_agent")


class Graph:
    def __init__(self, state: AgentState, start_state: AgentState) -> None:
        self.start_state = start_state

        graph = StateGraph(state)

        graph.add_node("router", RouterAgent())
        graph.add_node("sql_agent", SQLAgent())
        graph.add_node("rag_agent", RAGAgent())
        graph.add_node("synthesizer", SynthesizerAgent())

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            route_logic,
            ["sql_agent", "rag_agent"],
        )
        graph.add_edge("sql_agent", "synthesizer")
        graph.add_edge("rag_agent", "synthesizer")
        graph.add_edge("synthesizer", END)

        self.graph = graph.compile()

    def run(self):
        runtime_state = dict(self.start_state)
        runtime_state.setdefault("trace", [])
        runtime_state.setdefault("executed_agents", [])
        runtime_state.setdefault(
            "token_usage",
            {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0},
        )
        runtime_state.setdefault("total_start", time.perf_counter())
        return self.graph.invoke(runtime_state)

    def print_graph(self):
        return self.graph.get_graph().draw_mermaid()
