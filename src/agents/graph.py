from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.nodes import (
    analysis_node,
    comparison_node,
    output_node,
    planner_node,
    qa_node,
    report_node,
    retrieval_node,
    route_by_intent,
)
from src.agents.state import AgentState
from src.core.logging import get_logger

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("qa", qa_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("comparison", comparison_node)
    graph.add_node("report", report_node)
    graph.add_node("output", output_node)

    # Flow: start → planner → retrieval → [conditional] → output → END
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieval")

    graph.add_conditional_edges(
        "retrieval",
        route_by_intent,
        {
            "qa": "qa",
            "analysis": "analysis",
            "comparison": "comparison",
            "report": "report",
        },
    )

    graph.add_edge("qa", "output")
    graph.add_edge("analysis", "output")
    graph.add_edge("comparison", "output")
    graph.add_edge("report", "output")
    graph.add_edge("output", END)

    return graph


def run_agent(query: str) -> dict:
    """Run the full agent graph for a given query. Returns final state."""
    graph = build_graph()
    app = graph.compile()

    initial_state: AgentState = {
        "query": query,
        "plan": "",
        "retrieved_chunks": [],
        "analysis": "",
        "answer": "",
        "sources": [],
        "agent_trace": [],
        "intent": "unknown",
        "error": "",
        "iteration": 0,
    }

    logger.info(f"Running agent for query: {query[:80]}...")
    final_state = app.invoke(initial_state)
    logger.info("Agent run complete")
    return final_state
