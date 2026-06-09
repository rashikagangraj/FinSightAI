from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agents.state import AgentState


def _base_state(**kwargs) -> AgentState:
    defaults: AgentState = {
        "query": "What was Apple revenue?",
        "plan": "",
        "retrieved_chunks": [],
        "analysis": "",
        "answer": "",
        "sources": [],
        "agent_trace": [],
        "intent": "simple_qa",
        "error": "",
        "iteration": 0,
    }
    defaults.update(kwargs)
    return defaults


@patch("src.agents.nodes.get_llm_client")
def test_planner_node_returns_intent(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"intent": "simple_qa", "search_queries": ["Apple revenue 2023"]}'
    mock_get_llm.return_value = mock_llm

    from src.agents.nodes import planner_node
    state = _base_state(query="What was Apple's revenue?")
    result = planner_node(state)

    assert result["intent"] == "simple_qa"
    assert "[planner]" in result["agent_trace"][0]


@patch("src.agents.nodes.get_llm_client")
def test_planner_node_handles_bad_json(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "not valid json at all"
    mock_get_llm.return_value = mock_llm

    from src.agents.nodes import planner_node
    state = _base_state()
    result = planner_node(state)
    # Should default gracefully
    assert result["intent"] in {"simple_qa", "deep_analysis", "comparison", "report", "unknown"}


@patch("src.agents.nodes.get_llm_client")
def test_qa_node_with_empty_chunks_returns_no_info(mock_get_llm):
    mock_get_llm.return_value = MagicMock()
    from src.agents.nodes import qa_node

    state = _base_state(retrieved_chunks=[])
    result = qa_node(state)
    assert "could not find" in result["answer"].lower() or "no" in result["answer"].lower()


@patch("src.agents.nodes.get_llm_client")
def test_qa_node_uses_context(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Apple's revenue was $383.3 billion."
    mock_get_llm.return_value = mock_llm

    from src.agents.nodes import qa_node

    chunks = [{"text": "Apple revenue: $383.3 billion", "source": "apple.txt", "score": 0.9, "metadata": {}}]
    state = _base_state(retrieved_chunks=chunks)
    result = qa_node(state)

    assert "383.3 billion" in result["answer"]
    assert "[qa]" in result["agent_trace"][0]


def test_output_node_appends_sources():
    from src.agents.nodes import output_node

    state = _base_state(
        answer="Revenue was $100M.",
        sources=["docs/apple.txt", "docs/tesla.txt"],
    )
    result = output_node(state)
    assert "apple.txt" in result["answer"] or "Sources" in result["answer"]
