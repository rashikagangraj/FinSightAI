from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.agents.graph import run_agent
from src.agents.nodes import planner_node, qa_node
from src.agents.state import AgentState
from src.llm.fallback_client import FallbackClient


def _base_state(**kwargs) -> AgentState:
    defaults: AgentState = {
        "query": "What was AIKART's revenue in FY2025–26?",
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


AIKART_SAMPLE_TEXT = """
AIKART TECHNOLOGIES PRIVATE LIMITED
ANNUAL FINANCIAL HIGHLIGHTS & EARNINGS STATEMENT
Fiscal Year: FY2025–26 (Period Ended March 31, 2026)

Key Financial Metrics (FY2025–26):
- Total Revenue: ₹78.0 Cr (up 42% YoY compared to ₹55.0 Cr in FY2024–25)
- Net Profit (Profit After Tax): ₹5.7 Cr (net margin of 7.3%)
- Cash and Cash Equivalents (Cash Balance): ₹25.0 Cr
- Operating Income (EBITDA): ₹9.2 Cr
- Gross Profit Margin: 38.5%
- Total Assets: ₹112.0 Cr
- Total Debt: ₹14.0 Cr
- Shareholders' Equity: ₹68.0 Cr
"""


def test_aikart_revenue_query_classification():
    """Verify single-company factual query routes to simple_qa."""
    state = _base_state(query="What was AIKART's revenue in FY2025–26?")
    result = planner_node(state)
    assert result["intent"] == "simple_qa", f"Expected simple_qa, got {result['intent']}"


def test_aikart_net_profit_query_classification():
    """Verify net profit factual query routes to simple_qa."""
    state = _base_state(query="What was AIKART's net profit?")
    result = planner_node(state)
    assert result["intent"] == "simple_qa", f"Expected simple_qa, got {result['intent']}"


def test_aikart_cash_balance_query_classification():
    """Verify cash balance query routes to simple_qa."""
    state = _base_state(query="What was AIKART's cash balance?")
    result = planner_node(state)
    assert result["intent"] == "simple_qa", f"Expected simple_qa, got {result['intent']}"


def test_compare_aikart_tesla_query_classification():
    """Verify two-company comparison query routes to comparison."""
    state = _base_state(query="Compare AIKART and Tesla revenue")
    result = planner_node(state)
    assert result["intent"] == "comparison", f"Expected comparison, got {result['intent']}"


def test_qa_node_extracts_aikart_revenue():
    """Verify QA node extracts exact ₹78.0 Cr revenue from context."""
    chunks = [
        {
            "text": AIKART_SAMPLE_TEXT,
            "source": "aikart_financials.txt",
            "score": 0.95,
            "metadata": {"source": "aikart_financials.txt"},
        }
    ]
    state = _base_state(
        query="What was AIKART's revenue in FY2025–26?",
        retrieved_chunks=chunks,
    )
    result = qa_node(state)
    assert "78.0" in result["answer"]
    assert "Cr" in result["answer"] or "₹" in result["answer"]


def test_qa_node_extracts_aikart_net_profit():
    """Verify QA node extracts exact ₹5.7 Cr net profit from context."""
    chunks = [
        {
            "text": AIKART_SAMPLE_TEXT,
            "source": "aikart_financials.txt",
            "score": 0.95,
            "metadata": {"source": "aikart_financials.txt"},
        }
    ]
    state = _base_state(
        query="What was AIKART's net profit?",
        retrieved_chunks=chunks,
    )
    result = qa_node(state)
    assert "5.7" in result["answer"]
    assert "Cr" in result["answer"] or "₹" in result["answer"]


def test_qa_node_extracts_aikart_cash_balance():
    """Verify QA node extracts exact ₹25.0 Cr cash balance from context."""
    chunks = [
        {
            "text": AIKART_SAMPLE_TEXT,
            "source": "aikart_financials.txt",
            "score": 0.95,
            "metadata": {"source": "aikart_financials.txt"},
        }
    ]
    state = _base_state(
        query="What was AIKART's cash balance?",
        retrieved_chunks=chunks,
    )
    result = qa_node(state)
    assert "25.0" in result["answer"]
    assert "Cr" in result["answer"] or "₹" in result["answer"]


def test_fallback_client_direct_extraction():
    """Verify FallbackClient directly extracts exact numbers for AIKART queries."""
    client = FallbackClient()

    # Revenue
    rev_prompt = f"Context:\n{AIKART_SAMPLE_TEXT}\n\nQuestion: What was AIKART's revenue in FY2025–26?\n\nAnswer:"
    rev_ans = client.complete(rev_prompt)
    assert "78.0" in rev_ans
    assert "Cr" in rev_ans or "₹" in rev_ans

    # Profit
    prof_prompt = f"Context:\n{AIKART_SAMPLE_TEXT}\n\nQuestion: What was AIKART's net profit?\n\nAnswer:"
    prof_ans = client.complete(prof_prompt)
    assert "5.7" in prof_ans
    assert "Cr" in prof_ans or "₹" in prof_ans

    # Cash
    cash_prompt = f"Context:\n{AIKART_SAMPLE_TEXT}\n\nQuestion: What was AIKART's cash balance?\n\nAnswer:"
    cash_ans = client.complete(cash_prompt)
    assert "25.0" in cash_ans
    assert "Cr" in cash_ans or "₹" in cash_ans
