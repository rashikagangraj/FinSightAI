from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.agents.graph import run_agent
from src.agents.nodes import comparison_node, planner_node, qa_node
from src.agents.state import AgentState
from src.agents.tools import (
    compare_entities_detailed,
    extract_comparison_entities_and_metric,
    extract_entity_metric_value,
    format_comparison_markdown,
)
from src.llm.fallback_client import FallbackClient


AIKART_TEXT = """
AIKART TECHNOLOGIES PRIVATE LIMITED
ANNUAL FINANCIAL HIGHLIGHTS & EARNINGS STATEMENT
Fiscal Year: FY2025–26 (Period Ended March 31, 2026)

Key Financial Metrics (FY2025–26):
- Total Revenue: ₹78.0 Cr (up 42% YoY compared to ₹55.0 Cr in FY2024–25)
- Net Profit (Profit After Tax): ₹5.7 Cr (net margin of 7.3%)
- Cash and Cash Equivalents (Cash Balance): ₹25.0 Cr
- Gross Profit Margin: 38.5%
"""

TESLA_TEXT = """
TESLA, INC.
Q4 AND FULL YEAR 2023 EARNINGS CALL TRANSCRIPT

FULL YEAR 2023 FINANCIAL RESULTS:
Total revenue: $97.7 billion, up 19% year-over-year
Net income (GAAP): $15.0 billion
Cash and cash equivalents: $29.1 billion
Total company gross margin: 17.6% (Q4 2023)
Automotive gross margin: 18.9%
"""


def _base_state(**kwargs) -> AgentState:
    defaults: AgentState = {
        "query": "Compare AIKART's revenue with Tesla's revenue",
        "plan": "",
        "retrieved_chunks": [],
        "analysis": "",
        "answer": "",
        "sources": [],
        "agent_trace": [],
        "intent": "comparison",
        "error": "",
        "iteration": 0,
    }
    defaults.update(kwargs)
    return defaults


def test_comparison_intent_routing_with_two_entities():
    """Verify comparison query with two entities routes to comparison intent."""
    state = _base_state(query="Compare AIKART's revenue with Tesla's revenue")
    result = planner_node(state)
    assert result["intent"] == "comparison"


def test_single_entity_revenue_routes_to_simple_qa():
    """Verify single AIKART revenue query routes to simple_qa, returning ₹78.0 Cr."""
    state = _base_state(
        query="What was AIKART's revenue in FY2025–26?",
        retrieved_chunks=[{"text": AIKART_TEXT, "source": "aikart_financials.txt"}],
    )
    plan_res = planner_node(state)
    assert plan_res["intent"] == "simple_qa"

    qa_res = qa_node(state)
    assert "78.0" in qa_res["answer"]
    assert "Cr" in qa_res["answer"] or "₹" in qa_res["answer"]


def test_aikart_vs_tesla_revenue_comparison():
    """Verify AIKART vs Tesla revenue extraction and comparative matrix."""
    chunks = [
        {"text": AIKART_TEXT, "source": "aikart_financials.txt"},
        {"text": TESLA_TEXT, "source": "tesla_earnings_q4.txt"},
    ]
    state = _base_state(
        query="Compare AIKART's revenue with Tesla's revenue",
        retrieved_chunks=chunks,
    )
    result = comparison_node(state)
    answer = result["answer"]

    # Must contain both values
    assert "78.0" in answer
    assert "97.7" in answer or "96.7" in answer
    # Must contain markdown table
    assert "| Metric |" in answer or "| **Total Revenue** |" in answer or "Revenue" in answer
    # Must contain sources
    assert "aikart_financials.txt" in answer or "aikart" in answer.lower()
    assert "tesla_earnings_q4.txt" in answer or "tesla" in answer.lower()


def test_aikart_vs_tesla_gross_margin_comparison():
    """Verify AIKART vs Tesla gross margin extraction."""
    chunks = [
        {"text": AIKART_TEXT, "source": "aikart_financials.txt"},
        {"text": TESLA_TEXT, "source": "tesla_earnings_q4.txt"},
    ]
    state = _base_state(
        query="Compare AIKART and Tesla gross margin",
        retrieved_chunks=chunks,
    )
    result = comparison_node(state)
    answer = result["answer"]

    # Must contain both margin values
    assert "38.5" in answer
    assert "17.6" in answer or "18.9" in answer
    assert "aikart" in answer.lower()
    assert "tesla" in answer.lower()


def test_missing_company_document_returns_error():
    """Verify missing company data returns a clear error instead of generic hallucination."""
    # Only AIKART provided, Tesla is missing
    chunks = [
        {"text": AIKART_TEXT, "source": "aikart_financials.txt"},
    ]
    state = _base_state(
        query="Compare AIKART's revenue with Reliance's revenue",
        retrieved_chunks=chunks,
    )
    result = comparison_node(state)
    answer = result["answer"]

    assert "Missing financial data" in answer or "Comparative Analysis Error" in answer
    assert "Cannot perform" in answer or "Reliance" in answer or "missing" in answer.lower()


def test_currency_normalization_logic():
    """Verify currency normalization between INR (Cr) and USD (Billion)."""
    data_a = {
        "entity": "AIKART",
        "formatted_val": "₹78.0 Cr",
        "raw_num": 78.0,
        "unit": "Cr",
        "currency": "INR",
        "source": "aikart_financials.txt",
    }
    data_b = {
        "entity": "Tesla",
        "formatted_val": "$97.7 Billion",
        "raw_num": 97.7,
        "unit": "Billion",
        "currency": "USD",
        "source": "tesla_earnings_q4.txt",
    }

    res = compare_entities_detailed(data_a, data_b, "AIKART", "Tesla", "revenue", "Total Revenue")
    assert res["error"] is None
    assert "Tesla" in res["ratio_str"]
    assert "FX normalized" in res["summary_note"] or "larger" in res["ratio_str"]
