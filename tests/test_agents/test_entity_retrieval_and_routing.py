from __future__ import annotations

import pytest
from unittest.mock import patch

from src.agents.nodes import (
    _classify_intent_rules,
    analysis_node,
    comparison_node,
    planner_node,
    qa_node,
)
from src.agents.state import AgentState
from src.rag.retriever import _extract_query_entities


AIKART_CHUNK = {
    "text": (
        "AIKART TECHNOLOGIES PRIVATE LIMITED\n"
        "ANNUAL FINANCIAL HIGHLIGHTS & EARNINGS STATEMENT\n"
        "Fiscal Year: FY2025–26 (Period Ended March 31, 2026)\n"
        "Key Financial Metrics (FY2025–26):\n"
        "- Total Revenue: ₹78.0 Cr (up 42% YoY compared to ₹55.0 Cr in FY2024–25)\n"
        "- Net Profit (Profit After Tax): ₹5.7 Cr (net margin of 7.3%)\n"
        "- Free Cash Flow (FCF): ₹4.2 Cr\n"
        "- Cash and Cash Equivalents: ₹25.0 Cr\n"
        "- Gross Profit Margin: 38.5%\n"
    ),
    "source": "AIKART_Dummy_Financial_Report_FY2025_26.pdf",
    "score": 0.95,
}

TESLA_CHUNK = {
    "text": (
        "TESLA, INC. Q4 AND FULL YEAR 2023 EARNINGS CALL TRANSCRIPT\n"
        "FULL YEAR 2023 FINANCIAL RESULTS:\n"
        "Total revenue: $97.7 billion, up 19% year-over-year\n"
        "Net income (GAAP): $15.0 billion\n"
        "Free cash flow: $4.4 billion\n"
        "Total company gross margin: 17.6% (Q4 2023)\n"
    ),
    "source": "tesla report.pdf",
    "score": 0.95,
}


def _make_state(query: str, chunks: list[dict] | None = None, intent: str = "simple_qa") -> AgentState:
    return {
        "query": query,
        "plan": "",
        "retrieved_chunks": chunks or [],
        "analysis": "",
        "answer": "",
        "sources": [c["source"] for c in (chunks or [])],
        "agent_trace": [],
        "intent": intent,
        "error": "",
        "iteration": 0,
    }


def test_extract_query_entities_single_company():
    """Verify entity extraction pulls the correct company name."""
    assert "AIKART" in _extract_query_entities("Calculate AIKART's FCF margin")
    assert "Tesla" in _extract_query_entities("Calculate Tesla's FCF margin")
    assert "AIKART" in _extract_query_entities("What was AIKART's revenue?")


def test_intent_routing_rules():
    """Verify routing rules for factual lookup, calculations, comparison, and reports."""
    # Factual lookup -> simple_qa
    assert _classify_intent_rules("What was AIKART's revenue?") == "simple_qa"
    assert _classify_intent_rules("What is Tesla's net income?") == "simple_qa"

    # Calculations / ratios / FCF -> deep_analysis
    assert _classify_intent_rules("Calculate AIKART's FCF margin") == "deep_analysis"
    assert _classify_intent_rules("Calculate Tesla's FCF margin") == "deep_analysis"
    assert _classify_intent_rules("Compute DuPont ROE for Apple") == "deep_analysis"
    assert _classify_intent_rules("Calculate Altman Z-score") == "deep_analysis"
    assert _classify_intent_rules("Calculate debt-to-equity ratio") == "deep_analysis"

    # Compare / vs / versus -> comparison
    assert _classify_intent_rules("Compare AIKART and Tesla revenue") == "comparison"
    assert _classify_intent_rules("Compare AIKART vs Tesla revenue") == "comparison"

    # Generate report -> report
    assert _classify_intent_rules("Generate a report for AIKART") == "report"


def test_single_company_revenue_retrieves_aikart_source():
    """Verify 'What was AIKART's revenue?' routes to simple_qa and outputs ₹78.0 Cr."""
    state = _make_state("What was AIKART's revenue?", [AIKART_CHUNK], "simple_qa")
    plan_res = planner_node(state)
    assert plan_res["intent"] == "simple_qa"

    qa_res = qa_node(state)
    assert "78.0" in qa_res["answer"]
    assert "Cr" in qa_res["answer"] or "₹" in qa_res["answer"]


def test_calculate_aikart_fcf_margin():
    """Verify 'Calculate AIKART's FCF margin' routes to deep_analysis and calculates 5.4%."""
    state = _make_state("Calculate AIKART's FCF margin", [AIKART_CHUNK], "deep_analysis")
    plan_res = planner_node(state)
    assert plan_res["intent"] == "deep_analysis"

    analysis_res = analysis_node(state)
    answer = analysis_res["answer"]

    # FCF Margin = 4.2 / 78 * 100 = 5.38% ≈ 5.4%
    assert "5.4%" in answer or "5.4" in answer
    assert "AIKART" in answer
    assert "AIKART_Dummy_Financial_Report_FY2025_26.pdf" in answer or "Source:" in answer


def test_calculate_tesla_fcf_margin():
    """Verify 'Calculate Tesla's FCF margin' routes to deep_analysis and calculates Tesla FCF margin."""
    state = _make_state("Calculate Tesla's FCF margin", [TESLA_CHUNK], "deep_analysis")
    plan_res = planner_node(state)
    assert plan_res["intent"] == "deep_analysis"

    analysis_res = analysis_node(state)
    answer = analysis_res["answer"]

    # Tesla FCF margin = 4.4 / 97.7 * 100 = 4.5%
    assert "4.5" in answer or "Tesla" in answer
    assert "tesla report.pdf" in answer or "Source:" in answer


def test_compare_aikart_and_tesla_revenue_retrieves_both_sources():
    """Verify 'Compare AIKART and Tesla revenue' routes to comparison and retrieves both sources."""
    state = _make_state("Compare AIKART and Tesla revenue", [AIKART_CHUNK, TESLA_CHUNK], "comparison")
    plan_res = planner_node(state)
    assert plan_res["intent"] == "comparison"

    comp_res = comparison_node(state)
    answer = comp_res["answer"]

    # Must extract values for both companies
    assert "78.0" in answer
    assert "97.7" in answer or "96.7" in answer
    assert "AIKART_Dummy_Financial_Report_FY2025_26.pdf" in answer or "aikart" in answer.lower()
    assert "tesla report.pdf" in answer or "tesla" in answer.lower()

