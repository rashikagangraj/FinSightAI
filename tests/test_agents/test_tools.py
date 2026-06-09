from __future__ import annotations

from src.agents.tools import (
    calculate_financial_ratio,
    compare_companies,
    extract_key_metrics,
)


def test_calculate_ratio_normal():
    result = calculate_financial_ratio(97.0e9, 383.3e9, "Net Profit Margin")
    assert result["ratio_name"] == "Net Profit Margin"
    assert abs(result["value"] - 0.2531) < 0.001


def test_calculate_ratio_division_by_zero():
    result = calculate_financial_ratio(100.0, 0.0, "P/E")
    assert result["value"] is None
    assert "Division by zero" in result["error"]


def test_extract_key_metrics_revenue():
    text = "Total revenue was $383.3 billion for fiscal 2023."
    metrics = extract_key_metrics(text)
    assert "revenue" in metrics
    assert abs(metrics["revenue"] - 383.3e9) < 1e9


def test_extract_key_metrics_eps():
    text = "Diluted earnings per share (EPS): $6.13 for fiscal year 2023."
    metrics = extract_key_metrics(text)
    assert "eps" in metrics
    assert abs(metrics["eps"] - 6.13) < 0.01


def test_extract_key_metrics_gross_margin():
    text = "Gross margin was 44.1% for the period."
    metrics = extract_key_metrics(text)
    assert "gross_margin" in metrics
    assert abs(metrics["gross_margin"] - 44.1) < 0.1


def test_compare_companies_calculates_diff():
    data_a = {"revenue": 383.3e9, "gross_margin": 44.1}
    data_b = {"revenue": 97.7e9, "gross_margin": 17.6}
    result = compare_companies(data_a, data_b, "Apple", "Tesla")

    assert result["companies"] == ["Apple", "Tesla"]
    rev = result["metrics"]["revenue"]
    assert rev["Apple"] == 383.3e9
    assert rev["leader"] == "Apple"
    assert rev["diff_pct"] is not None


def test_compare_companies_missing_metric():
    data_a = {"revenue": 100.0}
    data_b = {}
    result = compare_companies(data_a, data_b, "Co A", "Co B")
    assert result["metrics"]["revenue"]["Co B"] is None
