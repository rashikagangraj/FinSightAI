from __future__ import annotations

import re
from typing import Any

from src.core.logging import get_logger
from src.rag.retriever import RetrievedChunk, hybrid_search

logger = get_logger(__name__)


def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Search the indexed financial documents and return ranked chunks."""
    chunks: list[RetrievedChunk] = hybrid_search(query, top_k)
    return [
        {
            "text": c.text,
            "source": c.source,
            "score": c.score,
            "metadata": c.metadata,
        }
        for c in chunks
    ]


def calculate_financial_ratio(
    numerator: float,
    denominator: float,
    ratio_name: str = "ratio",
) -> dict:
    """Safely compute a financial ratio and return a labelled result."""
    if denominator == 0:
        return {"ratio_name": ratio_name, "value": None, "error": "Division by zero"}
    value = round(numerator / denominator, 4)
    return {"ratio_name": ratio_name, "value": value, "numerator": numerator, "denominator": denominator}


def extract_key_metrics(text: str) -> dict[str, Any]:
    """
    Extract common financial metrics from raw text using regex patterns.
    Returns a dict of metric_name → value (float or string).
    """
    metrics: dict[str, Any] = {}

    patterns = {
        "revenue": r"(?:revenue|net sales|total revenue)[^\d]*\$?([\d,\.]+)\s*(billion|million|B|M)?",
        "net_income": r"(?:net income|net earnings|net profit)[^\d]*\$?([\d,\.]+)\s*(billion|million|B|M)?",
        "eps": r"(?:earnings per share|EPS|diluted EPS)[^\d]*\$?([\d,\.]+)",
        "operating_income": r"(?:operating income|income from operations)[^\d]*\$?([\d,\.]+)\s*(billion|million|B|M)?",
        "gross_margin": r"(?:gross margin|gross profit margin)[^\d]*([\d,\.]+)\s*%",
        "total_assets": r"(?:total assets)[^\d]*\$?([\d,\.]+)\s*(billion|million|B|M)?",
        "total_debt": r"(?:total debt|long.term debt)[^\d]*\$?([\d,\.]+)\s*(billion|million|B|M)?",
        "cash": r"(?:cash and cash equivalents|cash \& equivalents)[^\d]*\$?([\d,\.]+)\s*(billion|million|B|M)?",
    }

    multipliers = {"billion": 1e9, "b": 1e9, "million": 1e6, "m": 1e6}

    for metric, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                raw = float(match.group(1).replace(",", ""))
                unit = (match.group(2) or "").lower().strip() if match.lastindex >= 2 else ""
                multiplier = multipliers.get(unit, 1.0)
                metrics[metric] = raw * multiplier
            except (ValueError, IndexError):
                pass

    return metrics


def compare_companies(
    data_a: dict[str, Any],
    data_b: dict[str, Any],
    company_a: str,
    company_b: str,
) -> dict:
    """
    Produce a side-by-side comparison of two companies' extracted metrics.
    data_a / data_b are outputs from extract_key_metrics.
    """
    all_keys = sorted(set(data_a.keys()) | set(data_b.keys()))
    comparison = {"companies": [company_a, company_b], "metrics": {}}

    for key in all_keys:
        val_a = data_a.get(key)
        val_b = data_b.get(key)

        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)) and val_b != 0:
            diff_pct = round((val_a - val_b) / abs(val_b) * 100, 2)
            leader = company_a if val_a > val_b else company_b
        else:
            diff_pct = None
            leader = None

        comparison["metrics"][key] = {
            company_a: val_a,
            company_b: val_b,
            "diff_pct": diff_pct,
            "leader": leader,
        }

    return comparison
