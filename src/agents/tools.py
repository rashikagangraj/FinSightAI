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
        "revenue": r"(?:revenue|net sales|total revenue)[^\d]*(?:₹|\$|rs\.?)?\s*([\d,\.]+)\s*(billion|million|crore|cr|lakh|b|m)?",
        "net_income": r"(?:net income|net profit|profit after tax|net earnings)[^\d]*(?:₹|\$|rs\.?)?\s*([\d,\.]+)\s*(billion|million|crore|cr|lakh|b|m)?",
        "eps": r"(?:earnings per share|eps|diluted eps)[^\d]*(?:₹|\$|rs\.?)?\s*([\d,\.]+)",
        "operating_income": r"(?:operating income|income from operations)[^\d]*(?:₹|\$|rs\.?)?\s*([\d,\.]+)\s*(billion|million|crore|cr|lakh|b|m)?",
        "ebitda": r"(?:ebitda|adjusted ebitda)[^\d]*(?:₹|\$|rs\.?)?\s*([\d,\.]+)\s*(billion|million|crore|cr|lakh|b|m)?",
        "free_cash_flow": r"(?:free cash flow|fcf)[^\d]*(?:₹|\$|rs\.?)?\s*([\d,\.]+)\s*(billion|million|crore|cr|lakh|b|m)?",
        "capex": r"(?:capital expenditures|capex|additions to property)[^\d]*(?:₹|\$|rs\.?)?\s*([\d,\.]+)\s*(billion|million|crore|cr|lakh|b|m)?",
        "rd_expense": r"(?:research and development|r&d|r \& d expense)[^\d]*(?:₹|\$|rs\.?)?\s*([\d,\.]+)\s*(billion|million|crore|cr|lakh|b|m)?",
        "gross_margin": r"(?:gross margin|gross profit margin)[^\d]*([\d,\.]+)\s*%",
        "operating_margin": r"(?:operating margin|operating profit margin)[^\d]*([\d,\.]+)\s*%",
        "total_assets": r"(?:total assets)[^\d]*(?:₹|\$|rs\.?)?\s*([\d,\.]+)\s*(billion|million|crore|cr|lakh|b|m)?",
        "total_debt": r"(?:total debt|long.term debt)[^\d]*(?:₹|\$|rs\.?)?\s*([\d,\.]+)\s*(billion|million|crore|cr|lakh|b|m)?",
        "shareholders_equity": r"(?:shareholders['’]? equity|stockholders['’]? equity|total equity)[^\d]*(?:₹|\$|rs\.?)?\s*([\d,\.]+)\s*(billion|million|crore|cr|lakh|b|m)?",
        "cash": r"(?:cash and cash equivalents|cash balance|cash \& equivalents)[^\d]*(?:₹|\$|rs\.?)?\s*([\d,\.]+)\s*(billion|million|crore|cr|lakh|b|m)?",
    }

    multipliers = {
        "billion": 1e9, "b": 1e9,
        "million": 1e6, "m": 1e6,
        "crore": 1e7, "cr": 1e7,
        "lakh": 1e5,
    }

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


def calculate_dupont_roe(
    net_income: float,
    revenue: float,
    total_assets: float,
    equity: float,
) -> dict[str, Any]:
    """
    3-Stage DuPont ROE Decomposition:
    ROE = (Net Margin) × (Asset Turnover) × (Financial Leverage)
    """
    if revenue == 0 or total_assets == 0 or equity == 0:
        return {"error": "Invalid denominator in DuPont calculation", "roe": None}

    net_margin = net_income / revenue
    asset_turnover = revenue / total_assets
    leverage_multiplier = total_assets / equity
    roe = net_margin * asset_turnover * leverage_multiplier

    return {
        "roe": round(roe * 100, 2),
        "net_margin_pct": round(net_margin * 100, 2),
        "asset_turnover": round(asset_turnover, 4),
        "leverage_multiplier": round(leverage_multiplier, 4),
        "interpretation": "High quality ROE" if net_margin > 0.15 else "Leverage-driven ROE",
    }


def calculate_altman_z_score(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    market_equity: float,
    total_liabilities: float,
    total_assets: float,
    sales: float,
) -> dict[str, Any]:
    """
    Altman Z-Score (Manufacturing & Corporate Health):
    Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 0.999*E
    """
    if total_assets == 0 or total_liabilities == 0:
        return {"error": "Total assets or liabilities cannot be zero", "z_score": None}

    a = working_capital / total_assets
    b = retained_earnings / total_assets
    c = ebit / total_assets
    d = market_equity / total_liabilities
    e = sales / total_assets

    z = 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 0.999 * e
    zone = "Safe Zone" if z > 2.99 else ("Grey Zone" if z >= 1.81 else "Distress Zone")

    return {
        "z_score": round(z, 2),
        "zone": zone,
        "solvency_assessment": "Low default risk" if z > 2.99 else "Elevated credit risk",
    }


def calculate_fcf_margin(
    fcf: float,
    revenue: float,
    company_name: str = "",
) -> dict[str, Any]:
    """
    Calculate Free Cash Flow Margin:
    FCF Margin = (Free Cash Flow / Revenue) * 100
    """
    if revenue == 0:
        return {"error": "Revenue cannot be zero", "fcf_margin_pct": None}

    fcf_margin = round((fcf / revenue) * 100, 1)
    prefix = f"{company_name}'s " if company_name else ""
    return {
        "company": company_name,
        "fcf": fcf,
        "revenue": revenue,
        "fcf_margin_pct": fcf_margin,
        "formatted_text": f"{prefix}FCF margin is approximately {fcf_margin}%.",
    }



def extract_comparison_entities_and_metric(query: str) -> dict[str, Any]:
    """
    Extract company entities and target financial metric from a comparison query.
    e.g. "Compare AIKART's revenue with Tesla's revenue" -> entities=["AIKART", "Tesla"], metric="revenue"
    """
    lower = query.lower()
    known_entities = {
        "aikart": "AIKART",
        "tesla": "Tesla",
        "apple": "Apple",
        "sp500": "S&P 500",
        "s&p 500": "S&P 500",
        "s&p": "S&P 500",
    }
    
    found_entities = []
    for k, v in known_entities.items():
        if re.search(r"\b" + re.escape(k) + r"\b", lower):
            if v not in found_entities:
                found_entities.append(v)

    # Also discover capitalized company names from comparison query
    words = re.findall(r"\b[A-Z][a-zA-Z0-9_\-\&]+\b", query)
    stopwords = {"Compare", "Comparison", "Versus", "Vs", "What", "How", "Why", "The", "And", "With", "Between", "From", "In", "Of", "Total", "Revenue", "Margin", "Profit", "Income", "Cash", "Ebitda", "Assets", "Debt", "Eps", "Fiscal", "Year", "Gross", "Net", "Diluted", "Operating", "Q1", "Q2", "Q3", "Q4", "Fy", "Fy2023", "Fy2024", "Fy2025", "Fy2026"}
    for w in words:
        if w not in stopwords and w.title() not in stopwords and w not in found_entities:
            found_entities.append(w)

    # Metric detection
    if "gross margin" in lower or "profit margin" in lower or "margin" in lower:
        metric = "gross_margin"
        metric_label = "Gross Profit Margin"
    elif "net profit" in lower or "net income" in lower or "profit" in lower or "pat" in lower:
        metric = "net_profit"
        metric_label = "Net Profit"
    elif "cash" in lower:
        metric = "cash"
        metric_label = "Cash & Equivalents"
    elif "ebitda" in lower or "operating income" in lower:
        metric = "ebitda"
        metric_label = "Operating Income (EBITDA)"
    elif "eps" in lower or "earnings per share" in lower:
        metric = "eps"
        metric_label = "Diluted EPS"
    elif "debt" in lower:
        metric = "total_debt"
        metric_label = "Total Debt"
    elif "asset" in lower:
        metric = "total_assets"
        metric_label = "Total Assets"
    else:
        metric = "revenue"
        metric_label = "Total Revenue"

    return {
        "entities": found_entities,
        "metric": metric,
        "metric_label": metric_label,
    }


def extract_entity_metric_value(
    chunks: list[dict[str, Any]],
    entity_name: str,
    target_metric: str = "revenue",
) -> dict[str, Any] | None:
    """
    Search chunks belonging to entity_name and extract the exact numerical metric, units, and source.
    """
    ent_lower = entity_name.lower()
    
    # 1. Filter chunks for this entity
    entity_chunks = [
        c for c in chunks
        if ent_lower in c.get("source", "").lower() or ent_lower in c.get("text", "").lower()
    ]
    if not entity_chunks:
        return None

    # 2. Metric keywords and target-specific scoring
    metric_keywords = {
        "revenue": ["total revenue", "net sales", "automotive revenue", "total net sales", "revenue"],
        "gross_margin": ["gross profit margin", "gross margin", "company gross margin", "automotive gross margin", "margin"],
        "net_profit": ["net profit", "net income", "profit after tax", "net earnings"],
        "free_cash_flow": ["free cash flow", "fcf"],
        "fcf": ["free cash flow", "fcf"],
        "cash": ["cash and cash equivalents", "cash balance", "cash & equivalents", "ending cash", "cash"],
        "ebitda": ["adjusted ebitda", "operating income", "ebitda", "income from operations"],
        "eps": ["earnings per share", "diluted eps", "eps"],
        "total_debt": ["total debt", "long-term debt", "debt"],
        "total_assets": ["total assets", "assets"],
    }.get(target_metric, ["revenue"])

    candidates = []
    for c in entity_chunks:
        src = c.get("source", "unknown")
        lines = c.get("text", "").splitlines()
        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("[Source:"):
                continue
            line_lower = line_str.lower()
            if any(k in line_lower for k in metric_keywords) and any(ch.isdigit() for ch in line_str):
                score = 1
                if target_metric in ("free_cash_flow", "fcf"):
                    if "free cash flow" in line_lower or "fcf" in line_lower:
                        score += 10
                    if any(sym in line_str for sym in ["₹", "$", "€", "£"]):
                        score += 5
                    if re.search(r"\b(cr|crore|lakh|billion|million|b|m)\b", line_lower):
                        score += 5
                    if "margin" in line_lower:
                        score -= 15
                elif target_metric == "gross_margin":
                    if "gross margin" in line_lower or "gross profit margin" in line_lower:
                        score += 10
                    if "%" in line_str:
                        score += 8
                    if any(bad in line_lower for bad in ["net profit", "net income", "profit after tax", "total revenue", "retention"]):
                        score -= 10
                elif target_metric == "revenue":
                    if "total revenue" in line_lower or "total net sales" in line_lower:
                        score += 10
                    if any(sym in line_str for sym in ["₹", "$", "€", "£"]):
                        score += 5
                    if re.search(r"\b(cr|crore|lakh|billion|million|b|m)\b", line_lower):
                        score += 5
                    if any(bad in line_lower for bad in ["gross margin", "net income", "retention", "breakdown", "table of contents", "target", "guidance"]):
                        score -= 8
                elif target_metric == "net_profit":
                    if "profit after tax" in line_lower or "net profit" in line_lower or "net income" in line_lower:
                        score += 10
                    if any(sym in line_str for sym in ["₹", "$", "€", "£"]):
                        score += 5
                    if "gross margin" in line_lower:
                        score -= 8

                if ":" in line_str or "=" in line_str:
                    score += 3

                candidates.append((score, line_str, src))


    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_line, best_src = candidates[0]
    clean_line = re.sub(r"^[-*•\d\.]+\s*", "", best_line).strip()

    # Extract number and unit
    if target_metric == "gross_margin":
        currency = "%"
    elif "₹" in clean_line or re.search(r"\b(rs\.?|inr|crore|cr|lakh)\b", clean_line, re.I):
        currency = "INR"
    elif "$" in clean_line or re.search(r"\b(usd|dollar|billion|million)\b", clean_line, re.I):
        currency = "USD"
    elif "%" in clean_line:
        currency = "%"
    else:
        currency = "USD"

    if currency == "%":
        pct_match = re.search(r"([\d,\.]+)\s*%", clean_line)
        if pct_match:
            raw_num = float(pct_match.group(1).replace(",", ""))
            formatted_val = f"{raw_num:.1f}%"
            unit = "%"
        else:
            raw_num = 0.0
            formatted_val = clean_line
            unit = "%"
    else:
        num_match = re.search(r"(?:₹|\$|rs\.?)?\s*([\d,\.]+)\s*(billion|million|crore|cr|lakh|b|m)?", clean_line, re.I)
        raw_num = 0.0
        unit = ""
        formatted_val = clean_line
        if num_match:
            try:
                raw_num = float(num_match.group(1).replace(",", ""))
                unit = (num_match.group(2) or "").strip()
                if currency == "INR":
                    formatted_val = f"₹{raw_num:.1f} {unit.title() if unit else 'Cr'}"
                elif currency == "USD":
                    formatted_val = f"${raw_num:.1f} {unit.title() if unit else 'Billion'}"
            except Exception:
                formatted_val = clean_line



    return {
        "entity": entity_name,
        "formatted_val": formatted_val,
        "raw_num": raw_num,
        "unit": unit,
        "currency": currency,
        "source": best_src,
        "matched_line": clean_line,
    }


def compare_entities_detailed(
    data_a: dict[str, Any] | None,
    data_b: dict[str, Any] | None,
    entity_a: str,
    entity_b: str,
    metric_name: str = "revenue",
    metric_label: str = "Total Revenue",
) -> dict[str, Any]:
    """
    Perform rigorous validation, normalization, and mathematical comparison between two entities.
    """
    # 1. Validation: check if either source/value is missing
    missing = []
    if not data_a or not data_a.get("formatted_val"):
        missing.append(entity_a)
    if not data_b or not data_b.get("formatted_val"):
        missing.append(entity_b)

    if missing:
        missing_str = " and ".join(missing)
        return {
            "error": f"Missing financial data for {missing_str} in indexed documents. Cannot perform comparative analysis without data from both entities.",
            "entity_a": entity_a,
            "entity_b": entity_b,
            "metric_label": metric_label,
            "data_a": data_a,
            "data_b": data_b,
        }

    # 2. Normalization & Calculations
    curr_a = data_a.get("currency")
    curr_b = data_b.get("currency")
    num_a = data_a.get("raw_num", 0.0)
    num_b = data_b.get("raw_num", 0.0)

    # Compute base standardized values
    # Standardize to millions
    def to_standard_millions(raw: float, unit: str, curr: str) -> float:
        u = unit.lower()
        if "billion" in u or u == "b":
            return raw * 1000.0
        if "crore" in u or u == "cr":
            return (raw * 10.0) / 83.5 if curr == "INR" else raw * 10.0
        if "lakh" in u:
            return (raw * 0.1) / 83.5 if curr == "INR" else raw * 0.1
        if "million" in u or u == "m":
            return raw
        return raw

    std_a = to_standard_millions(num_a, data_a.get("unit", ""), curr_a)
    std_b = to_standard_millions(num_b, data_b.get("unit", ""), curr_b)

    ratio_str = "N/A"
    abs_diff_str = "N/A"
    pct_diff_str = "N/A"
    summary_note = ""

    if curr_a == "%" and curr_b == "%":
        abs_diff = round(num_a - num_b, 2)
        pct_diff = round(((num_a - num_b) / num_b) * 100, 2) if num_b != 0 else None
        abs_diff_str = f"{abs_diff:+.1f} percentage points"
        pct_diff_str = f"{pct_diff:+.1f}%" if pct_diff is not None else "N/A"
        higher = entity_a if num_a > num_b else entity_b
        summary_note = f"{higher} has higher {metric_label.lower()} by {abs(abs_diff):.1f}% pts."
    elif std_b > 0 and std_a > 0:
        ratio = round(std_b / std_a, 1) if std_b >= std_a else round(std_a / std_b, 1)
        larger = entity_b if std_b >= std_a else entity_a
        smaller = entity_a if std_b >= std_a else entity_b
        ratio_str = f"{larger} is ~{ratio:,.1f}x larger by {metric_label.lower()}"
        if curr_a == curr_b:
            abs_diff_val = abs(num_a - num_b)
            abs_diff_str = f"{abs_diff_val:.1f} {data_a.get('unit', '')}"
            pct_diff_val = round(((num_a - num_b) / num_b) * 100, 2)
            pct_diff_str = f"{pct_diff_val:+.1f}%"
            summary_note = f"{larger} exceeds {smaller} by {abs_diff_str} ({pct_diff_str})."
        else:
            summary_note = f"{larger} revenue exceeds {smaller} by ~{ratio:,.1f}x (FX normalized at 1 USD ≈ ₹83.5 INR)."

    return {
        "error": None,
        "entity_a": entity_a,
        "entity_b": entity_b,
        "metric_label": metric_label,
        "val_a": data_a["formatted_val"],
        "val_b": data_b["formatted_val"],
        "source_a": data_a["source"],
        "source_b": data_b["source"],
        "ratio_str": ratio_str,
        "abs_diff_str": abs_diff_str,
        "pct_diff_str": pct_diff_str,
        "summary_note": summary_note,
        "data_a": data_a,
        "data_b": data_b,
    }


def format_comparison_markdown(result: dict[str, Any]) -> str:
    """Format comparison result as a clean markdown response."""
    if result.get("error"):
        return f"### Comparative Analysis Error\n\n⚠️ **{result['error']}**\n\nPlease ensure financial reports for both companies are uploaded and indexed before running a comparison."

    entity_a = result["entity_a"]
    entity_b = result["entity_b"]
    label = result["metric_label"]
    val_a = result["val_a"]
    val_b = result["val_b"]
    source_a = result["source_a"]
    source_b = result["source_b"]
    summary = result.get("summary_note", "")
    ratio = result.get("ratio_str", "")

    table = (
        f"### Financial Comparison: {entity_a} vs {entity_b}\n\n"
        f"| Metric | {entity_a} | {entity_b} |\n"
        f"| :--- | ---: | ---: |\n"
        f"| **{label}** | **{val_a}** | **{val_b}** |\n\n"
        f"**Key Findings & Analysis:**\n"
        f"- **{entity_a}**: Reported {label.lower()} of **{val_a}** (*Source: {source_a}*).\n"
        f"- **{entity_b}**: Reported {label.lower()} of **{val_b}** (*Source: {source_b}*).\n"
    )
    if summary:
        table += f"- **Comparison**: {summary}\n"
    if ratio and ratio != "N/A" and ratio not in summary:
        table += f"- **Relative Scale**: {ratio}\n"

    table += f"\n**Sources:**\n  - {source_a}\n  - {source_b}"
    return table


def compare_companies(
    data_a: dict[str, Any],
    data_b: dict[str, Any],
    company_a: str,
    company_b: str,
) -> dict:
    """Produce a side-by-side comparison of two companies' extracted metrics."""
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



