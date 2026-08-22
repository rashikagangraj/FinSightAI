from __future__ import annotations

import json
import re

from src.agents.state import AgentState
from src.agents.tools import (
    calculate_altman_z_score,
    calculate_dupont_roe,
    calculate_fcf_margin,
    calculate_financial_ratio,
    compare_entities_detailed,
    extract_comparison_entities_and_metric,
    extract_entity_metric_value,
    extract_key_metrics,
    format_comparison_markdown,
    search_documents,
)
from src.core.logging import get_logger
from src.llm.factory import get_llm_client

logger = get_logger(__name__)

_SYSTEM_FINANCE = (
    "You are FinAgent, an expert financial analyst AI. "
    "You answer questions about financial documents accurately, citing sources. "
    "Always ground your answers in the provided context. "
    "If you cannot find sufficient evidence, say so explicitly."
)


def _classify_intent_rules(query: str) -> str:
    """Deterministic intent classification based on query keywords and entity count."""
    lower_q = query.lower()
    entities = [e for e in ["aikart", "apple", "tesla", "sp500", "s&p 500", "s&p"] if e in lower_q]
    
    # 1. Comparison requires comparison keywords AND (multiple entities or explicit comparison query)
    has_compare = any(k in lower_q for k in ["compare", "comparison", "versus", "vs.", " vs ", "differ", "difference between", "difference in", "difference"])
    if has_compare and (len(entities) >= 2 or " and " in lower_q or " vs " in lower_q or "with" in lower_q or "between" in lower_q):
        return "comparison"
    
    # 2. Calculation / deep analysis
    calc_keywords = [
        "calculate", "compute", "fcf", "free cash flow", "roa", "roe", "d/e", "debt to equity",
        "debt-to-equity", "dupont", "altman", "z-score", "ratio", "margin calculation", "p/e",
        "wacc", "valuation", "dcf",
    ]
    if any(k in lower_q for k in calc_keywords) or ("margin" in lower_q and not has_compare and "what" not in lower_q):
        return "deep_analysis"
    
    # 3. Structured report
    if any(k in lower_q for k in ["generate a report", "create a report", "financial report", "executive summary", "comprehensive overview", "full report"]):
        return "report"
    
    # 4. Single-company or single-metric questions default to simple_qa
    return "simple_qa"


def planner_node(state: AgentState) -> AgentState:
    """Classify the query intent and write a short retrieval plan."""
    query = state["query"]

    # Rule-based deterministic intent check
    intent = _classify_intent_rules(query)
    plan = f"Intent: {intent}\nSearch queries: {[query]}"
    logger.info(f"[planner] intent={intent}")
    return {
        **state,
        "intent": intent,
        "plan": plan,
        "agent_trace": [f"[planner] classified as '{intent}'"],
    }




def retrieval_node(state: AgentState) -> AgentState:
    """Retrieve relevant document chunks using hybrid search."""
    query = state["query"]
    plan = state.get("plan", "")

    # Extract search queries from plan if available
    search_q = query
    if "Search queries:" in plan:
        try:
            qs = plan.split("Search queries:")[1].strip().strip("[]").split(",")
            search_q = qs[0].strip().strip("'\"") if qs else query
        except Exception:
            pass

    chunks = search_documents(search_q, top_k=5)
    sources = list({c["source"] for c in chunks if c.get("source")})

    logger.info(f"[retrieval] found {len(chunks)} chunks from {len(sources)} sources")
    return {
        **state,
        "retrieved_chunks": chunks,
        "sources": sources,
        "agent_trace": [f"[retrieval] retrieved {len(chunks)} chunks"],
    }


def qa_node(state: AgentState) -> AgentState:
    """Answer the query from retrieved context with exact numerical extraction."""
    llm = get_llm_client()
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {
            **state,
            "answer": "I could not find relevant information in the indexed documents to answer your question.",
            "agent_trace": ["[qa] no context available"],
        }

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    prompt = f"""Answer the following question using ONLY the provided context.
Provide the exact numerical value, metric, and cite the source document.
Do NOT give generic or theoretical responses when exact numbers are present in the text.

Context:
{context}

Question: {query}

Answer:"""

    answer = llm.complete(prompt, system=_SYSTEM_FINANCE)
    logger.info(f"[qa] answer length={len(answer)} chars")
    return {
        **state,
        "answer": answer,
        "agent_trace": [f"[qa] generated answer ({len(answer)} chars)"],
    }



def analysis_node(state: AgentState) -> AgentState:
    """Extract metrics and perform financial calculations."""
    llm = get_llm_client()
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    combined_text = " ".join(c["text"] for c in chunks)
    lower_q = query.lower()

    # Identify entity name
    ent_match = re.search(r"\b(AIKART|Tesla|Apple|Reliance)\b", query, re.I)
    entity_name = ent_match.group(1).title() if ent_match else ("AIKART" if "aikart" in lower_q else "")
    if entity_name.upper() == "AIKART":
        entity_name = "AIKART"

    # 1. FCF Margin Calculation
    if "fcf" in lower_q or "free cash flow" in lower_q:
        data_fcf = extract_entity_metric_value(chunks, entity_name, "free_cash_flow")
        data_rev = extract_entity_metric_value(chunks, entity_name, "revenue")

        fcf_val = data_fcf["raw_num"] if data_fcf else None
        rev_val = data_rev["raw_num"] if data_rev else None
        source_doc = (
            data_fcf["source"]
            if (data_fcf and data_fcf.get("source"))
            else (data_rev["source"] if (data_rev and data_rev.get("source")) else (chunks[0]["source"] if chunks else "financial report"))
        )

        if fcf_val is not None and rev_val is not None and rev_val > 0:
            try:
                fcf_res = calculate_fcf_margin(fcf_val, rev_val, entity_name)
                margin = fcf_res["fcf_margin_pct"]
                answer = f"{entity_name}'s FCF margin is approximately {margin}%.\n\nSource: {source_doc}"
                return {
                    **state,
                    "analysis": f"FCF: {fcf_val}, Revenue: {rev_val}, FCF Margin: {margin}%",
                    "answer": answer,
                    "sources": [source_doc],
                    "agent_trace": [f"[analysis] calculated FCF margin = {margin}%"],
                }
            except Exception as exc:
                logger.warning(f"FCF margin calculation error: {exc}")



    metrics = extract_key_metrics(combined_text)

    analysis_parts = []
    if metrics:
        analysis_parts.append("**Extracted Metrics:**")
        for k, v in metrics.items():
            if v is not None:
                analysis_parts.append(f"  - {k.replace('_', ' ').title()}: {v:,.2f}" if isinstance(v, float) else f"  - {k}: {v}")

        if metrics.get("net_income") and metrics.get("total_assets"):
            roa = calculate_financial_ratio(metrics["net_income"], metrics["total_assets"], "Return on Assets (ROA)")
            analysis_parts.append(f"\n**Calculated Ratios:**")
            analysis_parts.append(f"  - {roa['ratio_name']}: {roa['value']:.4f}")

    analysis_summary = "\n".join(analysis_parts) if analysis_parts else "No structured metrics found in context."

    prompt = f"""Given this financial analysis and the original query, provide a clear analytical response.

{analysis_summary}

Context excerpts:
{combined_text[:2000]}

Query: {query}

Analytical response:"""

    answer = llm.complete(prompt, system=_SYSTEM_FINANCE)
    return {
        **state,
        "analysis": analysis_summary,
        "answer": answer,
        "agent_trace": [f"[analysis] extracted {len(metrics)} metrics, generated response"],
    }



def comparison_node(state: AgentState) -> AgentState:
    """Compare two entities by extracting, validating, and contrasting their metrics."""
    llm = get_llm_client()
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])

    # 1. Extract entities and target metric from query
    info = extract_comparison_entities_and_metric(query)
    entities = info["entities"]
    metric = info["metric"]
    metric_label = info["metric_label"]

    if len(entities) < 2:
        return {
            **state,
            "answer": "### Comparative Analysis Error\n\n⚠️ Comparative analysis requires two or more companies or entities to compare. Please specify both companies in your query (e.g., 'Compare AIKART and Tesla revenue').",
            "agent_trace": ["[comparison] aborted: fewer than 2 entities specified in query"],
        }

    entity_a, entity_b = entities[0], entities[1]

    # 2. Extract verified metrics for each entity from retrieved chunks
    data_a = extract_entity_metric_value(chunks, entity_a, metric)
    data_b = extract_entity_metric_value(chunks, entity_b, metric)

    # 3. Perform mathematical comparison & source validation (detect missing data)
    comparison_res = compare_entities_detailed(
        data_a=data_a,
        data_b=data_b,
        entity_a=entity_a,
        entity_b=entity_b,
        metric_name=metric,
        metric_label=metric_label,
    )

    # 4. Generate structured markdown table and findings
    answer = format_comparison_markdown(comparison_res)

    sources = []
    if data_a and data_a.get("source"):
        sources.append(data_a["source"])
    if data_b and data_b.get("source"):
        sources.append(data_b["source"])
    if not sources:
        sources = state.get("sources", [])

    return {
        **state,
        "answer": answer,
        "sources": list(dict.fromkeys(sources)),
        "agent_trace": [
            f"[comparison] extracted {metric_label} for {entity_a} and {entity_b}",
            f"[comparison] generated comparative financial matrix",
        ],
    }



def report_node(state: AgentState) -> AgentState:
    """Generate a structured markdown financial report."""
    llm = get_llm_client()
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    combined_text = " ".join(c["text"] for c in chunks)
    metrics = extract_key_metrics(combined_text)
    sources = state.get("sources", [])

    prompt = f"""Generate a professional financial analysis report in Markdown format.

Available data:
{combined_text[:3000]}

Extracted metrics:
{json.dumps(metrics, indent=2, default=str)}

Report requested: {query}

Structure the report with these sections:
## Executive Summary
## Key Financial Metrics
## Analysis & Insights
## Risks & Considerations
## Sources

Report:"""

    report = llm.complete(prompt, system=_SYSTEM_FINANCE)
    return {
        **state,
        "answer": report,
        "agent_trace": [
            f"[report] generated {len(report)} char report from {len(sources)} sources"
        ],
    }


def output_node(state: AgentState) -> AgentState:
    """Final formatting — ensures sources are clean and answer is complete."""
    sources = list(set(state.get("sources", [])))
    answer = state.get("answer", "No answer generated.")

    if sources:
        src_list = "\n".join(f"  - {s}" for s in sources)
        if "**Sources**" not in answer and "## Sources" not in answer:
            answer = answer + f"\n\n**Sources:**\n{src_list}"

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "agent_trace": ["[output] formatted final response"],
    }


def route_by_intent(state: AgentState) -> str:
    """LangGraph conditional edge: route to the correct worker node."""
    intent = state.get("intent", "simple_qa")
    routes = {
        "simple_qa": "qa",
        "deep_analysis": "analysis",
        "comparison": "comparison",
        "report": "report",
        "unknown": "qa",
    }
    return routes.get(intent, "qa")
