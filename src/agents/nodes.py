from __future__ import annotations

import json

from src.agents.state import AgentState
from src.agents.tools import (
    calculate_financial_ratio,
    compare_companies,
    extract_key_metrics,
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


def planner_node(state: AgentState) -> AgentState:
    """Classify the query intent and write a short retrieval plan."""
    llm = get_llm_client()
    query = state["query"]

    prompt = f"""Classify this financial query into exactly ONE of these intents:
- simple_qa: a direct factual question (e.g. "What was Apple's revenue in 2023?")
- deep_analysis: requires calculation or multi-step reasoning (e.g. "Calculate P/E ratio")
- comparison: comparing two companies or time periods (e.g. "Compare Apple vs Tesla margins")
- report: request for a structured summary or report

Query: {query}

Respond with ONLY a JSON object like:
{{"intent": "<one of the above>", "search_queries": ["<query1>", "<query2>"]}}
"""
    try:
        raw = llm.complete(prompt, system=_SYSTEM_FINANCE)
        # Extract JSON from response
        json_match = raw[raw.find("{") : raw.rfind("}") + 1]
        parsed = json.loads(json_match)
        intent = parsed.get("intent", "simple_qa")
        search_queries = parsed.get("search_queries", [query])
    except Exception as exc:
        logger.warning(f"Planner JSON parse failed: {exc} — defaulting to simple_qa")
        intent = "simple_qa"
        search_queries = [query]

    plan = f"Intent: {intent}\nSearch queries: {search_queries}"
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
    """Answer the query from retrieved context."""
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
Include specific numbers and cite the source document.

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
    """Compare two entities by extracting and contrasting their metrics."""
    llm = get_llm_client()
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])

    # Split chunks by source company
    source_texts: dict[str, str] = {}
    for chunk in chunks:
        src = chunk.get("source", "unknown")
        source_texts[src] = source_texts.get(src, "") + " " + chunk["text"]

    all_text = " ".join(source_texts.values())
    metrics_all = extract_key_metrics(all_text)

    comparison_context = json.dumps(
        {src: extract_key_metrics(txt) for src, txt in source_texts.items()},
        indent=2,
        default=str,
    )

    prompt = f"""You are comparing financial entities. Use the extracted data below to answer the comparison question.

Extracted data by source:
{comparison_context}

Query: {query}

Provide a structured side-by-side comparison with key takeaways:"""

    answer = llm.complete(prompt, system=_SYSTEM_FINANCE)
    return {
        **state,
        "answer": answer,
        "agent_trace": [f"[comparison] compared {len(source_texts)} sources"],
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
