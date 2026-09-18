from __future__ import annotations

import json
import uuid
from typing import Any
from src.cases.models import CaseStudy, FinancialHistorical
from src.core.logging import get_logger
from src.llm.factory import get_llm_client

logger = get_logger(__name__)

# Preloaded Benchmark Case Studies
BENCHMARK_CASES: list[CaseStudy] = [
    CaseStudy(
        id="case-tech-saas-mna",
        title="Project CloudPeak: $4.2B Cross-Border SaaS Acquisition",
        company_name="CloudPeak Systems Inc.",
        ticker="CPK",
        industry="Enterprise Software & Cloud AI",
        topic="M&A",
        difficulty="Advanced",
        headline="Strategic Acquisition vs. Organic Build Decision Under High Interest Rates",
        description="A major enterprise infrastructure conglomerate is evaluating the 100% stock-and-cash buyout of CloudPeak Systems at a 32% market premium. Analyze ARR retention, gross margin accretion, customer acquisition cost (CAC) payback, and integration risk.",
        key_question="Should the Investment Committee approve the $4.2B acquisition at 9.5x NTM Revenue, or pass in favor of an internal AI platform build?",
        context_dossier="""
## Executive Brief: Project CloudPeak
CloudPeak Systems provides mission-critical enterprise data synchronization and AI workflow engines for Fortune 500 banks and retail giants.
- **Current ARR**: $380M (Growing 34% YoY)
- **Net Revenue Retention (NRR)**: 124%
- **Gross Margin**: 74.5% (Expanding +150 bps YoY due to GPU optimization)
- **Total Debt**: $210M (Floating rate SOFR + 3.25%)
- **Cash & Short-Term Investments**: $115M
- **Shares Outstanding**: 85M shares trading at $38.50/share
- **Proposed Deal Terms**: $51.00 per share (60% cash, 40% stock), representing $4.335B Equity Value.
- **Key Risks**: High customer concentration (Top 5 clients contribute 38% of ARR), expiring enterprise MSAs in Q4, and potential regulatory anti-trust scrutiny in the EU.
        """,
        historical_financials=[
            FinancialHistorical(year="2021", revenue=210.0, cogs=60.0, gross_profit=150.0, operating_expenses=140.0, ebitda=28.0, net_income=12.0, capex=14.0, fcf=16.0),
            FinancialHistorical(year="2022", revenue=285.0, cogs=78.0, gross_profit=207.0, operating_expenses=175.0, ebitda=48.0, net_income=26.0, capex=18.0, fcf=32.0),
            FinancialHistorical(year="2023", revenue=380.0, cogs=97.0, gross_profit=283.0, operating_expenses=215.0, ebitda=82.0, net_income=51.0, capex=22.0, fcf=64.0),
        ],
        current_metrics={
            "revenue_ttm": "$380M",
            "arr_growth": "34.0%",
            "ebitda_margin": "21.6%",
            "nrr": "124%",
            "offer_price_per_share": "$51.00",
            "implied_ev": "$4.43B",
            "wacc": "10.2%",
        },
        ground_truth_insights=[
            "At 9.5x NTM Revenue, the valuation is in the 85th percentile of historical cloud M&A multiples.",
            "Run-rate cost synergies of $45M/year are needed to achieve the target 16.5% IRR hurdle rate.",
            "Top client concentration poses an asymmetrical downside risk if contract renegotiations stumble in Q4.",
        ],
        potential_risks=[
            "Integration friction between proprietary Kubernetes infra and acquirer legacy stack.",
            "Floating debt cost expansion if central bank rate cuts stall.",
            "Talent attrition of key AI engineering leadership post-earn-out.",
        ],
        consensus_decision="BUY",
        target_valuation_range="$3.9B - $4.4B",
    ),
    CaseStudy(
        id="case-ev-cap-allocation",
        title="VoltDynamics: $12B CapEx Gigafactory Expansion Decision",
        company_name="VoltDynamics Motors Corp.",
        ticker="VDYN",
        industry="Automotive & Energy Storage",
        topic="DCF Valuation",
        difficulty="Intermediate",
        headline="Evaluating Returns on Capital for Next-Gen Solid State Battery Gigafactory",
        description="VoltDynamics is deciding whether to greenlight a massive $12 Billion capital expenditure program over 4 years to commercialize solid-state battery cells, or return capital to shareholders via buybacks and dividends.",
        key_question="Will the project NPV justify the 11.5% cost of capital under escalating lithium price volatility and EV price compression?",
        context_dossier="""
## Executive Brief: VoltDynamics Capital Allocation
VoltDynamics is experiencing slowing automotive margins (down from 26% to 18.2%) due to global price competition.
- **2023 Total Revenue**: $24.8 Billion
- **Operating Margin**: 11.4%
- **Free Cash Flow (2023)**: $2.4 Billion
- **Cash Balance**: $8.2 Billion
- **Total Debt**: $4.5 Billion
- **Proposed Gigafactory Investment**: $3.0B/year for 4 years ($12B total CapEx)
- **Expected Production Lift**: 450,000 battery packs/year starting Year 3 with 35% unit cost reduction.
- **Sensitivity Threshold**: Break-even requires cell manufacturing cost to fall below $78/kWh by 2027.
        """,
        historical_financials=[
            FinancialHistorical(year="2021", revenue=14200.0, cogs=10500.0, gross_profit=3700.0, operating_expenses=1900.0, ebitda=2400.0, net_income=1400.0, capex=1800.0, fcf=900.0),
            FinancialHistorical(year="2022", revenue=19800.0, cogs=14600.0, gross_profit=5200.0, operating_expenses=2600.0, ebitda=3400.0, net_income=2100.0, capex=2600.0, fcf=1500.0),
            FinancialHistorical(year="2023", revenue=24800.0, cogs=19300.0, gross_profit=5500.0, operating_expenses=3100.0, ebitda=3600.0, net_income=2250.0, capex=3200.0, fcf=2400.0),
        ],
        current_metrics={
            "revenue_ttm": "$24.8B",
            "ebitda_margin": "14.5%",
            "net_debt": "-$3.7B (Net Cash)",
            "capex_growth": "+23%",
            "wacc": "11.5%",
        },
        ground_truth_insights=[
            "Without solid-state vertical integration, gross margins will permanently degrade to ~14% by 2028.",
            "The company's strong net cash position allows financing the expansion without dilutive equity issuance.",
            "Projected project IRR is 18.2%, surpassing the 11.5% WACC hurdle.",
        ],
        potential_risks=[
            "Battery yield ramps could encounter unforeseen manufacturing defects delaying ROI.",
            "Lithium carbonate pricing spikes could compress payback horizon by 2.5 years.",
        ],
        consensus_decision="INVEST",
        target_valuation_range="$65B - $78B",
    ),
    CaseStudy(
        id="case-retail-distress-lbo",
        title="Apex Brands: Restructuring & Debt Solvency Analysis",
        company_name="Apex Global Retail Brands",
        ticker="APEX",
        industry="Consumer Retail & Apparel",
        topic="Credit & Distress",
        difficulty="Expert",
        headline="Altman Z-Score Distress, Refinancing Cliff & Operational Turnaround",
        description="Apex Brands has $1.8B of Senior Secured Notes maturing in 18 months with leverage sitting at 6.8x Net Debt / EBITDA. Determine if a distressed debt restructuring or asset sale can prevent Chapter 11 filing.",
        key_question="Should creditors accept a 25% debt haircut in exchange for 40% equity warrants, or force an orderly asset liquidation?",
        context_dossier="""
## Executive Brief: Apex Global Restructuring
Apex Brands owns 4 legacy apparel labels suffering from mall foot-traffic declines and e-commerce channel cannibalization.
- **2023 Revenue**: $3.1 Billion (-6.2% YoY)
- **EBITDA**: $265 Million (down from $410M in 2021)
- **Total Outstanding Debt**: $1.82 Billion
- **Interest Expense**: $145 Million / year (effective rate 8.0%)
- **Interest Coverage Ratio**: 1.82x (Severely constrained)
- **Altman Z-Score**: 1.34 (Distress Territory)
- **Liquidation Value of Real Estate & Inventory**: Estimated at $1.15 Billion
        """,
        historical_financials=[
            FinancialHistorical(year="2021", revenue=3600.0, cogs=2100.0, gross_profit=1500.0, operating_expenses=1090.0, ebitda=410.0, net_income=180.0, capex=120.0, fcf=140.0),
            FinancialHistorical(year="2022", revenue=3350.0, cogs=2050.0, gross_profit=1300.0, operating_expenses=980.0, ebitda=320.0, net_income=85.0, capex=95.0, fcf=60.0),
            FinancialHistorical(year="2023", revenue=3100.0, cogs=1980.0, gross_profit=1120.0, operating_expenses=855.0, ebitda=265.0, net_income=-35.0, capex=60.0, fcf=-15.0),
        ],
        current_metrics={
            "revenue_ttm": "$3.1B",
            "ebitda_margin": "8.5%",
            "net_debt_to_ebitda": "6.6x",
            "interest_coverage": "1.82x",
            "z_score": "1.34",
        },
        ground_truth_insights=[
            "FCF is currently negative (-$15M) after debt service, creating a severe liquidity depletion trajectory.",
            "Debt-for-Equity swap reduces annual interest burden by $55M, restoring positive free cash flow.",
            "Liquidation yields only 63 cents on the dollar for senior debtholders.",
        ],
        potential_risks=[
            "Further store traffic attrition could erase projected post-restructuring EBITDA recovery.",
            "Supply chain vendors demanding Cash Before Delivery (CBD) could trigger working capital squeeze.",
        ],
        consensus_decision="HOLD",
        target_valuation_range="$950M - $1.2B",
    ),
    CaseStudy(
        id="case-semiconductor-dcf",
        title="NovaSilicon: AI ASIC Accelerator Equity Research Valuation",
        company_name="NovaSilicon Microelectronics",
        ticker="NVAS",
        industry="Semiconductor Hardware & Foundries",
        topic="Equity Research",
        difficulty="Intermediate",
        headline="Initiating Coverage with 5-Year DCF & Multi-Scenario Valuation Model",
        description="NovaSilicon has launched its custom neural processing unit (NPU) for high-density edge inferencing. Model the wafer cost ramp, gross margin expansion, and market share capture against incumbents.",
        key_question="Is NovaSilicon a compelling BUY at current market price of $142/share with a 15% discount rate?",
        context_dossier="""
## Executive Brief: NovaSilicon Equity Research Initiation
NovaSilicon designs power-efficient 3nm AI inference accelerators tailored for autonomous robotics and hyper-scale edge deployments.
- **2023 Total Revenue**: $1.65 Billion (+52% YoY)
- **Gross Margin**: 66.8%
- **Operating Margin**: 31.2%
- **Net Cash Balance**: $840 Million (Zero long-term debt)
- **Current Share Price**: $142.00 | Shares: 50 Million | Market Cap: $7.1 Billion
- **Foundry Partner**: Guaranteed 12,000 wafer starts/month allocation for next 2 years.
        """,
        historical_financials=[
            FinancialHistorical(year="2021", revenue=680.0, cogs=250.0, gross_profit=430.0, operating_expenses=260.0, ebitda=195.0, net_income=135.0, capex=45.0, fcf=120.0),
            FinancialHistorical(year="2022", revenue=1085.0, cogs=380.0, gross_profit=705.0, operating_expenses=390.0, ebitda=350.0, net_income=260.0, capex=75.0, fcf=210.0),
            FinancialHistorical(year="2023", revenue=1650.0, cogs=548.0, gross_profit=1102.0, operating_expenses=587.0, ebitda=555.0, net_income=420.0, capex=110.0, fcf=365.0),
        ],
        current_metrics={
            "revenue_ttm": "$1.65B",
            "rev_growth_yoy": "52.1%",
            "gross_margin": "66.8%",
            "pe_ratio": "16.9x",
            "wacc": "12.0%",
        },
        ground_truth_insights=[
            "Trading at 16.9x P/E with 50%+ top-line growth represents a PEG ratio of ~0.35, exceptionally attractive.",
            "Base Case DCF yields fair equity value of $194/share, offering +36.6% upside potential.",
            "Foundry wafer allocation agreement shields from near-term supply bottleneck risk.",
        ],
        potential_risks=[
            "Custom silicon design cycles at mega-cap cloud customers could reduce reliance on merchant chips by 2026.",
            "Export control restrictions in Asian markets could shave 12% off addressable pipeline.",
        ],
        consensus_decision="STRONG BUY",
        target_valuation_range="$9.2B - $10.5B ($184 - $210/share)",
    ),
]

_GENERATED_CASES_CACHE: list[CaseStudy] = []


def list_all_cases(topic_filter: str | None = None, difficulty_filter: str | None = None) -> list[CaseStudy]:
    """Return combined benchmark and generated case studies with optional filtering."""
    all_cases = BENCHMARK_CASES + _GENERATED_CASES_CACHE
    filtered = all_cases
    if topic_filter and topic_filter.lower() != "all":
        filtered = [c for c in filtered if c.topic.lower() == topic_filter.lower()]
    if difficulty_filter and difficulty_filter.lower() != "all":
        filtered = [c for c in filtered if c.difficulty.lower() == difficulty_filter.lower()]
    return filtered


def get_case_by_id(case_id: str) -> CaseStudy | None:
    for c in BENCHMARK_CASES + _GENERATED_CASES_CACHE:
        if c.id == case_id:
            return c
    return None


def generate_ai_case(industry: str, topic: str, difficulty: str) -> CaseStudy:
    """Generate a realistic financial case study on demand using the LLM backend."""
    llm = get_llm_client()
    prompt = f"""Generate a realistic, comprehensive, and data-rich financial case study in JSON format.
Industry: {industry}
Topic: {topic}
Difficulty Level: {difficulty}

The JSON MUST match this exact schema:
{{
  "title": "Title of the case study",
  "company_name": "Company Name",
  "ticker": "3-4 letter ticker",
  "headline": "One sentence summary of the strategic dilemma",
  "description": "2-3 paragraphs of rich context and financial background",
  "key_question": "The central strategic or valuation question for the analyst",
  "context_dossier": "Markdown formatted detailed financial breakdown with revenue, margins, balance sheet items, and operating KPIs",
  "historical_financials": [
    {{"year": "2021", "revenue": 100.0, "cogs": 60.0, "gross_profit": 40.0, "operating_expenses": 25.0, "ebitda": 20.0, "net_income": 12.0, "capex": 5.0, "fcf": 10.0}},
    {{"year": "2022", "revenue": 130.0, "cogs": 75.0, "gross_profit": 55.0, "operating_expenses": 32.0, "ebitda": 28.0, "net_income": 18.0, "capex": 7.0, "fcf": 15.0}},
    {{"year": "2023", "revenue": 175.0, "cogs": 98.0, "gross_profit": 77.0, "operating_expenses": 42.0, "ebitda": 40.0, "net_income": 26.0, "capex": 10.0, "fcf": 22.0}}
  ],
  "current_metrics": {{
    "revenue_ttm": "$175M",
    "ebitda_margin": "22.8%",
    "wacc": "10.5%"
  }},
  "ground_truth_insights": ["Key financial insight 1", "Key financial insight 2", "Key financial insight 3"],
  "potential_risks": ["Risk 1", "Risk 2", "Risk 3"],
  "consensus_decision": "BUY",
  "target_valuation_range": "$2.1B - $2.5B"
}}

Respond ONLY with valid, raw JSON (no surrounding markdown code fences)."""

    raw = llm.complete(prompt, system="You are an elite Wall Street investment banking case writer and financial model architect.")
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(cleaned)
        case_id = f"case-{uuid.uuid4().hex[:8]}"
        historical = [FinancialHistorical(**h) for h in data.get("historical_financials", [])]
        new_case = CaseStudy(
            id=case_id,
            title=data.get("title", f"{industry} Financial Valuation Case"),
            company_name=data.get("company_name", "Global Enterprise Corp"),
            ticker=data.get("ticker", "GEC"),
            industry=industry,
            topic=topic if topic in ["M&A", "DCF Valuation", "Equity Research", "Credit & Distress", "Growth Investment", "Corporate Finance"] else "DCF Valuation",
            difficulty=difficulty if difficulty in ["Beginner", "Intermediate", "Advanced", "Expert"] else "Intermediate",
            headline=data.get("headline", "Financial and Strategic Valuation Decision"),
            description=data.get("description", "Comprehensive financial analysis and valuation assessment."),
            key_question=data.get("key_question", "What is the optimal capital allocation or valuation decision?"),
            context_dossier=data.get("context_dossier", "Financial background and operating data."),
            historical_financials=historical,
            current_metrics=data.get("current_metrics", {}),
            ground_truth_insights=data.get("ground_truth_insights", []),
            potential_risks=data.get("potential_risks", []),
            consensus_decision=data.get("consensus_decision", "BUY"),
            target_valuation_range=data.get("target_valuation_range", "$1.0B - $1.5B"),
        )
        _GENERATED_CASES_CACHE.insert(0, new_case)
        return new_case
    except Exception as exc:
        logger.error(f"Failed to parse generated case: {exc}")
        # Fallback to a structured generated case
        fallback_id = f"case-{uuid.uuid4().hex[:8]}"
        fallback = CaseStudy(
            id=fallback_id,
            title=f"Strategic Valuation Assessment: {industry} Growth",
            company_name=f"{industry} Solutions Corp.",
            ticker="STRAT",
            industry=industry,
            topic=topic if topic in ["M&A", "DCF Valuation", "Equity Research", "Credit & Distress", "Growth Investment", "Corporate Finance"] else "DCF Valuation",
            difficulty=difficulty if difficulty in ["Beginner", "Intermediate", "Advanced", "Expert"] else "Intermediate",
            headline="Capital Allocation, Margin Defense, and Market Expansion Analysis",
            description="Evaluate corporate performance, cost of capital, and DCF enterprise valuation under dynamic macroeconomic shifts.",
            key_question="Does the projected 5-year cash flow profile support the required return on invested capital?",
            context_dossier=f"Detailed financial records and operational metrics for {industry} market leader.",
            historical_financials=[
                FinancialHistorical(year="2021", revenue=500.0, cogs=300.0, gross_profit=200.0, operating_expenses=120.0, ebitda=95.0, net_income=55.0, capex=30.0, fcf=45.0),
                FinancialHistorical(year="2022", revenue=640.0, cogs=380.0, gross_profit=260.0, operating_expenses=155.0, ebitda=125.0, net_income=72.0, capex=38.0, fcf=60.0),
                FinancialHistorical(year="2023", revenue=820.0, cogs=475.0, gross_profit=345.0, operating_expenses=195.0, ebitda=175.0, net_income=105.0, capex=48.0, fcf=85.0),
            ],
            current_metrics={"revenue_ttm": "$820M", "ebitda_margin": "21.3%", "wacc": "10.0%"},
            ground_truth_insights=["High free cash flow conversion provides balance sheet flexibility."],
            potential_risks=["Supply chain and raw material cost inflation."],
            consensus_decision="BUY",
            target_valuation_range="$1.8B - $2.2B",
        )
        _GENERATED_CASES_CACHE.insert(0, fallback)
        return fallback
