from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class FinancialHistorical(BaseModel):
    year: str
    revenue: float
    cogs: float
    gross_profit: float
    operating_expenses: float
    ebitda: float
    net_income: float
    capex: float
    fcf: float


class CaseStudy(BaseModel):
    id: str
    title: str
    company_name: str
    ticker: str = ""
    industry: str
    topic: Literal["M&A", "DCF Valuation", "Equity Research", "Credit & Distress", "Growth Investment", "Corporate Finance"]
    difficulty: Literal["Beginner", "Intermediate", "Advanced", "Expert"]
    headline: str
    description: str
    key_question: str
    context_dossier: str
    historical_financials: list[FinancialHistorical] = Field(default_factory=list)
    current_metrics: dict[str, Any] = Field(default_factory=dict)
    ground_truth_insights: list[str] = Field(default_factory=list)
    potential_risks: list[str] = Field(default_factory=list)
    consensus_decision: str = "BUY"
    target_valuation_range: str = "$45B - $52B"


class ScenarioParams(BaseModel):
    revenue_growth_delta_pct: float = 0.0  # e.g., +5.0% for Bull, -8.0% for Bear
    ebitda_margin_delta_pct: float = 0.0   # e.g., +2.0%
    wacc_pct: float = 9.5                 # Discount rate %
    terminal_growth_pct: float = 2.5      # Terminal growth %
    exit_multiple: float = 12.0           # EV/EBITDA multiple
    tax_rate_pct: float = 21.0
    capex_pct_rev: float = 4.5
    net_debt: float = 0.0
    shares_outstanding_m: float = 100.0


class SimulationResult(BaseModel):
    projected_years: list[str]
    projected_revenue: list[float]
    projected_ebitda: list[float]
    projected_fcf: list[float]
    pv_fcf: float
    terminal_value_dcf: float
    enterprise_value_dcf: float
    equity_value_dcf: float
    implied_share_price_dcf: float
    enterprise_value_multiple: float
    implied_share_price_multiple: float
    waterfall: dict[str, float]
    scenarios: dict[str, dict[str, Any]]
    sensitivity_matrix: dict[str, Any]


class UserSubmission(BaseModel):
    case_id: str
    user_decision: Literal["STRONG BUY", "BUY", "HOLD", "SELL", "INVEST", "PASS / REJECT"]
    target_price: float | None = None
    target_valuation: float | None = None
    key_assumptions: str
    strategic_rationale: str
    risk_assessment: str
    applied_scenario_params: ScenarioParams | None = None


class EvaluationFeedback(BaseModel):
    overall_score: int  # 0 - 100
    grade: str          # e.g., "A- (Senior Analyst Level)"
    verdict: str        # e.g., "Sound Strategy with Understated Execution Risk"
    scores_breakdown: dict[str, int]  # valuation_accuracy, risk_rigor, strategic_depth, scenario_stress
    ai_consensus_decision: str
    strengths: list[str]
    identified_mistakes: list[str]
    unaddressed_risks: list[str]
    expert_coaching_advice: str
    recommended_model_adjustments: dict[str, Any] = Field(default_factory=dict)


class LeaderboardEntry(BaseModel):
    user_name: str
    case_id: str
    case_title: str
    score: int
    grade: str
    decision: str
    timestamp: str
