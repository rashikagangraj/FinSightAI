from __future__ import annotations

import pytest
from src.cases.case_engine import get_case_by_id, list_all_cases
from src.cases.models import FinancialHistorical, ScenarioParams, UserSubmission
from src.cases.simulator import run_financial_simulation
from src.cases.evaluator import evaluate_user_case_submission


def test_list_and_get_cases():
    cases = list_all_cases()
    assert len(cases) >= 4
    case = get_case_by_id("case-tech-saas-mna")
    assert case is not None
    assert case.company_name == "CloudPeak Systems Inc."
    assert case.topic == "M&A"


def test_financial_simulation_dcf_and_scenarios():
    historical = [
        FinancialHistorical(year="2021", revenue=100.0, cogs=60.0, gross_profit=40.0, operating_expenses=20.0, ebitda=20.0, net_income=12.0, capex=5.0, fcf=10.0),
        FinancialHistorical(year="2022", revenue=130.0, cogs=75.0, gross_profit=55.0, operating_expenses=25.0, ebitda=30.0, net_income=18.0, capex=7.0, fcf=15.0),
        FinancialHistorical(year="2023", revenue=170.0, cogs=95.0, gross_profit=75.0, operating_expenses=35.0, ebitda=40.0, net_income=25.0, capex=10.0, fcf=22.0),
    ]
    params = ScenarioParams(
        revenue_growth_delta_pct=2.0,
        ebitda_margin_delta_pct=1.0,
        wacc_pct=10.0,
        terminal_growth_pct=2.5,
        exit_multiple=12.0,
        shares_outstanding_m=50.0,
        net_debt=20.0,
    )
    result = run_financial_simulation(historical, params)

    assert len(result.projected_years) == 5
    assert result.enterprise_value_dcf > 0
    assert result.equity_value_dcf > 0
    assert result.implied_share_price_dcf > 0
    assert "Bear Case" in result.scenarios
    assert "Bull Case" in result.scenarios
    assert len(result.sensitivity_matrix["wacc_labels"]) == 5


def test_evaluate_user_submission():
    sub = UserSubmission(
        case_id="case-tech-saas-mna",
        user_decision="BUY",
        target_valuation=4200.0,
        key_assumptions="28% revenue growth, 9.5% WACC, $35M cost synergies",
        strategic_rationale="Consolidates enterprise cloud AI moat and accelerates cross-selling.",
        risk_assessment="Customer concentration and floating debt headwinds.",
    )
    feedback = evaluate_user_case_submission(sub)
    assert 0 <= feedback.overall_score <= 100
    assert feedback.grade != ""
    assert len(feedback.strengths) > 0
    assert len(feedback.identified_mistakes) > 0
    assert feedback.expert_coaching_advice != ""
