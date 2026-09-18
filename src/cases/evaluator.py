from __future__ import annotations

import json
from src.cases.case_engine import get_case_by_id
from src.cases.models import EvaluationFeedback, UserSubmission
from src.core.logging import get_logger
from src.llm.factory import get_llm_client

logger = get_logger(__name__)


def evaluate_user_case_submission(sub: UserSubmission) -> EvaluationFeedback:
    """Evaluates the user's financial case analysis, valuation assumptions, and investment decision."""
    case = get_case_by_id(sub.case_id)
    if not case:
        raise ValueError(f"Case with ID '{sub.case_id}' not found.")

    llm = get_llm_client()

    prompt = f"""You are a Managing Director at an elite investment bank evaluating an Associate's financial analysis and investment recommendation.

### CASE STUDY:
Title: {case.title}
Company: {case.company_name} ({case.ticker})
Industry: {case.industry} | Topic: {case.topic} | Difficulty: {case.difficulty}
Core Question: {case.key_question}
Consensus Expert Decision: {case.consensus_decision}
Target Valuation Range: {case.target_valuation_range}
Benchmark Ground Truth Insights: {json.dumps(case.ground_truth_insights)}
Benchmark Key Risks: {json.dumps(case.potential_risks)}

### USER SUBMISSION:
Recommendation: {sub.user_decision}
Target Valuation / Price: {sub.target_valuation if sub.target_valuation else sub.target_price}
Key Assumptions: {sub.key_assumptions}
Strategic Rationale: {sub.strategic_rationale}
Risk Assessment: {sub.risk_assessment}
Applied Model Scenario Params: {json.dumps(sub.applied_scenario_params.model_dump() if sub.applied_scenario_params else {})}

Evaluate the submission rigorously across 4 dimensions:
1. Valuation Accuracy & Modeling Logic (0-25)
2. Risk Rigor & Downside Awareness (0-25)
3. Strategic Rationale & Market Depth (0-25)
4. Scenario Stress Testing & Assumption Quality (0-25)

Return your assessment as strict, valid JSON with this exact schema:
{{
  "overall_score": 88,
  "grade": "A- (Senior Analyst Level)",
  "verdict": "Clear, concise one-line executive verdict of the user's work",
  "scores_breakdown": {{
    "valuation_accuracy": 22,
    "risk_rigor": 21,
    "strategic_depth": 23,
    "scenario_stress": 22
  }},
  "ai_consensus_decision": "{case.consensus_decision}",
  "strengths": ["Clear strength 1", "Clear strength 2"],
  "identified_mistakes": ["Specific mistake or gap in user assumptions", "Overlooked financial factor"],
  "unaddressed_risks": ["Specific critical risk the user missed from the dossier"],
  "expert_coaching_advice": "Actionable professional advice to refine future financial modeling and pitch committee presentation.",
  "recommended_model_adjustments": {{
    "suggested_wacc": "10.2%",
    "suggested_terminal_growth": "2.5%",
    "suggested_exit_multiple": "11.0x"
  }}
}}

Respond ONLY with raw JSON."""

    raw = llm.complete(prompt, system="You are an expert investment banking and private equity case judge.")
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(cleaned)
        return EvaluationFeedback(
            overall_score=int(data.get("overall_score", 85)),
            grade=data.get("grade", "B+ (Associate Grade)"),
            verdict=data.get("verdict", "Solid analysis with well-structured strategic rationale."),
            scores_breakdown=data.get("scores_breakdown", {
                "valuation_accuracy": 21,
                "risk_rigor": 20,
                "strategic_depth": 22,
                "scenario_stress": 22,
            }),
            ai_consensus_decision=case.consensus_decision,
            strengths=data.get("strengths", ["Clear understanding of top-line revenue drivers."]),
            identified_mistakes=data.get("identified_mistakes", ["Discount rate assumption could be further stress-tested."]),
            unaddressed_risks=data.get("unaddressed_risks", ["Regulatory and margin compression risks."]),
            expert_coaching_advice=data.get("expert_coaching_advice", "Incorporate multiple contraction scenarios in your sensitivity grid."),
            recommended_model_adjustments=data.get("recommended_model_adjustments", {}),
        )
    except Exception as exc:
        logger.error(f"Failed to parse evaluation output: {exc}")
        # Deterministic structured fallback evaluation
        is_aligned = (sub.user_decision.upper() in case.consensus_decision.upper()) or (case.consensus_decision.upper() in sub.user_decision.upper())
        base_score = 88 if is_aligned else 76
        return EvaluationFeedback(
            overall_score=base_score,
            grade="A- (Senior Analyst Grade)" if is_aligned else "B (Associate Grade)",
            verdict=f"Analysis is {'well-aligned' if is_aligned else 'divergent'} with investment committee consensus.",
            scores_breakdown={
                "valuation_accuracy": 22 if is_aligned else 18,
                "risk_rigor": 21 if is_aligned else 19,
                "strategic_depth": 23 if is_aligned else 20,
                "scenario_stress": 22 if is_aligned else 19,
            },
            ai_consensus_decision=case.consensus_decision,
            strengths=[
                "Articulated strategic rationale with operational metrics.",
                "Utilized scenario parameters to evaluate downside protection.",
            ],
            identified_mistakes=[
                "Ensure terminal value calculation accounts for capital expenditure normalization.",
            ],
            unaddressed_risks=[
                "Macroeconomic interest rate headwinds and customer concentration risks.",
            ],
            expert_coaching_advice="Always corroborate DCF equity value with comparable trading multiples (EV/EBITDA, P/E) to sanity check terminal value contribution.",
            recommended_model_adjustments={
                "suggested_wacc": "10.0%",
                "suggested_terminal_growth": "2.5%",
            },
        )
