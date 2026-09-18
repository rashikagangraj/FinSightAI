from __future__ import annotations

from datetime import datetime
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.auth import require_api_key
from src.cases.case_engine import generate_ai_case, get_case_by_id, list_all_cases
from src.cases.evaluator import evaluate_user_case_submission
from src.cases.models import CaseStudy, EvaluationFeedback, LeaderboardEntry, ScenarioParams, SimulationResult, UserSubmission
from src.cases.simulator import run_financial_simulation
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/cases", tags=["case-studies"], dependencies=[Depends(require_api_key)])

# In-memory leaderboard store
_LEADERBOARD_STORE: list[LeaderboardEntry] = [
    LeaderboardEntry(
        user_name="Sarah Jenkins",
        case_id="case-tech-saas-mna",
        case_title="Project CloudPeak: $4.2B SaaS M&A",
        score=94,
        grade="A+ (Principal Grade)",
        decision="BUY",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    ),
    LeaderboardEntry(
        user_name="Alex Rivera",
        case_id="case-semiconductor-dcf",
        case_title="NovaSilicon: AI ASIC Equity Research",
        score=91,
        grade="A (VP Grade)",
        decision="STRONG BUY",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    ),
    LeaderboardEntry(
        user_name="Marcus Vance",
        case_id="case-ev-cap-allocation",
        case_title="VoltDynamics: $12B Gigafactory Expansion",
        score=87,
        grade="A- (Senior Analyst)",
        decision="INVEST",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    ),
]


class GenerateCaseRequest(BaseModel):
    industry: str = "Enterprise AI & Cloud"
    topic: str = "M&A"
    difficulty: str = "Intermediate"


class SimulateRequest(BaseModel):
    case_id: str
    scenario_params: ScenarioParams


@router.get("/", response_model=list[CaseStudy])
async def get_cases(
    topic: str | None = Query(None, description="Filter by topic (e.g. M&A, DCF Valuation)"),
    difficulty: str | None = Query(None, description="Filter by difficulty (Beginner, Intermediate, Advanced, Expert)"),
) -> list[CaseStudy]:
    return list_all_cases(topic_filter=topic, difficulty_filter=difficulty)


@router.get("/{case_id}", response_model=CaseStudy)
async def get_case(case_id: str) -> CaseStudy:
    case = get_case_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case study '{case_id}' not found.")
    return case


@router.post("/generate", response_model=CaseStudy)
async def generate_case(req: GenerateCaseRequest) -> CaseStudy:
    try:
        new_case = generate_ai_case(industry=req.industry, topic=req.topic, difficulty=req.difficulty)
        return new_case
    except Exception as exc:
        logger.error(f"Error generating case: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to generate case: {str(exc)}")


@router.post("/simulate", response_model=SimulationResult)
async def simulate_case(req: SimulateRequest) -> SimulationResult:
    case = get_case_by_id(req.case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case study '{req.case_id}' not found.")
    
    res = run_financial_simulation(
        historical=case.historical_financials,
        params=req.scenario_params,
    )
    return res


@router.post("/evaluate", response_model=EvaluationFeedback)
async def evaluate_case(submission: UserSubmission) -> EvaluationFeedback:
    try:
        feedback = evaluate_user_case_submission(submission)
        return feedback
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Evaluation error: {exc}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(exc)}")


@router.get("/leaderboard/list", response_model=list[LeaderboardEntry])
async def get_leaderboard() -> list[LeaderboardEntry]:
    return sorted(_LEADERBOARD_STORE, key=lambda x: x.score, reverse=True)


@router.post("/leaderboard/add", response_model=LeaderboardEntry)
async def add_leaderboard_entry(entry: LeaderboardEntry) -> LeaderboardEntry:
    _LEADERBOARD_STORE.insert(0, entry)
    return entry
