from __future__ import annotations

from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

import operator


class AgentState(TypedDict):
    query: str
    plan: str
    retrieved_chunks: list[dict]
    analysis: str
    answer: str
    sources: list[str]
    agent_trace: Annotated[list[str], operator.add]
    intent: Literal["simple_qa", "deep_analysis", "comparison", "report", "unknown"]
    error: str
    iteration: int
