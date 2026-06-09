from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_recall: float | None = None


def evaluate_answer(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> EvalResult:
    """
    Run RAGAS evaluation metrics on a single Q&A sample.
    Falls back to heuristic scores if RAGAS is unavailable.
    """
    result = EvalResult(question=question, answer=answer, contexts=contexts)

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness

        sample: dict[str, Any] = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        metrics = [faithfulness, answer_relevancy]
        if ground_truth:
            sample["ground_truth"] = [ground_truth]
            metrics.append(context_recall)

        ds = Dataset.from_dict(sample)
        scores = evaluate(ds, metrics=metrics)

        result.faithfulness = scores.get("faithfulness")
        result.answer_relevancy = scores.get("answer_relevancy")
        result.context_recall = scores.get("context_recall")

        logger.info(
            f"RAGAS eval | faithfulness={result.faithfulness:.2f} "
            f"relevancy={result.answer_relevancy:.2f}"
        )
    except ImportError:
        logger.warning("RAGAS not installed — using heuristic scores")
        result.faithfulness = _heuristic_faithfulness(answer, contexts)
        result.answer_relevancy = _heuristic_relevancy(question, answer)

    return result


def _heuristic_faithfulness(answer: str, contexts: list[str]) -> float:
    """Fraction of answer sentences that overlap with any context chunk."""
    if not contexts or not answer:
        return 0.0
    answer_words = set(answer.lower().split())
    context_words = set(" ".join(contexts).lower().split())
    overlap = len(answer_words & context_words)
    return round(min(overlap / max(len(answer_words), 1), 1.0), 3)


def _heuristic_relevancy(question: str, answer: str) -> float:
    """Simple keyword overlap between question and answer."""
    if not question or not answer:
        return 0.0
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    overlap = len(q_words & a_words)
    return round(min(overlap / max(len(q_words), 1), 1.0), 3)
