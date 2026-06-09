from __future__ import annotations

from src.rag.retriever import _reciprocal_rank_fusion, RetrievedChunk


def _make_chunk(text: str, source: str = "test.txt", score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, score=score)


def test_rrf_merges_and_reranks():
    a = [_make_chunk(f"chunk_a_{i}") for i in range(5)]
    b = [_make_chunk(f"chunk_b_{i}") for i in range(5)]

    fused = _reciprocal_rank_fusion(a, b, weight_a=0.6, weight_b=0.4)
    assert len(fused) == 10
    # First items should have the highest RRF scores
    assert fused[0].score >= fused[-1].score


def test_rrf_deduplicates_overlapping():
    shared = _make_chunk("shared chunk text that overlaps in both lists")
    a = [shared, _make_chunk("unique_a")]
    b = [shared, _make_chunk("unique_b")]

    fused = _reciprocal_rank_fusion(a, b, weight_a=0.5, weight_b=0.5)
    texts = [c.text for c in fused]
    # Shared chunk should appear only once
    assert texts.count(shared.text[:100]) <= 1


def test_rrf_empty_lists():
    fused = _reciprocal_rank_fusion([], [], weight_a=0.5, weight_b=0.5)
    assert fused == []


def test_rrf_single_list():
    a = [_make_chunk("only_in_a")]
    fused = _reciprocal_rank_fusion(a, [], weight_a=1.0, weight_b=0.0)
    assert len(fused) == 1
    assert fused[0].text == "only_in_a"
