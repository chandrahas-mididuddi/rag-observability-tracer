"""Tests for retrieval/groundedness metrics."""
from rag_observability_tracer.metrics import (context_precision, context_recall, groundedness_score, ndcg_at_k,)
def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1.0, 1.0, 0.0], 3) == 1.0
def test_ndcg_worst_ranking_is_low():
    assert ndcg_at_k([0.0, 0.0, 1.0], 3) <= 0.5
    assert ndcg_at_k([0.0, 0.0, 1.0], 3) > 0.0
def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k([0.0, 0.0, 0.0], 3) == 0.0
    assert ndcg_at_k([], 3) == 0.0
def test_ndcg_partial_ranking_between_zero_and_one():
    score = ndcg_at_k([1.0, 0.0, 1.0], 3)
    assert 0.0 < score < 1.0
def test_context_precision():
    assert context_precision([True, True, False]) == 2 / 3
    assert context_precision([]) == 0.0
    assert context_precision([True, False], k=1) == 1.0
def test_context_recall():
    assert context_recall(["a", "b"], ["a", "b", "c"]) == 2 / 3
    assert context_recall(["a"], ["a"]) == 1.0
    assert context_recall([], ["a"]) == 0.0
    assert context_recall(["a"], []) == 0.0
def test_groundedness_supported_answer_scores_high():
    context = "The Team plan costs $49 per user per month when billed annually."
    answer = "The Team plan costs $49 per user per month when billed annually."
    result = groundedness_score(answer, context)
    assert result["score"] == 1.0 and result["supported"] == 1
def test_groundedness_unsupported_answer_scores_low():
    context = "The Team plan costs $49 per user per month when billed annually."
    answer = "The Enterprise plan includes a private jet and a yacht."
    result = groundedness_score(answer, context)
    assert result["score"] < 0.5
def test_groundedness_number_mismatch_is_flagged():
    context = "The Team plan costs $49 per user per month."
    answer = "The Team plan costs $99 per user per month."
    result = groundedness_score(answer, context)
    assert result["per_sentence"][0]["contradiction"] is True
    assert result["score"] == 0.0
def test_groundedness_empty_answer():
    result = groundedness_score("", "some context")
    assert result["score"] == 0.0 and result["total_sentences"] == 0
