"""Retrieval-quality and answer-groundedness scoring. Local heuristics — no external model. ndcg_at_k is exact math. context_precision/recall approximate RAGAS. groundedness_score is token-overlap + negation/number checks."""
from __future__ import annotations
import math
import re
from typing import Iterable, Sequence
_WORD_RE = re.compile(r"[a-z0-9]+")
def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))
def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        total += (2**rel - 1) / math.log2(i + 2)
    return total
def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    if k <= 0 or not relevances:
        return 0.0
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal
def context_precision(retrieved_relevant: Sequence[bool], k: int | None = None) -> float:
    items = list(retrieved_relevant[:k] if k is not None else retrieved_relevant)
    if not items:
        return 0.0
    return sum(1 for x in items if x) / len(items)
def context_recall(retrieved_ids: Iterable[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = set(retrieved_ids)
    return len(retrieved & relevant) / len(relevant)
_NEGATIONS = {"not", "no", "never", "none", "without", "cannot", "can't", "won't", "doesn't", "don't", "isn't", "aren't", "wasn't", "weren't",}
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on", "for", "with", "and", "or", "it", "its", "this", "that", "these", "those", "as", "at", "by", "from", "has", "have", "had", "will", "would", "can", "could", "should", "may", "might", "do", "does", "did", "i", "you", "he", "she", "we", "they", "them", "his", "her", "their", "our", "your", "but", "if", "then", "than", "so", "such", "into", "about", "over", "under", "per",}
def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOPWORDS}
def _sentence_support(sentence: str, context: str) -> float:
    sent_tokens = _content_tokens(sentence)
    if not sent_tokens:
        return 1.0
    ctx_tokens = _content_tokens(context)
    return len(sent_tokens & ctx_tokens) / len(sent_tokens)
def _has_contradiction(sentence: str, context: str) -> bool:
    sent_tokens = _tokens(sentence)
    ctx_tokens = _tokens(context)
    sent_numbers = set(_NUMBER_RE.findall(sentence))
    ctx_numbers = set(_NUMBER_RE.findall(context))
    if sent_numbers and not sent_numbers <= ctx_numbers:
        return True
    if sent_tokens & _NEGATIONS and not (ctx_tokens & _NEGATIONS):
        return True
    return False
def groundedness_score(answer: str, context: str) -> dict:
    sentences = _sentences(answer)
    if not sentences:
        return {"score": 0.0, "supported": 0, "total_sentences": 0, "per_sentence": [],}
    per_sentence = []
    supported = 0
    total_support = 0.0
    for sentence in sentences:
        support = _sentence_support(sentence, context)
        contradiction = _has_contradiction(sentence, context)
        if contradiction:
            support = 0.0
        if support >= 0.5:
            supported += 1
        total_support += support
        per_sentence.append({"sentence": sentence, "support": round(support, 3), "contradiction": contradiction,})
    return {"score": round(total_support / len(sentences), 3), "supported": supported, "total_sentences": len(sentences), "per_sentence": per_sentence,}
