"""Demo RAG pipeline wired through the tracer. Corpus: 20 docs about fictional LumenDesk. Retrieval: TF-IDF cosine, top-k. Rerank: 0.7*TF-IDF + 0.3*keyword overlap. Generation: MockLLM extractive, abstains when context does not cover question. Cost: tokens ~ words/0.75, priced $0.50/$1.50 per 1M in/out."""
from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .tracer import Tracer
DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_INPUT_PRICE_PER_MTOK = 0.50
DEFAULT_OUTPUT_PRICE_PER_MTOK = 1.50
@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str
    score: float = 0.0
    rerank_score: float = 0.0
    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "title": self.title, "text": self.text, "score": round(self.score, 4), "rerank_score": round(self.rerank_score, 4),}
@dataclass
class RAGResult:
    question: str
    answer: str
    chunks: list[Chunk] = field(default_factory=list)
    prompt: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    trace: Any = None
    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "answer": self.answer, "chunks": [c.to_dict() for c in self.chunks], "input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "cost_usd": round(self.cost_usd, 6), "total_duration_ms": (self.trace.total_duration_ms if self.trace else None),}
def estimate_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, math.ceil(words / 0.75))
def estimate_cost(input_tokens: int, output_tokens: int, input_price: float = DEFAULT_INPUT_PRICE_PER_MTOK, output_price: float = DEFAULT_OUTPUT_PRICE_PER_MTOK) -> float:
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
class MockLLM:
    def __init__(self, max_sentences: int = 3, min_overlap: int = 1) -> None:
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap
    @staticmethod
    def _tokens(text: str) -> set[str]:
        stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "in", "on", "for", "with", "and", "or", "it", "this", "that", "what", "how", "does", "do", "can", "i", "you",}
        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in stopwords}
    def generate(self, question: str, context_chunks: list[Chunk]) -> str:
        q_tokens = self._tokens(question)
        scored: list[tuple[int, str]] = []
        for chunk in context_chunks:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text.strip()):
                sentence = sentence.strip()
                if not sentence:
                    continue
                overlap = len(q_tokens & self._tokens(sentence))
                if overlap >= self.min_overlap:
                    scored.append((overlap, sentence))
        if not scored:
            return "I don't know based on the provided documents. The retrieved context does not cover this question."
        scored.sort(key=lambda item: -item[0])
        seen: set[str] = set()
        sentences: list[str] = []
        for _, sentence in scored:
            if sentence not in seen:
                seen.add(sentence)
                sentences.append(sentence)
            if len(sentences) >= self.max_sentences:
                break
        return " ".join(sentences)
class DemoRAGPipeline:
    def __init__(self, corpus_path: str | Path | None = None, top_k: int = 5, rerank_k: int = 3, tracer: Optional[Tracer] = None, llm: Optional[MockLLM] = None, input_price: float = DEFAULT_INPUT_PRICE_PER_MTOK, output_price: float = DEFAULT_OUTPUT_PRICE_PER_MTOK,) -> None:
        self.top_k = top_k
        self.rerank_k = rerank_k
        self.tracer = tracer or Tracer()
        self.llm = llm or MockLLM()
        self.input_price = input_price
        self.output_price = output_price
        corpus_path = Path(corpus_path) if corpus_path else DATA_DIR / "corpus.json"
        with open(corpus_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        self.documents = [{"doc_id": d["id"], "title": d["title"], "text": d["text"]} for d in raw]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([d["text"] for d in self.documents])
    def retrieve(self, question: str, top_k: Optional[int] = None) -> list[Chunk]:
        k = top_k or self.top_k
        with self.tracer.span("retrieval", kind="retrieval") as span:
            with self.tracer.span("query_embedding", kind="embedding") as emb:
                query_vec = self._vectorizer.transform([question])
                emb.set_attribute("model", "tfidf")
                emb.set_attribute("vocab_size", len(self._vectorizer.vocabulary_))
            sims = cosine_similarity(query_vec, self._matrix)[0]
            ranked = sorted(range(len(self.documents)), key=lambda i: -sims[i])[:k]
            chunks = [Chunk(doc_id=self.documents[i]["doc_id"], title=self.documents[i]["title"], text=self.documents[i]["text"], score=float(sims[i]),) for i in ranked]
            span.set_attribute("top_k", k)
            span.set_attribute("retrieved_doc_ids", [c.doc_id for c in chunks])
            span.set_attribute("scores", [round(c.score, 4) for c in chunks])
        return chunks
    def rerank(self, question: str, chunks: list[Chunk]) -> list[Chunk]:
        with self.tracer.span("rerank", kind="rerank") as span:
            q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
            for chunk in chunks:
                c_tokens = set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
                overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
                chunk.rerank_score = 0.7 * chunk.score + 0.3 * overlap
            ranked = sorted(chunks, key=lambda c: -c.rerank_score)[: self.rerank_k]
            span.set_attribute("rerank_k", self.rerank_k)
            span.set_attribute("reranked_doc_ids", [c.doc_id for c in ranked])
        return ranked
    @staticmethod
    def build_prompt(question: str, chunks: list[Chunk]) -> str:
        context = "\n\n".join(f"[{c.doc_id}] {c.title}: {c.text}" for c in chunks)
        return ("Answer the question using only the context below. If the context does not contain the answer, say you don't know.\n\n" f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
    def answer(self, question: str) -> RAGResult:
        self.tracer.start_trace()
        with self.tracer.span("rag.query", kind="rag_query") as root:
            root.set_attribute("question", question)
            retrieved = self.retrieve(question)
            reranked = self.rerank(question, retrieved)
            with self.tracer.span("prompt_build", kind="prompt") as pspan:
                prompt = self.build_prompt(question, reranked)
                pspan.set_attribute("n_context_chunks", len(reranked))
                pspan.set_attribute("prompt_chars", len(prompt))
            with self.tracer.span("llm.generate", kind="llm") as lspan:
                answer = self.llm.generate(question, reranked)
                lspan.set_attribute("model", "mock-extractive-llm")
                lspan.set_attribute("answer_chars", len(answer))
            input_tokens = estimate_tokens(prompt)
            output_tokens = estimate_tokens(answer)
            cost = estimate_cost(input_tokens, output_tokens, self.input_price, self.output_price)
            root.set_attribute("input_tokens", input_tokens)
            root.set_attribute("output_tokens", output_tokens)
            root.set_attribute("cost_usd", round(cost, 6))
        trace = self.tracer.finish_trace(question=question, answer=answer, metadata={"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": round(cost, 6), "retrieved_doc_ids": [c.doc_id for c in retrieved], "reranked_doc_ids": [c.doc_id for c in reranked],},)
        return RAGResult(question=question, answer=answer, chunks=reranked, prompt=prompt, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=round(cost, 6), trace=trace,)
