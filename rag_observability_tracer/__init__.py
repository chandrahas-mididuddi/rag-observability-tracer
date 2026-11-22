"""rag-observability-tracer: tracing, evaluation and cost/latency observability for RAG pipelines."""
from .metrics import (context_precision, context_recall, groundedness_score, ndcg_at_k,)
from .pipeline import DemoRAGPipeline, MockLLM
from .store import TraceStore
from .tracer import Span, Tracer, trace_span, traced
__all__ = ["Tracer", "Span", "trace_span", "traced", "ndcg_at_k", "context_precision", "context_recall", "groundedness_score", "TraceStore", "DemoRAGPipeline", "MockLLM",]
__version__ = "0.1.0"
