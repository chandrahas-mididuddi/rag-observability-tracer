"""Span recording for RAG pipelines. A Tracer collects Span objects for one logical request. Spans form a flat list with parent/child links, each with start/end timestamps, attributes and events. Thread-local."""
from __future__ import annotations
import functools
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional
@dataclass
class Span:
    name: str
    kind: str = "internal"
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: Optional[str] = None
    trace_id: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0
    def set_attribute(self, key: str, value: Any) -> "Span":
        self.attributes[key] = value
        return self
    def add_event(self, name: str, **attributes: Any) -> "Span":
        self.events.append({"name": name, "time": time.time(), "attributes": attributes})
        return self
    def finish(self, status: str = "ok") -> "Span":
        self.end_time = time.time()
        self.status = status
        return self
    def to_dict(self) -> dict[str, Any]:
        return {"span_id": self.span_id, "parent_id": self.parent_id, "trace_id": self.trace_id, "name": self.name, "kind": self.kind, "start_time": self.start_time, "end_time": self.end_time, "duration_ms": self.duration_ms, "attributes": self.attributes, "events": self.events, "status": self.status,}
@dataclass
class Trace:
    trace_id: str
    question: str
    answer: Optional[str] = None
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    @property
    def total_duration_ms(self) -> Optional[float]:
        durations = [s.duration_ms for s in self.spans if s.duration_ms is not None]
        if not durations:
            return None
        roots = [s for s in self.spans if s.parent_id is None]
        if roots and roots[0].duration_ms is not None:
            return roots[0].duration_ms
        return sum(durations)
    def span_by_name(self, name: str) -> Optional[Span]:
        for span in self.spans:
            if span.name == name:
                return span
        return None
    def to_dict(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "question": self.question, "answer": self.answer, "metadata": self.metadata, "created_at": self.created_at, "total_duration_ms": self.total_duration_ms, "spans": [s.to_dict() for s in self.spans],}
class Tracer:
    def __init__(self) -> None:
        self._local = threading.local()
    def _stack(self) -> list[Span]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack
    def _spans(self) -> list[Span]:
        if not hasattr(self._local, "spans"):
            self._local.spans = []
        return self._local.spans
    def _trace_id(self) -> str:
        if not hasattr(self._local, "trace_id"):
            self._local.trace_id = uuid.uuid4().hex
        return self._local.trace_id
    @contextmanager
    def span(self, name: str, kind: str = "internal", **attributes: Any) -> Iterator[Span]:
        stack = self._stack()
        parent = stack[-1] if stack else None
        span = Span(name=name, kind=kind, trace_id=self._trace_id(), parent_id=parent.span_id if parent else None, attributes=dict(attributes),)
        stack.append(span)
        self._spans().append(span)
        try:
            yield span
        except Exception as exc:
            span.set_attribute("error", f"{type(exc).__name__}: {exc}")
            span.finish(status="error")
            raise
        else:
            span.finish()
        finally:
            stack.pop()
    def start_trace(self) -> str:
        self._local.trace_id = uuid.uuid4().hex
        self._local.stack = []
        self._local.spans = []
        return self._local.trace_id
    def finish_trace(self, question: str, answer: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,) -> Trace:
        trace = Trace(trace_id=self._trace_id(), question=question, answer=answer, spans=list(self._spans()), metadata=dict(metadata or {}),)
        self._local.stack = []
        self._local.spans = []
        self._local.trace_id = uuid.uuid4().hex
        return trace
    @property
    def current_span(self) -> Optional[Span]:
        stack = self._stack()
        return stack[-1] if stack else None
def trace_span(tracer: Tracer, name: str, kind: str = "internal"):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            with tracer.span(name, kind=kind):
                return func(*args, **kwargs)
        return wrapper
    return decorator
def traced(name: Optional[str] = None, kind: str = "internal"):
    def decorator(func: Callable) -> Callable:
        tracer = Tracer()
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            with tracer.span(span_name, kind=kind):
                return func(*args, **kwargs)
        wrapper.tracer = tracer
        return wrapper
    return decorator
