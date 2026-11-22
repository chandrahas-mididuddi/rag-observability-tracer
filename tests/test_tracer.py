"""Tests for the tracer: span recording, nesting, errors, decorators."""
from rag_observability_tracer.tracer import Tracer, trace_span, traced
def test_span_records_duration_and_attributes():
    tracer = Tracer()
    with tracer.span("test.op", kind="test", foo="bar") as span:
        span.set_attribute("n", 3)
    assert span.duration_ms is not None and span.duration_ms >= 0
    assert span.attributes["foo"] == "bar"
    assert span.attributes["n"] == 3
    assert span.status == "ok"
    assert span.end_time is not None
def test_nested_spans_have_parent_links():
    tracer = Tracer()
    with tracer.span("outer") as outer:
        with tracer.span("inner") as inner:
            pass
    assert inner.parent_id == outer.span_id
    assert outer.parent_id is None
    assert inner.trace_id == outer.trace_id
def test_span_marks_error_on_exception():
    tracer = Tracer()
    try:
        with tracer.span("failing"):
            raise ValueError("boom")
    except ValueError:
        pass
    trace = tracer.finish_trace(question="q")
    assert trace.spans[0].status == "error"
    assert "boom" in trace.spans[0].attributes["error"]
def test_finish_trace_returns_all_spans():
    tracer = Tracer()
    with tracer.span("a"):
        with tracer.span("b"):
            pass
    trace = tracer.finish_trace(question="hello", answer="world")
    assert trace.question == "hello" and trace.answer == "world" and len(trace.spans) == 2
    assert trace.total_duration_ms is not None and trace.total_duration_ms >= 0
def test_trace_span_decorator():
    tracer = Tracer()
    @trace_span(tracer, "decorated.op", kind="test")
    def work():
        return 42
    assert work() == 42
    trace = tracer.finish_trace(question="q")
    assert [s.name for s in trace.spans] == ["decorated.op"]
def test_traced_decorator_exposes_tracer():
    @traced("my.func")
    def work():
        return "done"
    assert work() == "done"
    trace = work.tracer.finish_trace(question="q")
    assert trace.spans[0].name == "my.func"
def test_span_events_recorded():
    tracer = Tracer()
    with tracer.span("ev") as span:
        span.add_event("cache_miss", key="abc")
    assert span.events[0]["name"] == "cache_miss"
    assert span.events[0]["attributes"]["key"] == "abc"
