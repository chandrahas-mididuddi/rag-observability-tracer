"""Tests for the SQLite store and the demo pipeline + eval runner."""
import json
import pytest
from fastapi.testclient import TestClient
from rag_observability_tracer.eval_runner import aggregate, load_golden, run_eval
from rag_observability_tracer.pipeline import DemoRAGPipeline, MockLLM
from rag_observability_tracer.store import TraceStore
@pytest.fixture()
def store(tmp_path):
    s = TraceStore(tmp_path / "test.db")
    yield s
    s.close()
@pytest.fixture()
def pipeline():
    return DemoRAGPipeline()
def test_save_and_get_trace(store, pipeline):
    result = pipeline.answer("What is the Team plan price?")
    store.save_trace(result.trace)
    assert store.trace_count() == 1
    loaded = store.get_trace(result.trace.trace_id)
    assert loaded is not None
    assert loaded["question"] == "What is the Team plan price?"
    assert len(loaded["spans"]) >= 5
    names = {s["name"] for s in loaded["spans"]}
    assert {"rag.query", "retrieval", "rerank", "llm.generate"} <= names
def test_get_missing_trace_returns_none(store):
    assert store.get_trace("nope") is None
def test_slowest_traces_ordered(store, pipeline):
    for q in ("What is the Team plan price?", "Is LumenDesk SOC 2 certified?"):
        store.save_trace(pipeline.answer(q).trace)
    slowest = store.slowest_traces(limit=1)
    assert len(slowest) == 1
    assert slowest[0]["total_duration_ms"] is not None
def test_retrieval_returns_pricing_doc_first(pipeline):
    chunks = pipeline.retrieve("What is the monthly price of the Team plan?")
    assert chunks[0].doc_id in {"pricing-01", "pricing-02"}
    assert all(0.0 <= c.score <= 1.0 for c in chunks)
def test_rerank_keeps_top_k(pipeline):
    chunks = pipeline.retrieve("single sign-on SAML", top_k=5)
    ranked = pipeline.rerank("single sign-on SAML", chunks)
    assert len(ranked) == pipeline.rerank_k
    assert ranked[0].rerank_score >= ranked[-1].rerank_score
def test_answer_is_grounded_in_corpus(pipeline):
    result = pipeline.answer("What is the monthly price of the Team plan?")
    assert "$49" in result.answer
    assert result.cost_usd > 0
    assert result.input_tokens > result.output_tokens
    assert result.trace is not None
    assert result.trace.total_duration_ms is not None
def test_mock_llm_abstains_off_topic(pipeline):
    result = pipeline.answer("What is the capital of France?")
    assert "don't know" in result.answer.lower()
def test_mock_llm_never_invents_numbers():
    llm = MockLLM()
    chunks = [type("C", (), {"text": "The Team plan costs $49 per user per month."})()]
    answer = llm.generate("What does the Team plan cost?", chunks)
    assert "$49" in answer and "$99" not in answer
def test_load_golden_has_ten_items():
    items = load_golden("data/golden.jsonl")
    assert len(items) == 10
    assert all("question" in i and "relevant_doc_ids" in i for i in items)
def test_run_eval_produces_report(store, pipeline):
    report = run_eval(pipeline=pipeline, store=store)
    assert report["aggregates"]["n_questions"] == 10
    assert report["aggregates"]["mean_groundedness"] >= 0.5
    assert report["aggregates"]["mean_context_recall"] >= 0.5
    assert len(report["results"]) == 10
    assert all("passed" in r for r in report["results"])
    assert "markdown" in report
    assert "# RAG Eval Report" in report["markdown"]
    runs = store.list_eval_runs()
    assert len(runs) == 1
    assert runs[0]["aggregates"]["n_questions"] == 10
def test_aggregate_empty():
    agg = aggregate([])
    assert agg["n_questions"] == 0 and agg["pass_rate"] == 0.0
def test_api_ask_and_summary(monkeypatch, tmp_path):
    import rag_observability_tracer.app as app_module
    monkeypatch.setattr(app_module, "_store", TraceStore(tmp_path / "api.db"))
    client = TestClient(app_module.app)
    resp = client.post("/ask", json={"question": "What is the Team plan price?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "$49" in body["answer"] and body["trace_id"]
    summary = client.get("/metrics/summary").json()
    assert summary["trace_count"] == 1
    assert summary["latency_ms"]["p50"] is not None
    trace = client.get(f"/traces/{body['trace_id']}").json()
    assert trace["question"] == "What is the Team plan price?"
    assert client.get("/traces/does-not-exist").status_code == 404
    assert client.get("/health").json() == {"status": "ok"}
def test_api_dashboard_html():
    from fastapi.testclient import TestClient
    import rag_observability_tracer.app as app_module
    client = TestClient(app_module.app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "RAG Observability Tracer" in resp.text
