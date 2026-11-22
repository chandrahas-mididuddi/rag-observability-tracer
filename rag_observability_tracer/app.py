"""FastAPI dashboard: GET / HTML dashboard, GET /health, POST /ask, GET /traces, GET /traces/{id}, GET /metrics/summary, GET /evals, GET /evals/{run_id}, POST /evals/run. Run: uvicorn rag_observability_tracer.app:app --reload"""
from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from .eval_runner import run_eval
from .pipeline import DemoRAGPipeline
from .store import TraceStore
DB_PATH = os.environ.get("TRACES_DB", "traces.db")
app = FastAPI(title="RAG Observability Tracer", description="Trace, evaluate and inspect a demo RAG pipeline.", version="0.1.0",)
_pipeline = DemoRAGPipeline()
_store = TraceStore(DB_PATH)
def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(pct / 100 * len(ordered))
    return round(ordered[max(0, min(rank, len(ordered)) - 1)], 1)
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
class AskResponse(BaseModel):
    question: str
    answer: str
    chunks: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    total_duration_ms: Optional[float]
    trace_id: str
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    result = _pipeline.answer(payload.question)
    _store.save_trace(result.trace)
    return AskResponse(question=result.question, answer=result.answer, chunks=[c.to_dict() for c in result.chunks], input_tokens=result.input_tokens, output_tokens=result.output_tokens, cost_usd=result.cost_usd, total_duration_ms=result.trace.total_duration_ms, trace_id=result.trace.trace_id,)
@app.get("/traces")
def list_traces(limit: int = 50) -> list[dict[str, Any]]:
    return _store.list_traces(limit=limit)
@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, Any]:
    trace = _store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace
@app.get("/metrics/summary")
def metrics_summary() -> dict[str, Any]:
    traces = _store.list_traces(limit=1000)
    latencies = [t["total_duration_ms"] for t in traces if t.get("total_duration_ms") is not None]
    costs = []
    for t in traces:
        full = _store.get_trace(t["trace_id"])
        if full and full["metadata"].get("cost_usd") is not None:
            costs.append(float(full["metadata"]["cost_usd"]))
    return {"trace_count": _store.trace_count(), "latency_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95), "mean": round(sum(latencies) / len(latencies), 1) if latencies else None, "n": len(latencies),}, "cost_usd": {"mean_per_query": round(sum(costs) / len(costs), 6) if costs else None, "total": round(sum(costs), 6), "n": len(costs),}, "slowest_traces": _store.slowest_traces(limit=5),}
@app.get("/evals")
def list_evals(limit: int = 20) -> list[dict[str, Any]]:
    return _store.list_eval_runs(limit=limit)
@app.get("/evals/{run_id}")
def get_eval(run_id: str) -> dict[str, Any]:
    results = _store.get_eval_results(run_id)
    if not results:
        raise HTTPException(status_code=404, detail="eval run not found")
    return {"run_id": run_id, "results": results}
@app.post("/evals/run")
def run_eval_endpoint() -> dict[str, Any]:
    report = run_eval(pipeline=_pipeline, store=_store)
    return {"run_id": report["run_id"], "aggregates": report["aggregates"], "results": [{k: v for k, v in r.items() if k != "groundedness_detail"} for r in report["results"]],}
DASHBOARD_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>RAG Observability Tracer</title><style>body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; } h1 { font-size: 1.6rem; } .card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; min-width: 160px; } table { border-collapse: collapse; width: 100%; } th, td { border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; } input[type=text] { width: 70%; padding: 0.5rem; } button { padding: 0.5rem 1rem; cursor: pointer; } pre { background: #f6f6f6; padding: 1rem; overflow-x: auto; white-space: pre-wrap; } .muted { color: #666; }</style></head><body><h1>RAG Observability Tracer</h1><p class="muted">Demo RAG pipeline (TF-IDF retrieval + mock LLM) with tracing, retrieval scoring, groundedness checks and cost/latency metrics.</p><h2>Ask a question</h2><input id="q" type="text" placeholder="e.g. What does the Team plan cost per month?"><button onclick="ask()">Ask</button><pre id="answer">No question asked yet.</pre><h2>Metrics</h2><div id="metrics"><p>Loading…</p></div><h2>Slowest traces</h2><table id="slowest"></table><h2>Recent traces</h2><table id="traces"></table><script>async function load() { const m = await (await fetch('/metrics/summary')).json(); document.getElementById('metrics').innerHTML = '<div>Traces: <b>'+m.trace_count+'</b> | p50: <b>'+m.latency_ms.p50+' ms</b> | p95: <b>'+m.latency_ms.p95+' ms</b> | Mean cost: <b>$'+m.cost_usd.mean_per_query+'</b></div>'; const row = t => '<tr><td>'+t.trace_id.slice(0,8)+'…</td><td>'+t.question+'</td><td>'+t.total_duration_ms+'</td></tr>'; const traces = await (await fetch('/traces?limit=10')).json(); document.getElementById('traces').innerHTML = traces.map(row).join(''); } async function ask() { const q = document.getElementById('q').value; if (!q) return; document.getElementById('answer').textContent = 'Thinking…'; const r = await fetch('/ask', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: q})}); const d = await r.json(); document.getElementById('answer').textContent = d.answer + '\\n\\n— ' + d.total_duration_ms + ' ms · $' + d.cost_usd + ' · trace ' + d.trace_id.slice(0,8) + '…'; load(); } load();</script></body></html>"""
@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML
