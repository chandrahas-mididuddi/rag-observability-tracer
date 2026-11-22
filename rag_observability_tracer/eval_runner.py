"""Eval runner: run a golden Q&A set through the pipeline and score it. A question passes when context recall >= 0.5, nDCG@k >= 0.5 and groundedness >= 0.5."""
from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from .metrics import context_precision, context_recall, groundedness_score, ndcg_at_k
from .pipeline import DemoRAGPipeline
from .store import TraceStore
DATA_DIR = Path(__file__).parent.parent / "data"
def load_golden(path: str | Path) -> list[dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"n_questions": 0, "mean_context_precision": 0.0, "mean_context_recall": 0.0, "mean_ndcg_at_k": 0.0, "mean_groundedness": 0.0, "mean_latency_ms": 0.0, "mean_cost_usd": 0.0, "pass_rate": 0.0,}
    n = len(results)
    return {"n_questions": n, "mean_context_precision": round(sum(r["context_precision"] for r in results) / n, 3), "mean_context_recall": round(sum(r["context_recall"] for r in results) / n, 3), "mean_ndcg_at_k": round(sum(r["ndcg_at_k"] for r in results) / n, 3), "mean_groundedness": round(sum(r["groundedness"] for r in results) / n, 3), "mean_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 1), "mean_cost_usd": round(sum(r["cost_usd"] for r in results) / n, 6), "pass_rate": round(sum(1 for r in results if r["passed"]) / n, 3),}
def to_markdown(report: dict[str, Any]) -> str:
    agg = report["aggregates"]
    lines = ["# RAG Eval Report", "", f"Run: `{report['run_id']}`  |  Dataset: `{report['dataset']}`  |  Questions: {agg['n_questions']}", "", "## Aggregates", "", "| Metric | Value |", "|---|---|", f"| Mean context precision | {agg['mean_context_precision']} |", f"| Mean context recall | {agg['mean_context_recall']} |", f"| Mean nDCG@k | {agg['mean_ndcg_at_k']} |", f"| Mean groundedness | {agg['mean_groundedness']} |", f"| Mean latency (ms) | {agg['mean_latency_ms']} |", f"| Mean cost (USD) | ${agg['mean_cost_usd']:.6f} |", f"| Pass rate | {agg['pass_rate']:.0%} |", "", "## Per-question results", "", "| # | Question | Prec. | Recall | nDCG | Ground. | Latency (ms) | Cost | Pass |", "|---|---|---|---|---|---|---|---|---|",]
    for i, r in enumerate(report["results"], start=1):
        lines.append(f"| {i} | {r['question'][:60]} | {r['context_precision']} | {r['context_recall']} | {r['ndcg_at_k']} | {r['groundedness']} | {r['latency_ms']} | ${r['cost_usd']:.6f} | {'✅' if r['passed'] else '❌'} |")
    lines.append("")
    lines.append("> Groundedness is a local sentence-overlap heuristic (see README), not an NLI-model judgement. Treat it as a triage signal.")
    return "\n".join(lines) + "\n"
def run_eval(golden_path: str | Path | None = None, pipeline: Optional[DemoRAGPipeline] = None, store: Optional[TraceStore] = None, thresholds: Optional[dict[str, float]] = None, top_k: int = 5,) -> dict[str, Any]:
    golden_path = Path(golden_path) if golden_path else DATA_DIR / "golden.jsonl"
    items = load_golden(golden_path)
    pipeline = pipeline or DemoRAGPipeline(top_k=top_k)
    thresholds = {"context_recall": 0.5, "ndcg_at_k": 0.5, "groundedness": 0.5, **(thresholds or {}),}
    run_id = uuid.uuid4().hex[:12]
    started_at = time.time()
    results = []
    for item in items:
        question = item["question"]
        relevant_ids = item.get("relevant_doc_ids", [])
        result = pipeline.answer(question)
        trace = result.trace
        retrieved_ids = [c.doc_id for c in result.chunks]
        # Note: eval uses reranked chunks (top 3) for precision/recall; retrieved set for recall is the full top-5 retrieval — recompute via pipeline.retrieve for the golden check
        full = pipeline.retrieve(question)
        retrieved_ids = [c.doc_id for c in full]
        relevance_flags = [doc_id in relevant_ids for doc_id in retrieved_ids]
        context = "\n".join(c.text for c in result.chunks)
        ground = groundedness_score(result.answer, context)
        latency = trace.total_duration_ms or 0.0
        per_question = {"question": question, "expected_answer": item.get("expected_answer", ""), "answer": result.answer, "retrieved_doc_ids": retrieved_ids, "relevant_doc_ids": relevant_ids, "context_precision": round(context_precision(relevance_flags), 3), "context_recall": round(context_recall(retrieved_ids, relevant_ids), 3), "ndcg_at_k": round(ndcg_at_k([1.0 if f else 0.0 for f in relevance_flags], top_k), 3), "groundedness": ground["score"], "groundedness_detail": ground, "latency_ms": round(latency, 1), "cost_usd": result.cost_usd, "trace_id": trace.trace_id,}
        per_question["passed"] = bool(per_question["context_recall"] >= thresholds["context_recall"] and per_question["ndcg_at_k"] >= thresholds["ndcg_at_k"] and per_question["groundedness"] >= thresholds["groundedness"])
        results.append(per_question)
        if store is not None:
            store.save_trace(trace)
    report = {"run_id": run_id, "dataset": str(golden_path), "started_at": started_at, "results": results, "aggregates": aggregate(results),}
    report["markdown"] = to_markdown(report)
    if store is not None:
        store.save_eval_run(run_id, str(golden_path), results, report["aggregates"], started_at)
    return report
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run the RAG eval suite.")
    parser.add_argument("--golden", default=str(DATA_DIR / "golden.jsonl"))
    parser.add_argument("--db", default="traces.db")
    parser.add_argument("--out", default="eval_report")
    args = parser.parse_args()
    store = TraceStore(args.db)
    report = run_eval(args.golden, store=store)
    json_path = Path(f"{args.out}.json")
    md_path = Path(f"{args.out}.md")
    json_path.write_text(json.dumps({k: v for k, v in report.items() if k != "markdown"}, indent=2), encoding="utf-8",)
    md_path.write_text(report["markdown"], encoding="utf-8")
    print(report["markdown"])
    print(f"Saved {json_path} and {md_path}")
if __name__ == "__main__":
    main()
