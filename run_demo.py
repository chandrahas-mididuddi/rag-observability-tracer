"""Demo entry point: ask a question, print the traced answer, run the eval. Usage: python run_demo.py "What is the Team plan price?" / python run_demo.py --eval"""
from __future__ import annotations
import argparse
import json
from rag_observability_tracer.eval_runner import run_eval
from rag_observability_tracer.pipeline import DemoRAGPipeline
from rag_observability_tracer.store import TraceStore
def main() -> None:
    parser = argparse.ArgumentParser(description="RAG observability tracer demo.")
    parser.add_argument("question", nargs="?", default="What is the Team plan price?")
    parser.add_argument("--eval", action="store_true", help="run the golden eval set")
    parser.add_argument("--db", default="traces.db")
    args = parser.parse_args()
    pipeline = DemoRAGPipeline()
    store = TraceStore(args.db)
    if args.eval:
        report = run_eval(pipeline=pipeline, store=store)
        print(report["markdown"])
        return
    result = pipeline.answer(args.question)
    store.save_trace(result.trace)
    print(f"Q: {result.question}\n")
    print(f"A: {result.answer}\n")
    print(f"Retrieved: {[c.doc_id for c in result.chunks]}")
    print(f"Tokens in/out: {result.input_tokens}/{result.output_tokens}  Cost: ${result.cost_usd:.6f}  Latency: {result.trace.total_duration_ms:.1f} ms")
    print(f"Trace ID: {result.trace.trace_id}")
    print("\nSpans:")
    for span in result.trace.spans:
        print(f"  - {span.name:20s} {span.duration_ms:7.1f} ms  [{span.kind}]")
    print("\nFull trace JSON:")
    print(json.dumps(result.trace.to_dict(), indent=2, default=str))
if __name__ == "__main__":
    main()
