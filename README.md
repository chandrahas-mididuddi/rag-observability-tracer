# rag-observability-tracer

Observability and evaluation tracing for RAG pipelines: trace every
retrieval + generation step, score retrieval quality and answer
groundedness, and expose cost/latency metrics through a small dashboard.

A portfolio project. It is **not** a production system and has **no
affiliation with any company** — the corpus is about a fictional product
(LumenDesk), and the "LLM" is a local extractive mock so everything runs
offline.

## What it does

- **Tracer library** (`rag_observability_tracer/tracer.py`) — decorators
  and context managers that record spans with parent/child links,
  attributes, events and error status. Thread-local, so concurrent
  requests don't interleave.
- **Retrieval scoring** (`metrics.py`) — nDCG@k on ranked relevance,
  context precision/recall (RAGAS-style approximations against a golden
  set).
- **Groundedness scoring** (`metrics.py`) — a *local heuristic*: per-sentence
  token overlap against the retrieved context, plus a lightweight
  contradiction check (numbers that appear in the answer but not the
  context, and negations with no support). No external model is used.
  Treat it as a triage signal, not ground truth — see Limitations.
- **Storage** (`store.py`) — SQLite store for traces (with spans) and
  eval runs, safe to use from FastAPI's thread pool.
- **Demo RAG pipeline** (`pipeline.py`) — 20-document in-memory corpus,
  TF-IDF retrieval (scikit-learn), keyword-blend reranking, prompt build
  and a mock extractive LLM that abstains when the context doesn't cover
  the question. Every stage is traced; token counts and cost are
  estimated per query.
- **Eval runner** (`eval_runner.py`) — runs a golden JSONL set through the
  pipeline and writes a JSON + Markdown report with per-question scores
  and aggregates.
- **Dashboard** (`app.py`) — FastAPI API plus a plain HTML page showing
  latency p50/p95, mean cost per query, groundedness, and the slowest
  traces.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py "What is the monthly price of the Team plan?"
python run_demo.py --eval
uvicorn rag_observability_tracer.app:app --reload
docker build -t rag-observability-tracer .
docker run -p 8000:8000 rag-observability-tracer
```

## Example output

```
Q: What is the monthly price of the Team plan?
A: The Team plan costs $49 per user per month when billed annually,
   or $59 per user per month when billed monthly. ...
Retrieved: ['pricing-02', 'billing-01', 'limits-01']
Tokens in/out: 158/50   Cost: $0.000154   Latency: 1.9 ms
```

## Example eval report

Real output of `python run_demo.py --eval` on the bundled 10-question
golden set (`data/golden.jsonl`):

| Metric | Value |
|---|---|
| Mean context precision | 0.22 |
| Mean context recall | 0.95 |
| Mean nDCG@k | 1.0 |
| Mean groundedness | 1.0 |
| Mean latency (ms) | 2.5 |
| Mean cost (USD) | $0.000148 |
| Pass rate | 100% |

Precision is low (0.22) because retrieval returns 5 chunks while only
1–2 are golden-relevant — the relevant chunks are ranked first
(nDCG 1.0), but the extra chunks dilute precision. That is a real,
visible trade-off of top-k=5 retrieval. A pass requires recall ≥ 0.5, nDCG ≥ 0.5
and groundedness ≥ 0.5.

## API reference

| Method & path | Description |
|---|---|
| `GET /` | HTML dashboard |
| `GET /health` | Liveness |
| `POST /ask` `{"question": "..."}` | Run one traced question |
| `GET /traces?limit=50` | Recent traces, newest first |
| `GET /traces/{trace_id}` | One trace with all spans |
| `GET /metrics/summary` | p50/p95 latency, mean cost, slowest traces |
| `GET /evals` | Eval runs |
| `GET /evals/{run_id}` | Per-question results for a run |
| `POST /evals/run` | Run the golden eval set now |

Set `TRACES_DB` to change the SQLite path (default `traces.db`).

## Tests

```bash
pip install -r requirements.txt
pytest -v
```

30 tests covering span recording/nesting/errors, nDCG math, precision/
recall, groundedness (including number-mismatch detection), SQLite
storage, the demo pipeline (including mock-LLM abstention), the eval
runner, and the FastAPI endpoints.

## Limitations

- **Groundedness is a heuristic**, not an NLI model. Token overlap plus
  negation/number checks catch obvious hallucinations but will miss
  subtle contradictions. For production use, replace it with an NLI or
  LLM-as-judge scorer.
- **The LLM is a mock.** It extracts sentences from the retrieved
  context; it cannot synthesise or reason. Latency and cost figures are
  illustrative.
- **Retrieval is TF-IDF**, not dense embeddings — fine for a 20-document
  demo corpus, not a substitute for an embedding index at scale.
- **Cost model is an assumption**: tokens ≈ words / 0.75, priced at
  $0.50 / $1.50 per 1M input/output tokens. Adjust the rates in
  `pipeline.py` to match your model.
- **SQLite is single-writer**; the store is guarded by a lock, but this
  is a local observability store, not a replacement for a tracing
  backend (Jaeger, LangSmith, Arize, etc.) at production volume.

## License

MIT — see [LICENSE](LICENSE).
