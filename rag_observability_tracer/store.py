"""SQLite storage for traces and eval runs. Tables: traces, spans, eval_runs, eval_results. Connection check_same_thread=False guarded by a lock."""
from __future__ import annotations
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional
from .tracer import Trace
_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT UNIQUE NOT NULL, question TEXT NOT NULL, answer TEXT, created_at REAL NOT NULL, total_duration_ms REAL, metadata_json TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS spans (id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL, span_id TEXT NOT NULL, parent_id TEXT, name TEXT NOT NULL, kind TEXT NOT NULL, start_time REAL NOT NULL, end_time REAL, duration_ms REAL, status TEXT NOT NULL DEFAULT 'ok', attributes_json TEXT NOT NULL DEFAULT '{}', events_json TEXT NOT NULL DEFAULT '[]');
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE TABLE IF NOT EXISTS eval_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT UNIQUE NOT NULL, started_at REAL NOT NULL, finished_at REAL, dataset TEXT NOT NULL, n_questions INTEGER NOT NULL, aggregates_json TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS eval_results (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, question TEXT NOT NULL, answer TEXT, context_precision REAL, context_recall REAL, ndcg_at_k REAL, groundedness REAL, latency_ms REAL, cost_usd REAL, passed INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(run_id);
"""
class TraceStore:
    def __init__(self, path: str | Path = "traces.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
    def save_trace(self, trace: Trace) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("INSERT OR REPLACE INTO traces (trace_id, question, answer, created_at, total_duration_ms, metadata_json) VALUES (?, ?, ?, ?, ?, ?)", (trace.trace_id, trace.question, trace.answer, trace.created_at, trace.total_duration_ms, json.dumps(trace.metadata),),)
            cur.execute("DELETE FROM spans WHERE trace_id = ?", (trace.trace_id,))
            for span in trace.spans:
                cur.execute("INSERT INTO spans (trace_id, span_id, parent_id, name, kind, start_time, end_time, duration_ms, status, attributes_json, events_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (trace.trace_id, span.span_id, span.parent_id, span.name, span.kind, span.start_time, span.end_time, span.duration_ms, span.status, json.dumps(span.attributes), json.dumps(span.events),),)
            self._conn.commit()
    def get_trace(self, trace_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
            if row is None:
                return None
            trace = dict(row)
            trace["metadata"] = json.loads(trace["metadata_json"])
            trace["spans"] = [self._span_row(dict(r)) for r in self._conn.execute("SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time", (trace_id,))]
            return trace
    @staticmethod
    def _span_row(row: dict[str, Any]) -> dict[str, Any]:
        row["attributes"] = json.loads(row.pop("attributes_json"))
        row["events"] = json.loads(row.pop("events_json"))
        return row
    def list_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT trace_id, question, created_at, total_duration_ms FROM traces ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
    def trace_count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM traces").fetchone()
            return int(row["n"])
    def slowest_traces(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT trace_id, question, total_duration_ms, created_at FROM traces WHERE total_duration_ms IS NOT NULL ORDER BY total_duration_ms DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
    def save_eval_run(self, run_id: str, dataset: str, results: list[dict[str, Any]], aggregates: dict[str, Any], started_at: Optional[float] = None,) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("INSERT OR REPLACE INTO eval_runs (run_id, started_at, finished_at, dataset, n_questions, aggregates_json) VALUES (?, ?, ?, ?, ?, ?)", (run_id, started_at or time.time(), time.time(), dataset, len(results), json.dumps(aggregates),),)
            cur.execute("DELETE FROM eval_results WHERE run_id = ?", (run_id,))
            for r in results:
                cur.execute("INSERT INTO eval_results (run_id, question, answer, context_precision, context_recall, ndcg_at_k, groundedness, latency_ms, cost_usd, passed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (run_id, r["question"], r.get("answer"), r.get("context_precision"), r.get("context_recall"), r.get("ndcg_at_k"), r.get("groundedness"), r.get("latency_ms"), r.get("cost_usd"), 1 if r.get("passed") else 0,),)
            self._conn.commit()
    def list_eval_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT run_id, started_at, finished_at, dataset, n_questions, aggregates_json FROM eval_runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["aggregates"] = json.loads(d.pop("aggregates_json"))
                out.append(d)
            return out
    def get_eval_results(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM eval_results WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
            return [dict(r) for r in rows]
    def close(self) -> None:
        with self._lock:
            self._conn.close()
