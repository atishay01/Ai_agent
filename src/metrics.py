"""Process-local metrics registry.

Thread-safe counters for token usage, query counts, cache hits, latency
percentiles, guardrail-block / SQL-error rates, and live cache + session
size from the SQLite store. Exposed as JSON via the FastAPI ``/metrics``
endpoint so anyone eyeballing the backend can see how the agent is
behaving and whether the recent guardrail/self-repair work is paying off.

Per-token rates come from ``settings`` so they can be overridden via env
vars when switching to a priced LLM endpoint.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from config import settings

# Rolling window for latency percentiles. 1000 samples keeps memory tiny
# while giving stable p50/p95/p99 numbers even under bursty traffic.
_LATENCY_WINDOW = 1000


@dataclass
class Metrics:
    """Mutable, thread-safe counter bag. Single module-level instance."""

    queries_total: int = 0
    queries_failed: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    guardrail_blocks: int = 0
    sql_errors: int = 0
    _latencies: deque = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_query(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        failed: bool = False,
    ) -> None:
        with self._lock:
            self.queries_total += 1
            if failed:
                self.queries_failed += 1
            self.prompt_tokens_total += int(prompt_tokens or 0)
            self.completion_tokens_total += int(completion_tokens or 0)

    def record_cache(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def record_latency(self, ms: int) -> None:
        """Record the wall-clock latency of one /query in milliseconds."""
        with self._lock:
            self._latencies.append(int(ms))

    def record_guardrail_block(self) -> None:
        """Increment when validate_sql rejects an agent-generated query."""
        with self._lock:
            self.guardrail_blocks += 1

    def record_sql_error(self) -> None:
        """Increment when DBAPI raises during query execution."""
        with self._lock:
            self.sql_errors += 1

    def _percentile(self, samples: list[int], p: float) -> int:
        """Nearest-rank percentile. ``samples`` must already be sorted."""
        if not samples:
            return 0
        idx = min(len(samples) - 1, max(0, int(round(p * (len(samples) - 1)))))
        return samples[idx]

    def snapshot(self) -> dict:
        # Take consistent counter + latency snapshot under the lock; do
        # the SQLite read outside it so a slow disk doesn't block writers.
        with self._lock:
            counters = {
                "queries_total": self.queries_total,
                "queries_failed": self.queries_failed,
                "prompt_tokens_total": self.prompt_tokens_total,
                "completion_tokens_total": self.completion_tokens_total,
                "total_tokens": self.prompt_tokens_total + self.completion_tokens_total,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "guardrail_blocks": self.guardrail_blocks,
                "sql_errors": self.sql_errors,
            }
            latencies = sorted(self._latencies)

        prompt_cost = counters["prompt_tokens_total"] / 1_000_000 * settings.prompt_cost_per_1m_usd
        completion_cost = (
            counters["completion_tokens_total"] / 1_000_000 * settings.completion_cost_per_1m_usd
        )

        # Live state-store stats. Imported lazily so ``Metrics`` stays
        # standalone-testable and we avoid an import cycle.
        try:
            from state_store import transaction

            with transaction() as conn:
                cache_size = conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0]
                session_count = conn.execute(
                    "SELECT COUNT(DISTINCT session_id) FROM session_history"
                ).fetchone()[0]
        except Exception:
            cache_size = 0
            session_count = 0

        return {
            **counters,
            "estimated_cost_usd": round(prompt_cost + completion_cost, 6),
            "p50_latency_ms": self._percentile(latencies, 0.50),
            "p95_latency_ms": self._percentile(latencies, 0.95),
            "p99_latency_ms": self._percentile(latencies, 0.99),
            "latency_samples": len(latencies),
            "cache_size": cache_size,
            "session_count": session_count,
        }

    def reset(self) -> None:
        with self._lock:
            self.queries_total = 0
            self.queries_failed = 0
            self.prompt_tokens_total = 0
            self.completion_tokens_total = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.guardrail_blocks = 0
            self.sql_errors = 0
            self._latencies.clear()


METRICS = Metrics()
