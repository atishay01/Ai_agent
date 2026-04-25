"""Process-local metrics registry.

Thread-safe counters for token usage, query counts and cache hits.
Exposed as JSON via the FastAPI ``/metrics`` endpoint so that anyone
eyeballing the backend can see how many tokens the agent has burned.

Per-token rates come from ``settings`` so they can be overridden via
env vars when switching to a priced LLM endpoint.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from config import settings


@dataclass
class Metrics:
    """Mutable, thread-safe counter bag. Single module-level instance."""

    queries_total: int = 0
    queries_failed: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
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

    def snapshot(self) -> dict:
        with self._lock:
            prompt_cost = self.prompt_tokens_total / 1_000_000 * settings.prompt_cost_per_1m_usd
            completion_cost = (
                self.completion_tokens_total / 1_000_000 * settings.completion_cost_per_1m_usd
            )
            return {
                "queries_total": self.queries_total,
                "queries_failed": self.queries_failed,
                "prompt_tokens_total": self.prompt_tokens_total,
                "completion_tokens_total": self.completion_tokens_total,
                "total_tokens": self.prompt_tokens_total + self.completion_tokens_total,
                "estimated_cost_usd": round(prompt_cost + completion_cost, 6),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
            }

    def reset(self) -> None:
        with self._lock:
            self.queries_total = 0
            self.queries_failed = 0
            self.prompt_tokens_total = 0
            self.completion_tokens_total = 0
            self.cache_hits = 0
            self.cache_misses = 0


METRICS = Metrics()
