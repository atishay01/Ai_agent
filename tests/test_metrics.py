"""Unit tests for the METRICS counter registry."""

from __future__ import annotations

import threading


def test_record_query_accumulates() -> None:
    from metrics import Metrics

    m = Metrics()
    m.record_query(100, 50)
    m.record_query(25, 10)
    snap = m.snapshot()
    assert snap["queries_total"] == 2
    assert snap["queries_failed"] == 0
    assert snap["prompt_tokens_total"] == 125
    assert snap["completion_tokens_total"] == 60
    assert snap["total_tokens"] == 185


def test_record_query_failed_flag() -> None:
    from metrics import Metrics

    m = Metrics()
    m.record_query(10, 5, failed=True)
    m.record_query(10, 5, failed=False)
    snap = m.snapshot()
    assert snap["queries_total"] == 2
    assert snap["queries_failed"] == 1


def test_cost_math() -> None:
    from config import settings
    from metrics import Metrics

    m = Metrics()
    m.record_query(1_000_000, 1_000_000)
    snap = m.snapshot()
    expected = round(settings.prompt_cost_per_1m_usd + settings.completion_cost_per_1m_usd, 6)
    assert snap["estimated_cost_usd"] == expected


def test_record_cache_hits_and_misses() -> None:
    from metrics import Metrics

    m = Metrics()
    m.record_cache(hit=True)
    m.record_cache(hit=True)
    m.record_cache(hit=False)
    snap = m.snapshot()
    assert snap["cache_hits"] == 2
    assert snap["cache_misses"] == 1


def test_reset_zeroes_all_counters() -> None:
    from metrics import Metrics

    m = Metrics()
    m.record_query(10, 20)
    m.record_cache(hit=True)
    m.reset()
    snap = m.snapshot()
    assert snap["queries_total"] == 0
    assert snap["prompt_tokens_total"] == 0
    assert snap["completion_tokens_total"] == 0
    assert snap["cache_hits"] == 0
    assert snap["cache_misses"] == 0


def test_snapshot_keys() -> None:
    from metrics import Metrics

    m = Metrics()
    snap = m.snapshot()
    for key in (
        "queries_total",
        "queries_failed",
        "prompt_tokens_total",
        "completion_tokens_total",
        "total_tokens",
        "estimated_cost_usd",
        "cache_hits",
        "cache_misses",
        "guardrail_blocks",
        "sql_errors",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "latency_samples",
        "cache_size",
        "session_count",
    ):
        assert key in snap


def test_record_latency_percentiles() -> None:
    from metrics import Metrics

    m = Metrics()
    # 1..100ms — p50≈50, p95≈95, p99≈99 with nearest-rank.
    for ms in range(1, 101):
        m.record_latency(ms)
    snap = m.snapshot()
    assert snap["latency_samples"] == 100
    # Allow a small slack for rounding behaviour around the rank index.
    assert 49 <= snap["p50_latency_ms"] <= 51
    assert 94 <= snap["p95_latency_ms"] <= 96
    assert 98 <= snap["p99_latency_ms"] <= 100


def test_latency_percentiles_zero_when_no_samples() -> None:
    from metrics import Metrics

    m = Metrics()
    snap = m.snapshot()
    assert snap["latency_samples"] == 0
    assert snap["p50_latency_ms"] == 0
    assert snap["p95_latency_ms"] == 0


def test_guardrail_and_sql_error_counters() -> None:
    from metrics import Metrics

    m = Metrics()
    m.record_guardrail_block()
    m.record_guardrail_block()
    m.record_sql_error()
    snap = m.snapshot()
    assert snap["guardrail_blocks"] == 2
    assert snap["sql_errors"] == 1


def test_reset_clears_new_counters_and_latencies() -> None:
    from metrics import Metrics

    m = Metrics()
    m.record_latency(100)
    m.record_guardrail_block()
    m.record_sql_error()
    m.reset()
    snap = m.snapshot()
    assert snap["guardrail_blocks"] == 0
    assert snap["sql_errors"] == 0
    assert snap["latency_samples"] == 0


def test_snapshot_includes_state_store_sizes() -> None:
    """cache_size / session_count come from the SQLite store, not the lock."""
    from cache import CACHE
    from metrics import Metrics
    from session_history import append

    CACHE.set("hello world", {"answer": "hi"})
    append("sess-1", "User: hi")
    append("sess-2", "User: yo")

    snap = Metrics().snapshot()
    assert snap["cache_size"] >= 1
    assert snap["session_count"] >= 2


def test_thread_safety() -> None:
    from metrics import Metrics

    m = Metrics()

    def hammer() -> None:
        for _ in range(100):
            m.record_query(1, 2)
            m.record_cache(hit=True)

    threads = [threading.Thread(target=hammer) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = m.snapshot()
    assert snap["queries_total"] == 1000
    assert snap["prompt_tokens_total"] == 1000
    assert snap["completion_tokens_total"] == 2000
    assert snap["cache_hits"] == 1000
