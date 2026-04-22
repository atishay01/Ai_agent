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
    from metrics import COMPLETION_COST_PER_1M_USD, PROMPT_COST_PER_1M_USD, Metrics

    m = Metrics()
    m.record_query(1_000_000, 1_000_000)
    snap = m.snapshot()
    expected = round(PROMPT_COST_PER_1M_USD + COMPLETION_COST_PER_1M_USD, 6)
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
    ):
        assert key in snap


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
