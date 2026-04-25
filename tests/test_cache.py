"""Unit tests for the in-memory ResponseCache."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_metrics():
    from metrics import METRICS

    METRICS.reset()
    yield
    METRICS.reset()


def test_hit_and_miss() -> None:
    from cache import ResponseCache
    from metrics import METRICS

    c = ResponseCache(max_size=4)
    assert c.get("how many orders?") is None

    c.set("how many orders?", {"answer": "42"})
    hit = c.get("how many orders?")
    assert hit == {"answer": "42"}

    snap = METRICS.snapshot()
    assert snap["cache_hits"] == 1
    assert snap["cache_misses"] == 1


def test_normalization_matches_whitespace_and_case() -> None:
    from cache import ResponseCache

    c = ResponseCache()
    c.set("How Many ORDERS?", {"answer": "42"})
    assert c.get("   how   many orders?   ") == {"answer": "42"}


def test_normalization_strips_punctuation() -> None:
    """Punctuation differences shouldn't fragment the cache."""
    from cache import ResponseCache

    c = ResponseCache()
    c.set("How many orders?", {"answer": "42"})
    # Different punctuation, same words → should hit the same slot.
    assert c.get("how many orders") == {"answer": "42"}
    assert c.get("How, many orders!") == {"answer": "42"}


def test_lru_eviction() -> None:
    from cache import ResponseCache

    c = ResponseCache(max_size=2)
    c.set("a", {"v": 1})
    c.set("b", {"v": 2})
    c.set("c", {"v": 3})  # evicts "a"

    assert c.get("a") is None
    assert c.get("b") == {"v": 2}
    assert c.get("c") == {"v": 3}


def test_recent_access_protects_from_eviction() -> None:
    from cache import ResponseCache

    c = ResponseCache(max_size=2)
    c.set("a", {"v": 1})
    c.set("b", {"v": 2})
    _ = c.get("a")  # promote "a"
    c.set("c", {"v": 3})  # should evict "b", not "a"

    assert c.get("a") == {"v": 1}
    assert c.get("b") is None


def test_clear_empties_store() -> None:
    from cache import ResponseCache

    c = ResponseCache()
    c.set("a", {"v": 1})
    c.set("b", {"v": 2})
    c.clear()
    assert len(c) == 0
    assert c.get("a") is None


def test_set_returns_copy_not_reference() -> None:
    from cache import ResponseCache

    c = ResponseCache()
    payload = {"answer": "42"}
    c.set("q", payload)
    payload["answer"] = "mutated"

    got = c.get("q")
    assert got == {"answer": "42"}
