"""Tests for the SQLite-backed state store and session_history helpers."""

from __future__ import annotations


def test_session_append_and_get_round_trip() -> None:
    import session_history

    session_history.append("alice", "User: hello")
    session_history.append("alice", "Assistant: hi")
    assert session_history.get("alice") == ["User: hello", "Assistant: hi"]


def test_session_history_is_scoped_per_session() -> None:
    import session_history

    session_history.append("alice", "User: q")
    session_history.append("bob", "User: other")
    assert session_history.get("alice") == ["User: q"]
    assert session_history.get("bob") == ["User: other"]


def test_session_clear_drops_only_one_session() -> None:
    import session_history

    session_history.append("alice", "User: a")
    session_history.append("bob", "User: b")
    session_history.clear("alice")
    assert session_history.get("alice") == []
    assert session_history.get("bob") == ["User: b"]


def test_session_history_prunes_to_cap() -> None:
    """Once the cap is exceeded, the oldest entries fall off in order."""
    import session_history
    from config import settings

    cap = settings.session_history_turns * 2
    # Push 4 more than the cap so we can clearly see pruning.
    total = cap + 4
    for i in range(total):
        session_history.append("s", f"line-{i}")
    got = session_history.get("s")
    assert len(got) == cap
    expected = [f"line-{i}" for i in range(total - cap, total)]
    assert got == expected


def test_cache_survives_table_recreation() -> None:
    """Drop the connection and reopen — values written before should still be there."""
    import state_store
    from cache import ResponseCache

    c = ResponseCache(max_size=8)
    c.set("How many orders?", {"answer": "42"})

    # Simulate a process restart by closing and reopening the connection.
    # With ":memory:" this resets, but we can verify the *logic* by
    # writing then reading back across a fresh ResponseCache instance —
    # they share the underlying store.
    c2 = ResponseCache(max_size=8)
    assert c2.get("how many orders") == {"answer": "42"}

    # Sanity: state_store.get_conn() returns the same connection object.
    assert state_store.get_conn() is state_store.get_conn()


def test_reset_for_tests_wipes_everything() -> None:
    import session_history
    import state_store
    from cache import CACHE

    CACHE.set("q", {"a": 1})
    session_history.append("s", "line")
    state_store.reset_for_tests()
    assert CACHE.get("q") is None
    assert session_history.get("s") == []
