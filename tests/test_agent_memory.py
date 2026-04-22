"""Tests for the ask() orchestration — cache, session memory, steps."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class _FakeAction:
    def __init__(self, tool: str, tool_input: str) -> None:
        self.tool = tool
        self.tool_input = tool_input


@pytest.fixture(autouse=True)
def _reset_state():
    import agent
    from cache import CACHE
    from metrics import METRICS

    CACHE.clear()
    METRICS.reset()
    agent._history.clear()
    yield
    CACHE.clear()
    METRICS.reset()
    agent._history.clear()


def _fake_agent(answer: str = "answer", steps=None) -> MagicMock:
    fake = MagicMock()
    fake.invoke.return_value = {
        "output": answer,
        "intermediate_steps": steps or [],
    }
    return fake


def test_stateless_call_populates_cache() -> None:
    import agent

    fake = _fake_agent("42 orders.")
    with patch.object(agent, "get_agent", return_value=fake):
        first = agent.ask("How many orders?")
        second = agent.ask("How many orders?")

    assert first["answer"] == "42 orders."
    assert first["cached"] is False
    assert second["cached"] is True
    assert fake.invoke.call_count == 1


def test_cache_normalizes_case_and_whitespace() -> None:
    import agent

    fake = _fake_agent("42")
    with patch.object(agent, "get_agent", return_value=fake):
        agent.ask("How many orders?")
        again = agent.ask("  how   MANY orders?  ")

    assert again["cached"] is True
    assert fake.invoke.call_count == 1


def test_session_history_prepends_prior_turns_on_followup() -> None:
    """A *different* follow-up question in the same session must see prior turns."""
    import agent

    fake = _fake_agent("first answer")
    with patch.object(agent, "get_agent", return_value=fake):
        agent.ask("First question?", session_id="s1")
        fake.invoke.return_value = {"output": "second answer", "intermediate_steps": []}
        agent.ask("Different follow-up about SP?", session_id="s1")

    assert fake.invoke.call_count == 2
    second_input = fake.invoke.call_args_list[1].args[0]["input"]
    assert "Prior conversation" in second_input
    assert "first answer" in second_input
    assert "Different follow-up about SP?" in second_input


def test_cache_fires_within_same_session() -> None:
    """Re-asking the same literal question hits the cache even with a session_id."""
    import agent

    fake = _fake_agent("42 orders.")
    with patch.object(agent, "get_agent", return_value=fake):
        r1 = agent.ask("How many orders?", session_id="s1")
        r2 = agent.ask("How many orders?", session_id="s1")

    assert r1["cached"] is False
    assert r2["cached"] is True
    assert fake.invoke.call_count == 1


def test_cache_fires_across_different_sessions() -> None:
    import agent

    fake = _fake_agent("answer")
    with patch.object(agent, "get_agent", return_value=fake):
        agent.ask("Some question?", session_id="alice")
        r = agent.ask("Some question?", session_id="bob")

    assert r["cached"] is True
    assert fake.invoke.call_count == 1


def test_session_history_is_scoped_by_id() -> None:
    import agent

    fake = _fake_agent("alpha")
    with patch.object(agent, "get_agent", return_value=fake):
        agent.ask("Q1", session_id="s1")
        fake.invoke.return_value = {"output": "beta", "intermediate_steps": []}
        agent.ask("Q2", session_id="s2")

    # Session s2's first call must not carry s1's history.
    second_input = fake.invoke.call_args_list[1].args[0]["input"]
    assert "Prior conversation" not in second_input
    assert second_input == "Q2"


def test_clear_session_drops_history() -> None:
    import agent

    fake = _fake_agent("one")
    with patch.object(agent, "get_agent", return_value=fake):
        agent.ask("Q?", session_id="s1")
        agent.clear_session("s1")
        fake.invoke.return_value = {"output": "two", "intermediate_steps": []}
        agent.ask("Q2?", session_id="s1")

    second_input = fake.invoke.call_args_list[1].args[0]["input"]
    assert "Prior conversation" not in second_input


def test_steps_are_serialized() -> None:
    import agent

    steps = [(_FakeAction("sql_db_query", "SELECT 1"), "1")]
    fake = _fake_agent("ok", steps=steps)
    with patch.object(agent, "get_agent", return_value=fake):
        r = agent.ask("q")

    assert len(r["steps"]) == 1
    assert r["steps"][0]["tool"] == "sql_db_query"
    assert r["steps"][0]["tool_input"] == "SELECT 1"
    assert r["steps"][0]["observation"] == "1"


def test_metrics_are_recorded_on_success() -> None:
    import agent
    from metrics import METRICS

    fake = _fake_agent("ok")
    with patch.object(agent, "get_agent", return_value=fake):
        agent.ask("q?")

    snap = METRICS.snapshot()
    assert snap["queries_total"] == 1
    assert snap["queries_failed"] == 0


def test_metrics_record_failure_on_exception() -> None:
    import agent
    from metrics import METRICS

    fake = MagicMock()
    fake.invoke.side_effect = RuntimeError("boom")
    with patch.object(agent, "get_agent", return_value=fake), pytest.raises(RuntimeError):
        agent.ask("q?")

    snap = METRICS.snapshot()
    assert snap["queries_total"] == 1
    assert snap["queries_failed"] == 1
