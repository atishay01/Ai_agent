"""API endpoint tests.

These tests patch ``agent.ask`` so they run without Postgres or Groq.
They verify the HTTP contract: response shape, headers, auth, rate limit,
and validation behaviour.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    import api  # import lazily so conftest env vars are applied first

    return TestClient(api.app)


# ---------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------
def test_root_ok(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.headers.get("x-trace-id")
    assert r.headers.get("x-latency-ms")


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["version"]


# ---------------------------------------------------------------------
# /query — success path
# ---------------------------------------------------------------------
def test_query_success(client: TestClient) -> None:
    fake = {
        "question": "How many orders?",
        "answer": "42 orders.",
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "cached": False,
        "steps": [{"tool": "sql_db_query", "tool_input": "SELECT 1", "observation": "1"}],
    }
    with patch("api.ask", return_value=fake):
        r = client.post("/query", json={"question": "How many orders?"})

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "42 orders."
    assert body["question"] == "How many orders?"
    # Observability fields added in Phase 1
    assert len(body["trace_id"]) == 12
    assert body["latency_ms"] >= 0
    assert body["timestamp"].endswith("Z") or "+" in body["timestamp"]
    # Phase 3 fields
    assert body["prompt_tokens"] == 123
    assert body["completion_tokens"] == 45
    assert body["cached"] is False
    assert len(body["steps"]) == 1
    assert body["steps"][0]["tool"] == "sql_db_query"


def test_query_forwards_session_id(client: TestClient) -> None:
    fake = {"question": "q", "answer": "a"}
    with patch("api.ask", return_value=fake) as mock_ask:
        r = client.post(
            "/query",
            json={"question": "How many orders?", "session_id": "sess-xyz"},
        )
    assert r.status_code == 200
    assert mock_ask.call_args.kwargs.get("session_id") == "sess-xyz"


def test_metrics_endpoint(client: TestClient) -> None:
    from metrics import METRICS

    METRICS.reset()
    METRICS.record_query(100, 50)
    METRICS.record_cache(hit=True)

    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["queries_total"] == 1
    assert body["prompt_tokens_total"] == 100
    assert body["completion_tokens_total"] == 50
    assert body["total_tokens"] == 150
    assert body["cache_hits"] == 1
    assert "estimated_cost_usd" in body
    METRICS.reset()


def test_delete_session(client: TestClient) -> None:
    with patch("api.clear_session") as mock_clear:
        r = client.delete("/session/abc123")
    assert r.status_code == 204
    mock_clear.assert_called_once_with("abc123")


def test_query_validates_short_input(client: TestClient) -> None:
    r = client.post("/query", json={"question": "hi"})
    assert r.status_code == 422  # pydantic min_length


def test_query_validates_long_input(client: TestClient) -> None:
    r = client.post("/query", json={"question": "x" * 501})
    assert r.status_code == 422


# ---------------------------------------------------------------------
# /query — agent failures
# ---------------------------------------------------------------------
def test_query_rate_limit_message(client: TestClient) -> None:
    """Groq 429 → friendly answer, not 500."""
    with patch("api.ask", side_effect=RuntimeError("rate_limit exceeded")):
        r = client.post("/query", json={"question": "How many orders?"})
    assert r.status_code == 200
    assert "free-tier" in r.json()["answer"].lower()


def test_query_unhandled_error(client: TestClient) -> None:
    with patch("api.ask", side_effect=RuntimeError("boom")):
        r = client.post("/query", json={"question": "How many orders?"})
    assert r.status_code == 500


# ---------------------------------------------------------------------
# Auth (when API_KEY is set)
# ---------------------------------------------------------------------
def test_query_requires_api_key_when_configured(monkeypatch) -> None:
    import importlib

    monkeypatch.setenv("API_KEY", "secret-abc")
    import config

    config.get_settings.cache_clear()
    config.settings = config.get_settings()
    import api

    importlib.reload(api)
    c = TestClient(api.app)

    # Without header → 401
    with patch("api.ask", return_value={"question": "x", "answer": "y"}):
        r = c.post("/query", json={"question": "How many orders?"})
    assert r.status_code == 401

    # With correct header → 200
    with patch("api.ask", return_value={"question": "x", "answer": "y"}):
        r = c.post(
            "/query",
            json={"question": "How many orders?"},
            headers={"X-API-Key": "secret-abc"},
        )
    assert r.status_code == 200

    # Cleanup: restore no-auth state for subsequent tests.
    monkeypatch.setenv("API_KEY", "")
    config.get_settings.cache_clear()
    config.settings = config.get_settings()
    importlib.reload(api)
