"""Tests for the eval harness — exercise scoring + CLI wiring without
hitting Groq/Postgres by injecting a fake ``ask`` function."""

from __future__ import annotations

import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent / "eval"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))


def test_score_case_passes_when_all_substrings_found() -> None:
    from run_eval import score_case

    assert score_case(["99,441"], "There are 99,441 orders.") == []


def test_score_case_is_case_insensitive() -> None:
    from run_eval import score_case

    assert score_case(["Rio de Janeiro"], "the capital is rio de janeiro.") == []


def test_score_case_reports_missing_substrings() -> None:
    from run_eval import score_case

    missing = score_case(["health_beauty", "toys"], "health_beauty is top")
    assert missing == ["toys"]


def test_run_case_passes_on_matching_answer() -> None:
    from run_eval import run_case

    def fake_ask(q, session_id=None):
        return {"answer": "There are 99,441 orders.", "prompt_tokens": 10, "completion_tokens": 5}

    case = {"id": "orders_count", "question": "How many?", "contains": ["99,441"]}
    r = run_case(case, fake_ask)
    assert r.passed is True
    assert r.missing == []
    assert r.prompt_tokens == 10
    assert r.completion_tokens == 5
    assert r.slow is False


def test_run_case_flags_slow_result() -> None:
    import time

    from run_eval import run_case

    def slow_ask(q, session_id=None):
        time.sleep(0.05)
        return {"answer": "ok", "prompt_tokens": 0, "completion_tokens": 0}

    case = {
        "id": "slow",
        "question": "anything?",
        "contains": ["ok"],
        "max_latency_ms": 1,
    }
    r = run_case(case, slow_ask)
    assert r.passed is True
    assert r.slow is True


def test_run_case_captures_exception() -> None:
    from run_eval import run_case

    def boom(q, session_id=None):
        raise RuntimeError("nope")

    case = {"id": "err", "question": "?", "contains": ["x"]}
    r = run_case(case, boom)
    assert r.passed is False
    assert r.error is not None
    assert "nope" in r.error


def test_load_golden_reads_yaml() -> None:
    from run_eval import load_golden

    path = EVAL / "golden.yaml"
    cases = load_golden(path)
    assert isinstance(cases, list)
    assert len(cases) >= 5
    assert all("id" in c and "question" in c for c in cases)
    ids = [c["id"] for c in cases]
    assert "orders_count" in ids


def test_main_exits_zero_when_all_pass(monkeypatch, tmp_path) -> None:
    import run_eval

    golden = tmp_path / "g.yaml"
    golden.write_text(
        "- id: t1\n  question: q?\n  contains: ['42']\n",
        encoding="utf-8",
    )

    def fake_ask(q, session_id=None):
        return {"answer": "the answer is 42", "prompt_tokens": 1, "completion_tokens": 1}

    # Inject a stub `agent` module so the lazy import inside main() picks it up.
    import types

    stub = types.ModuleType("agent")
    stub.ask = fake_ask
    monkeypatch.setitem(sys.modules, "agent", stub)

    rc = run_eval.main(["--golden", str(golden)])
    assert rc == 0


def test_main_exits_nonzero_when_any_fail(monkeypatch, tmp_path) -> None:
    import types

    import run_eval

    golden = tmp_path / "g.yaml"
    golden.write_text(
        "- id: t1\n  question: q?\n  contains: ['missing-string']\n",
        encoding="utf-8",
    )

    def fake_ask(q, session_id=None):
        return {"answer": "unrelated", "prompt_tokens": 0, "completion_tokens": 0}

    stub = types.ModuleType("agent")
    stub.ask = fake_ask
    monkeypatch.setitem(sys.modules, "agent", stub)

    rc = run_eval.main(["--golden", str(golden)])
    assert rc == 1
