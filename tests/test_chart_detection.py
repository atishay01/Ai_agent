"""Unit tests for the chart-auto-detection logic in app.py.

These cover the pure-Python helpers (``_parse_observation``,
``_looks_like_date``, ``_last_sql_result``) without booting Streamlit.
``render_chart_if_useful`` itself calls Streamlit primitives, so it's
left for manual smoke-testing in the live UI.
"""

from __future__ import annotations

# Streamlit emits ScriptRunContext warnings on bare import; they're
# noise here. The module-level Streamlit calls in app.py are also
# harmless under bare import — the page just doesn't render.
import importlib

app = importlib.import_module("app")


# ---------------------------------------------------------------------
# _parse_observation
# ---------------------------------------------------------------------
def test_parse_observation_returns_list_of_tuples_for_well_formed_input() -> None:
    obs = "[('2017-01', 134567.89), ('2017-02', 152341.10)]"
    rows = app._parse_observation(obs)
    assert rows == [("2017-01", 134567.89), ("2017-02", 152341.10)]


def test_parse_observation_handles_integer_values() -> None:
    rows = app._parse_observation("[('SP', 41746), ('RJ', 12852)]")
    assert rows == [("SP", 41746), ("RJ", 12852)]


def test_parse_observation_returns_none_on_garbage() -> None:
    assert app._parse_observation("Some explanation text, not a list.") is None
    assert app._parse_observation("[bad python]") is None
    assert app._parse_observation("") is None


def test_parse_observation_rejects_inconsistent_row_widths() -> None:
    """Mixed-width rows are not chartable — refuse parse."""
    rows = app._parse_observation("[('a', 1), ('b', 2, 3)]")
    assert rows is None


def test_parse_observation_rejects_single_column_results() -> None:
    """A scalar like '[(99441,)]' shouldn't render as a chart."""
    assert app._parse_observation("[(99441,)]") is None


# ---------------------------------------------------------------------
# _looks_like_date
# ---------------------------------------------------------------------
def test_looks_like_date_accepts_iso_dates_and_months() -> None:
    assert app._looks_like_date("2017-01-15") is True
    assert app._looks_like_date("2017-01") is True
    assert app._looks_like_date("2017") is True


def test_looks_like_date_rejects_text_labels() -> None:
    assert app._looks_like_date("health_beauty") is False
    assert app._looks_like_date("SP") is False
    assert app._looks_like_date("") is False


# ---------------------------------------------------------------------
# _last_sql_result
# ---------------------------------------------------------------------
def test_last_sql_result_picks_latest_sql_step() -> None:
    steps = [
        {
            "tool": "sql_db_query",
            "tool_input": "SELECT 1",
            "observation": "[('a', 1)]",
        },
        {
            "tool": "sql_db_query",
            "tool_input": "SELECT 2",
            "observation": "[('b', 2)]",
        },
        {"tool": "calculate", "tool_input": "1+1", "observation": "= 2"},
    ]
    sql, rows = app._last_sql_result(steps)
    # Most recent SQL step wins.
    assert sql == "SELECT 2"
    assert rows == [("b", 2)]


def test_last_sql_result_returns_none_when_no_sql_step_exists() -> None:
    steps = [
        {"tool": "calculate", "tool_input": "1+1", "observation": "= 2"},
        {"tool": "lookup_brazilian_state", "tool_input": "SP", "observation": "São Paulo"},
    ]
    sql, rows = app._last_sql_result(steps)
    assert sql is None
    assert rows is None


def test_last_sql_result_skips_unparsable_observations() -> None:
    """If the most recent SQL step has unparsable output, fall through."""
    steps = [
        {
            "tool": "sql_db_query",
            "tool_input": "SELECT 1",
            "observation": "[('a', 1)]",
        },
        {
            "tool": "sql_db_query",
            "tool_input": "SELECT 2",
            "observation": "Something unparseable",
        },
    ]
    sql, rows = app._last_sql_result(steps)
    assert sql == "SELECT 1"
    assert rows == [("a", 1)]
