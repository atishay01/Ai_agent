"""Tests for the SQL guardrail — the project's main security barrier."""

import pytest

from sql_guardrail import UnsafeSQLError, enforce_row_cap, format_sql_error, validate_sql


# ---------------------------------------------------------------------
# Safe queries — must NOT raise.
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1",
        "SELECT * FROM orders LIMIT 10",
        "select id from orders;",
        "  SELECT COUNT(*) FROM customers  ",
        "WITH x AS (SELECT 1 AS n) SELECT n FROM x",
        "SELECT o.order_id, c.state FROM orders o JOIN customers c USING (customer_id)",
        "SELECT * FROM orders WHERE is_late = TRUE AND state = 'SP'",
    ],
)
def test_safe_queries_pass(query: str) -> None:
    validate_sql(query)  # must not raise


# ---------------------------------------------------------------------
# Dangerous / malformed queries — must raise.
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    ("query", "expected_fragment"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("INSERT INTO orders VALUES (1)", "INSERT"),
        ("UPDATE orders SET id = 1", "UPDATE"),
        ("DELETE FROM orders", "DELETE"),
        ("DROP TABLE customers", "DROP"),
        ("TRUNCATE orders", "TRUNCATE"),
        ("ALTER TABLE orders ADD COLUMN x INT", "ALTER"),
        ("CREATE TABLE evil (id INT)", "CREATE"),
        ("VACUUM ANALYZE", "VACUUM"),
        ("CALL sp_foo()", "CALL"),
        ("SELECT 1; DROP TABLE orders", "one SQL statement"),
        ("SELECT * FROM orders; SELECT * FROM customers", "one SQL statement"),
    ],
)
def test_dangerous_queries_blocked(query: str, expected_fragment: str) -> None:
    with pytest.raises(UnsafeSQLError) as excinfo:
        validate_sql(query)
    assert expected_fragment.lower() in str(excinfo.value).lower()


def test_subclass_is_value_error() -> None:
    """UnsafeSQLError is-a ValueError so existing handlers catch it."""
    assert issubclass(UnsafeSQLError, ValueError)


# ---------------------------------------------------------------------
# enforce_row_cap — defense-in-depth row limiter.
# ---------------------------------------------------------------------
def test_enforce_row_cap_appends_when_missing() -> None:
    out = enforce_row_cap("SELECT * FROM customers", cap=1000)
    assert out.endswith("LIMIT 1000")


def test_enforce_row_cap_strips_trailing_semicolon_before_appending() -> None:
    out = enforce_row_cap("SELECT * FROM customers;", cap=1000)
    assert out.endswith("LIMIT 1000")
    assert ";" not in out


def test_enforce_row_cap_skips_when_limit_present() -> None:
    q = "SELECT * FROM customers LIMIT 50"
    assert enforce_row_cap(q, cap=1000) == q


def test_enforce_row_cap_skips_when_limit_in_subquery() -> None:
    """Conservative: any LIMIT (even inner) suppresses the auto-cap."""
    q = "SELECT * FROM (SELECT id FROM orders LIMIT 5) sub"
    assert enforce_row_cap(q, cap=1000) == q


def test_enforce_row_cap_handles_empty_input() -> None:
    assert enforce_row_cap("", cap=1000) == ""
    assert enforce_row_cap("   ", cap=1000) == "   "


def test_enforce_row_cap_disabled_when_cap_zero() -> None:
    q = "SELECT * FROM customers"
    assert enforce_row_cap(q, cap=0) == q


# ---------------------------------------------------------------------
# format_sql_error — agent self-repair signal.
# ---------------------------------------------------------------------
def test_format_sql_error_uses_consistent_prefix() -> None:
    out = format_sql_error(RuntimeError("boom"))
    assert out.startswith("SQL_ERROR:")
    assert "rewrite the query" in out


def test_format_sql_error_unknown_column_hint() -> None:
    exc = RuntimeError('column "ordrs.id" does not exist')
    out = format_sql_error(exc)
    assert "SQL_ERROR" in out
    assert "Hint" in out
    assert "column" in out.lower()


def test_format_sql_error_unknown_table_lists_options() -> None:
    exc = RuntimeError('relation "ordres" does not exist')
    out = format_sql_error(exc)
    assert "Hint" in out
    # Hint mentions actual table names so the agent can correct.
    assert "order_items" in out
    assert "customers" in out


def test_format_sql_error_syntax_error_hint() -> None:
    exc = RuntimeError('syntax error at or near "FORM"')
    out = format_sql_error(exc)
    assert "Hint" in out
    assert "syntax" in out.lower()


def test_format_sql_error_unknown_pattern_no_hint() -> None:
    """Errors that don't match any pattern still produce a valid SQL_ERROR line."""
    out = format_sql_error(RuntimeError("some unknown DBAPI failure"))
    assert out.startswith("SQL_ERROR:")
    assert "Hint" not in out


def test_format_sql_error_truncates_long_messages() -> None:
    long_msg = "x" * 10_000
    out = format_sql_error(RuntimeError(long_msg))
    # Cap is 400 + the prefix/suffix, so total stays bounded.
    assert len(out) < 700
    assert "…" in out
