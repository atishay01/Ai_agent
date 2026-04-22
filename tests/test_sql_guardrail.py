"""Tests for the SQL guardrail — the project's main security barrier."""

import pytest

from sql_guardrail import UnsafeSQLError, validate_sql


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
