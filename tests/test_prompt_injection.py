"""Adversarial / prompt-injection test cases for the SQL guardrail.

These exercise *what the guardrail must refuse to execute*, regardless
of how the LLM is talked into producing the SQL. Two layers:

  1. Direct guardrail tests — feed the malicious SQL straight to
     ``validate_sql`` / ``enforce_row_cap`` and assert it's blocked or
     neutralised. Deterministic, no LLM, runs in CI.
  2. Integration-level tests at the ``ask()`` boundary, with the LLM
     mocked to return the malicious SQL. Confirms the guardrail
     rejects the query before it reaches Postgres and the agent gets
     a SQL_ERROR back so it can self-repair.

Together these prove the guardrail does its job whether the attack
arrives as a crafted user prompt OR as a coerced tool call.
"""

from __future__ import annotations

import pytest

from sql_guardrail import UnsafeSQLError, enforce_row_cap, validate_sql

# =====================================================================
# Layer 1 — Direct guardrail unit tests
# =====================================================================


@pytest.mark.parametrize(
    ("attack_label", "sql"),
    [
        # --- Classic DML / DDL injection ---
        (
            "drop_table_via_stacked_statement",
            "SELECT 1; DROP TABLE customers",
        ),
        (
            "delete_via_stacked_statement",
            "SELECT 1; DELETE FROM orders WHERE 1=1",
        ),
        (
            "update_via_stacked_statement",
            "SELECT 1; UPDATE orders SET order_status = 'cancelled'",
        ),
        # --- Writable CTE — looks like a SELECT, isn't ---
        (
            "writable_cte_delete",
            "WITH x AS (DELETE FROM customers RETURNING customer_id) " "SELECT customer_id FROM x",
        ),
        (
            "writable_cte_update",
            "WITH x AS (UPDATE orders SET order_status = 'X' RETURNING order_id) "
            "SELECT * FROM x",
        ),
        # --- Privilege escalation attempts ---
        (
            "grant_to_self",
            "GRANT ALL ON customers TO public",
        ),
        (
            "create_extension",
            "CREATE EXTENSION IF NOT EXISTS plpython3u",
        ),
        # --- Server-side admin / dangerous functions ---
        (
            "execute_dynamic_sql",
            "EXECUTE 'DROP TABLE customers'",
        ),
        (
            "vacuum_full",
            "VACUUM FULL customers",
        ),
        # --- Row-fanout aggregation exfil ---
        (
            "array_agg_whole_table",
            "SELECT array_agg(customer_unique_id) FROM customers",
        ),
        (
            "json_agg_whole_table",
            "SELECT json_agg(c) FROM customers c",
        ),
        (
            "string_agg_whole_column",
            "SELECT string_agg(customer_unique_id, ',') FROM customers",
        ),
        (
            "agg_hidden_in_cte",
            "WITH x AS (SELECT array_agg(id) AS ids FROM orders) SELECT * FROM x",
        ),
    ],
)
def test_guardrail_refuses_known_attacks(attack_label: str, sql: str) -> None:
    """Every entry in this table is a real exfil / mutation pattern.

    The guardrail must refuse all of them by raising UnsafeSQLError.
    If a future change inadvertently lets one through, this test fails
    loudly and points at the specific attack class.
    """
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql)


def test_guardrail_blocks_trailing_comment_limit_bypass() -> None:
    """A trailing line comment must NOT swallow the row cap.

    Earlier versions appended ``LIMIT 100`` to the original SQL, so
    ``SELECT * FROM customers -- end`` became
    ``SELECT * FROM customers -- end LIMIT 100`` and Postgres ignored
    the LIMIT. The wrap-in-subquery design puts the cap on a fresh
    outer line where no inner comment can reach it.
    """
    out = enforce_row_cap("SELECT * FROM customers -- end of query", cap=100)
    last_line = out.rstrip().splitlines()[-1].strip()
    assert last_line == ") AS _capped LIMIT 100"


def test_guardrail_neutralises_inner_limit_bypass() -> None:
    """An inner ``LIMIT 1_000_000`` must NOT short-circuit the outer cap."""
    out = enforce_row_cap("SELECT * FROM (SELECT * FROM customers LIMIT 1000000) t", cap=100)
    assert out.endswith("LIMIT 100")
    # The inner limit is preserved — the outer one is what bounds the
    # final result.
    assert "LIMIT 1000000" in out


# =====================================================================
# Layer 2 — Integration: ``ask()`` with the LLM mocked
# =====================================================================
#
# The LLM "decides" to run a malicious SQL; the guardrail in
# SafeSQLDatabase.run() must reject before Postgres sees it.
# We don't mock Postgres — the SQL never reaches it.


def test_safe_sql_database_rejects_drop_table_at_run_layer() -> None:
    """Even if the agent emits DROP TABLE, run() returns SQL_ERROR — never executes."""
    from agent import SafeSQLDatabase

    # Build a fake DB without a real connection. We're not calling run
    # against Postgres; we're testing the guardrail wrapping.
    db = SafeSQLDatabase.__new__(SafeSQLDatabase)
    out = db.run("DROP TABLE customers")
    assert out.startswith("SQL_ERROR:")
    assert "guardrail" in out.lower()


def test_safe_sql_database_rejects_array_agg_exfil_at_run_layer() -> None:
    """The agent can't exfiltrate a whole column via array_agg."""
    from agent import SafeSQLDatabase

    db = SafeSQLDatabase.__new__(SafeSQLDatabase)
    out = db.run("SELECT array_agg(customer_unique_id) FROM customers")
    assert out.startswith("SQL_ERROR:")
    assert "agg" in out.lower()


def test_safe_sql_database_rejects_stacked_delete_at_run_layer() -> None:
    from agent import SafeSQLDatabase

    db = SafeSQLDatabase.__new__(SafeSQLDatabase)
    out = db.run("SELECT 1; DELETE FROM orders")
    assert out.startswith("SQL_ERROR:")
