"""SQL guardrail — reject anything the agent writes that isn't a safe read.

Defense-in-depth for the LangChain SQL agent. The system prompt *asks*
the model not to issue DDL/DML; this module *prevents* it.

Rules enforced by ``validate_sql``:
  1. Query parses cleanly.
  2. First statement is ``SELECT`` (or a CTE: ``WITH ... SELECT ...``).
  3. No banned top-level keywords anywhere: ``INSERT``, ``UPDATE``,
     ``DELETE``, ``DROP``, ``TRUNCATE``, ``ALTER``, ``CREATE``, ``GRANT``,
     ``REVOKE``, ``COPY``, ``CALL``, ``EXECUTE``, ``MERGE``, ``VACUUM``,
     ``REINDEX``, ``COMMENT``, ``SET``, ``RESET``, ``LOCK``.
  4. No stacked statements — only one top-level statement per query.
     (``SELECT 1; DROP TABLE users`` would otherwise be valid SQL.)
  5. No row-fan-out aggregations (``array_agg``, ``json_agg``,
     ``string_agg``, ``jsonb_agg``, ``xmlagg``, ``json_object_agg``,
     ``jsonb_object_agg``). These collapse arbitrarily many rows into a
     single cell and would otherwise defeat the row cap.

Violations raise ``UnsafeSQLError``. The agent's SQL-execution path
wraps every ``run()`` call in this check (see ``SafeSQLDatabase``).

In addition, ``enforce_row_cap`` wraps the query in a parenthesised
subquery and appends ``LIMIT <n>`` on the *outside*. This is robust
against trailing line comments (``-- …``) and inner ``LIMIT`` clauses
that could otherwise let the cap be commented out or short-circuited.
"""

from __future__ import annotations

import re

import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import DDL, DML, Keyword

BANNED_AGG_FUNCTIONS: frozenset[str] = frozenset(
    {
        # Aggregations that collapse many rows into one cell — they would
        # let an attacker exfiltrate a whole column past the row cap.
        "ARRAY_AGG",
        "JSON_AGG",
        "JSONB_AGG",
        "STRING_AGG",
        "XMLAGG",
        "JSON_OBJECT_AGG",
        "JSONB_OBJECT_AGG",
    }
)


BANNED_KEYWORDS: frozenset[str] = frozenset(
    {
        # DML writes
        "INSERT",
        "UPDATE",
        "DELETE",
        "MERGE",
        "UPSERT",
        "REPLACE",
        # DDL
        "DROP",
        "TRUNCATE",
        "ALTER",
        "CREATE",
        "RENAME",
        # DCL
        "GRANT",
        "REVOKE",
        # Server-side execution / admin
        "COPY",
        "CALL",
        "EXECUTE",
        "EXEC",
        "VACUUM",
        "REINDEX",
        "COMMENT",
        "SET",
        "RESET",
        "LOCK",
        "UNLOCK",
        "ATTACH",
        "DETACH",
    }
)


class UnsafeSQLError(ValueError):
    """Raised when a SQL string fails the guardrail check."""


def _statement_kind(stmt: Statement) -> str:
    """Return the first meaningful keyword, upper-cased (e.g. 'SELECT', 'WITH', 'DROP')."""
    for tok in stmt.tokens:
        if tok.is_whitespace or tok.ttype is sqlparse.tokens.Comment:
            continue
        if tok.ttype in (DML, DDL, Keyword):
            return tok.normalized.upper()
    return ""


def _contains_banned_keyword(stmt: Statement) -> str | None:
    """Scan every token; return the first banned keyword found, else None."""
    for tok in stmt.flatten():
        if tok.ttype in (DDL, DML, Keyword):
            word = tok.normalized.upper()
            if word in BANNED_KEYWORDS:
                return word
    return None


def _contains_banned_aggregation(stmt: Statement) -> str | None:
    """Scan every token for a banned row-fan-out aggregation function.

    Anywhere in the parsed statement is enough — there is no legitimate
    use of ``array_agg``/``json_agg``/etc. against the Olist schema, and
    permitting them in a CTE or subquery would still let the outer
    ``LIMIT`` see one giant row.
    """
    for tok in stmt.flatten():
        if tok.ttype is None or tok.ttype is sqlparse.tokens.Name:
            word = tok.value.upper()
            if word in BANNED_AGG_FUNCTIONS:
                return word
    return None


def validate_sql(query: str) -> None:
    """Raise ``UnsafeSQLError`` if ``query`` is not a single safe SELECT.

    Whitespace, trailing semicolons, and comments are tolerated.
    """
    if not query or not query.strip():
        raise UnsafeSQLError("empty SQL")

    parsed = sqlparse.parse(query)
    # ``parse()`` can return a trailing empty statement for "SELECT 1;"
    statements = [s for s in parsed if s.tokens and str(s).strip().rstrip(";").strip()]

    if len(statements) != 1:
        raise UnsafeSQLError(f"expected exactly one SQL statement, got {len(statements)}")

    stmt = statements[0]
    kind = _statement_kind(stmt)
    if kind not in {"SELECT", "WITH"}:
        raise UnsafeSQLError(f"only SELECT/CTE queries are allowed (got {kind!r})")

    banned = _contains_banned_keyword(stmt)
    if banned:
        raise UnsafeSQLError(f"banned keyword in query: {banned}")

    banned_agg = _contains_banned_aggregation(stmt)
    if banned_agg:
        raise UnsafeSQLError(f"banned aggregation function: {banned_agg} (would bypass row cap)")


def enforce_row_cap(query: str, cap: int) -> str:
    """Wrap the query in a subquery and apply ``LIMIT <cap>`` on the outside.

    Wrapping (rather than appending) defends against two bypasses:

      * A trailing line comment — ``SELECT * FROM customers -- end`` —
        would otherwise consume the appended ``LIMIT`` so Postgres sees
        no cap at all.
      * An inner ``LIMIT 1000000`` would be honoured as-is if we just
        skipped queries that "already had a LIMIT", letting the agent
        return arbitrarily many rows.

    The outer ``LIMIT`` is always at most as restrictive as any inner
    ``LIMIT`` the agent supplied (a query that returns 50 rows internally
    still returns 50 rows when capped at 1000), so there is no
    correctness loss from always wrapping.
    """
    if cap <= 0:
        return query
    cleaned = query.strip().rstrip(";").strip()
    if not cleaned:
        return query
    return f"SELECT * FROM (\n{cleaned}\n) AS _capped LIMIT {cap}"


# ---------------------------------------------------------------------
# SQL error formatting — converts raw DBAPI exceptions into a clean,
# agent-readable string so the LLM can rewrite the query and retry.
# ---------------------------------------------------------------------
_MAX_ERR_LEN = 400

_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"column .* does not exist", re.IGNORECASE),
        "Check the schema for exact column names — the case must match "
        "and joins must use the right side's column.",
    ),
    (
        re.compile(r"relation .* does not exist", re.IGNORECASE),
        "The table name is wrong. Available tables: customers, sellers, "
        "products, geolocation, orders, order_items, order_payments, "
        "order_reviews.",
    ),
    (
        re.compile(r"syntax error", re.IGNORECASE),
        "Recheck SQL syntax — common causes: missing comma, unbalanced "
        "parentheses, unquoted string literal, or reserved word as alias.",
    ),
    (
        re.compile(r"function .* does not exist", re.IGNORECASE),
        "Postgres function name or argument types are wrong — verify the "
        "function exists and check argument casts.",
    ),
    (
        re.compile(r"division by zero", re.IGNORECASE),
        "Guard the divisor with NULLIF(divisor, 0) to avoid division by zero.",
    ),
    (
        re.compile(r"group by", re.IGNORECASE),
        "Every non-aggregated SELECT column must appear in GROUP BY.",
    ),
)


def format_sql_error(exc: BaseException) -> str:
    """Render an exception from SQL execution as a single agent-readable line.

    Output always starts with ``SQL_ERROR:`` so the LLM can recognise the
    pattern. A short hint is appended when the message matches a known
    failure mode (unknown column, syntax error, etc.) — this nudges the
    agent toward a correct rewrite without over-prescribing.
    """
    raw = str(exc).strip().replace("\n", " ")
    if len(raw) > _MAX_ERR_LEN:
        raw = raw[:_MAX_ERR_LEN] + "…"
    hint = ""
    for pattern, advice in _HINTS:
        if pattern.search(raw):
            hint = f" Hint: {advice}"
            break
    return f"SQL_ERROR: {raw}{hint} Please rewrite the query and try again."
