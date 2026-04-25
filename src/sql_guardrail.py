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

Violations raise ``UnsafeSQLError``. The agent's SQL-execution path
wraps every ``run()`` call in this check (see ``SafeSQLDatabase``).

In addition, ``enforce_row_cap`` appends ``LIMIT <n>`` to any SELECT
that has no LIMIT, so a buggy or coerced agent query can't dump whole
tables (defense-in-depth against data exfiltration).
"""

from __future__ import annotations

import re

import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import DDL, DML, Keyword

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


_LIMIT_RE = re.compile(r"\blimit\s+\d+", re.IGNORECASE)


def enforce_row_cap(query: str, cap: int) -> str:
    """Append ``LIMIT <cap>`` to a SELECT that has no LIMIT clause.

    Conservative: if any LIMIT (in a subquery, CTE, or the outer query)
    exists, we leave the query alone — the agent or query author already
    bounded the result. Aggregate queries (``COUNT``, ``SUM``, ...) are
    unaffected since they return one row regardless.
    """
    if cap <= 0:
        return query
    cleaned = query.strip().rstrip(";").strip()
    if not cleaned:
        return query
    if _LIMIT_RE.search(cleaned):
        return query
    return f"{cleaned} LIMIT {cap}"


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
