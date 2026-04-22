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
"""

from __future__ import annotations

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
