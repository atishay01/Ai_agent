"""
LangChain SQL agent: converts natural-language questions into SQL
against the Olist Postgres DB, with extra tools for currency conversion,
state lookup and exact arithmetic.

Features:
  * Per-query token accounting via ``TokenUsageCallback`` → ``METRICS``.
  * SQLite-backed LRU cache for identical questions (survives restarts;
    see ``cache.py`` and ``state_store.py``).
  * Optional ``session_id`` for multi-turn follow-ups — prior turns are
    persisted via ``session_history`` and prepended to the prompt so
    the agent can resolve pronouns like "and for São Paulo?".
  * Returns serialised intermediate steps so the UI can render a trace.
"""

from __future__ import annotations

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

import session_history
from cache import CACHE
from callbacks import TokenUsageCallback
from config import settings
from db import get_connection_string
from logging_setup import logger, redact
from metrics import METRICS
from sql_guardrail import UnsafeSQLError, enforce_row_cap, format_sql_error, validate_sql
from web_tools import WEB_TOOLS

_log = logger.bind(component="agent")


class SafeSQLDatabase(SQLDatabase):
    """SQLDatabase that runs every query through the guardrail first.

    Any attempt to execute non-SELECT SQL (INSERT/UPDATE/DROP/etc.) is
    blocked before it reaches Postgres and is returned to the agent as
    an error string. Postgres-side execution errors (unknown column,
    syntax error, ...) are also caught and reformatted via
    ``format_sql_error`` so the agent gets a consistent ``SQL_ERROR:``
    prefix it can recognise and self-repair from, rather than a raw
    DBAPI traceback.
    """

    def run(self, command, *args, **kwargs):  # type: ignore[override]
        try:
            validate_sql(command)
        except UnsafeSQLError as exc:
            METRICS.record_guardrail_block()
            _log.bind(sql=command[:200]).warning("guardrail blocked query: {}", exc)
            return f"SQL_ERROR: query blocked by guardrail ({exc})."
        capped = enforce_row_cap(command, settings.sql_max_rows)
        _log.bind(sql=capped[:200]).debug("executing sql")
        try:
            return super().run(capped, *args, **kwargs)
        except Exception as exc:
            METRICS.record_sql_error()
            _log.bind(sql=capped[:200]).warning("sql execution failed: {}", exc)
            return format_sql_error(exc)

    def run_no_throw(self, command, *args, **kwargs):  # type: ignore[override]
        try:
            validate_sql(command)
        except UnsafeSQLError as exc:
            METRICS.record_guardrail_block()
            _log.bind(sql=command[:200]).warning("guardrail blocked query: {}", exc)
            return f"SQL_ERROR: query blocked by guardrail ({exc})."
        capped = enforce_row_cap(command, settings.sql_max_rows)
        try:
            return super().run_no_throw(capped, *args, **kwargs)
        except Exception as exc:
            METRICS.record_sql_error()
            _log.bind(sql=capped[:200]).warning("sql execution failed: {}", exc)
            return format_sql_error(exc)


SYSTEM_PROMPT = """You are a data analyst for a Brazilian e-commerce
company (Olist). Answer questions by writing SQL against a Postgres DB
with 8 tables: customers, sellers, products, geolocation, orders,
order_items, order_payments, order_reviews. Use category_name_en for
category filters. Order revenue = SUM(price + freight_value) from
order_items. Late deliveries: is_late = TRUE. Never run UPDATE/DELETE/
INSERT/DROP. Always LIMIT to 100 rows unless asking for aggregates.

You have three extra tools:
  - get_usd_brl_rate for USD conversions (DB is in BRL)
  - lookup_brazilian_state for state full names / capitals
  - calculate for any arithmetic (use this for BRL * rate, percentages,
    and any large-number math — never compute yourself).

Tool use rules:
  - Call get_usd_brl_rate only ONCE per question (reuse the result).
  - When the user asks for multiple states' capitals, call
    lookup_brazilian_state once per distinct state code, then combine.
  - Be decisive: run your queries, call your tools, then answer. Do
    not re-inspect the schema repeatedly.

Error recovery:
  - Any tool result that begins with ``SQL_ERROR:`` means the previous
    query failed. Read the message and the hint that follows it, fix
    the underlying issue (column typo, wrong table, missing GROUP BY,
    ...) and retry with a corrected query. Do not give up after a
    single failure — most errors are one-line fixes.
  - If the same query fails twice with the same error, switch
    strategies (rewrite the JOIN, double-check the schema) instead of
    re-submitting.
"""


def _db() -> SafeSQLDatabase:
    return SafeSQLDatabase.from_uri(
        get_connection_string(),
        include_tables=[
            "customers",
            "sellers",
            "products",
            "geolocation",
            "orders",
            "order_items",
            "order_payments",
            "order_reviews",
        ],
        sample_rows_in_table_info=2,
    )


def _llm() -> ChatGroq:
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key.get_secret_value(),
        temperature=0,
    )


_agent = None


def get_agent():
    """Build the SQL agent (once per process)."""
    global _agent
    if _agent is None:
        _agent = create_sql_agent(
            llm=_llm(),
            db=_db(),
            agent_type="tool-calling",
            verbose=False,
            prefix=SYSTEM_PROMPT,
            extra_tools=WEB_TOOLS,
            handle_parsing_errors=True,
            max_iterations=settings.agent_max_iterations,
            return_intermediate_steps=True,
        )
    return _agent


def _serialize_steps(steps) -> list[dict]:
    """Turn a list of (AgentAction, observation) into JSON-safe dicts."""
    out: list[dict] = []
    for item in steps or []:
        try:
            action, observation = item
        except (TypeError, ValueError):
            continue
        tool = getattr(action, "tool", type(action).__name__)
        tool_input = getattr(action, "tool_input", "")
        out.append(
            {
                "tool": str(tool),
                "tool_input": str(tool_input)[:500],
                "observation": str(observation)[:1000],
            }
        )
    return out


def _build_input(question: str, session_id: str | None) -> str:
    """Prepend prior-turn context when a session is active."""
    if not session_id:
        return question
    history = session_history.get(session_id)
    if not history:
        return question
    prior = "\n".join(history)
    return f"Prior conversation:\n{prior}\n\nCurrent question: {question}"


def clear_session(session_id: str) -> None:
    """Forget all prior turns for a session."""
    session_history.clear(session_id)


def ask(question: str, session_id: str | None = None, trace_id: str | None = None) -> dict:
    """Run a single question through the agent.

    The response cache is checked first regardless of ``session_id`` —
    identical literal questions short-circuit the LLM call. When a
    session is active, the cached answer is still threaded into the
    session history so follow-ups stay coherent.

    ``trace_id`` is contextualised into loguru so every downstream log
    line (guardrail blocks, SQL execution, DBAPI errors in
    ``SafeSQLDatabase``) carries the same identifier as the inbound
    request, enabling single-grep correlation.
    """
    with logger.contextualize(trace_id=trace_id or "-"):
        cached = CACHE.get(question)
        if cached is not None:
            result = dict(cached)
            result["cached"] = True
            if session_id:
                session_history.append(session_id, f"User: {question}")
                session_history.append(session_id, f"Assistant: {result['answer']}")
            _log.bind(q_hash=redact(question), session_hash=redact(session_id)).info("cache hit")
            return result

        cb = TokenUsageCallback()
        prompt_input = _build_input(question, session_id)

        try:
            invoke_result = get_agent().invoke(
                {"input": prompt_input},
                config={"callbacks": [cb]},
            )
        except Exception:
            METRICS.record_query(cb.prompt_tokens, cb.completion_tokens, failed=True)
            raise

        answer = invoke_result.get("output", "No answer.")
        steps = _serialize_steps(invoke_result.get("intermediate_steps"))

        METRICS.record_query(cb.prompt_tokens, cb.completion_tokens, failed=False)

        if session_id:
            session_history.append(session_id, f"User: {question}")
            session_history.append(session_id, f"Assistant: {answer}")

        response = {
            "question": question,
            "answer": answer,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "steps": steps,
            "cached": False,
        }

        CACHE.set(question, response)
        return response


if __name__ == "__main__":
    import json

    print(json.dumps(ask("How many orders are in the database?"), indent=2))
