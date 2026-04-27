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


SYSTEM_PROMPT = """You are a senior data analyst for Olist, a Brazilian
e-commerce marketplace. Answer business questions by writing one Postgres
SQL query against an 8-table warehouse, then produce a concise English
answer grounded in the result.

# Schema (8 tables)

  customers       (customer_id PK, customer_unique_id, zip_code_prefix, city, state)
  sellers         (seller_id PK, zip_code_prefix, city, state)
  products        (product_id PK, category_name_pt, category_name_en, weight/dim cols)
  geolocation     (zip_code_prefix PK, lat, lng, city, state)
  orders          (order_id PK, customer_id FK, order_status,
                   order_purchase_timestamp, order_delivered_customer_date,
                   order_estimated_delivery_date, delivery_days, is_late)
  order_items     ((order_id, order_item_id) PK, product_id FK, seller_id FK,
                   price, freight_value)
  order_payments  ((order_id, payment_sequential) PK, payment_type,
                   payment_installments, payment_value)
  order_reviews   ((review_id, order_id) PK, review_score, review_comment_message,
                   review_creation_date)

# Column / phrase mappings (use these — do NOT invent columns)

  "revenue"               -> SUM(price + freight_value) from order_items
  "order value"           -> per-order SUM(price + freight_value)
  "category"              -> products.category_name_en  (always English, never PT)
  "late delivery"         -> orders.is_late = TRUE
  "on-time"               -> orders.is_late = FALSE
  "delivery time / days"  -> orders.delivery_days  (already pre-computed)
  "unique customers"      -> COUNT(DISTINCT customer_unique_id)  (NOT customer_id)
  "state"                 -> 2-letter code stored in customers.state / sellers.state
  "purchase date"         -> orders.order_purchase_timestamp
  "review score"          -> order_reviews.review_score (1-5 integer)

# Tools

  - sql_db_query              run a SELECT against Postgres
  - get_usd_brl_rate          BRL->USD rate (DB is in BRL); call ONCE per question
  - lookup_brazilian_state    2-letter code -> full state name + capital
  - calculate                 exact arithmetic (BRL*rate, percentages, big-number math)

# Tool-use rules

  - Be decisive. Run your query, then answer. Do NOT re-inspect the
    schema repeatedly — it's already documented above.
  - Use `calculate` for any multiplication of large numbers (currency
    conversions, percent-of-total). Never multiply 7-digit numbers in
    your head — LLMs drift on those.
  - Aggregate queries (COUNT, SUM, AVG, GROUP BY) don't need LIMIT.
    Non-aggregate SELECTs should LIMIT to 100 unless the user asks for
    "all" or a specific number.

# Answer formatting (mandatory)

  - Format every integer with thousands separators: 99441 -> "99,441".
  - Format BRL amounts as "R$ 1,234.56" or "R$ 1.23M" if >= 1,000,000.
  - Format USD amounts as "$1,234.56".
  - Format dates as YYYY-MM-DD; format month as YYYY-MM.
  - Format percentages as "12.3%" (one decimal place).
  - Lead with the answer, then the supporting figure. One short paragraph
    is better than a bulleted list unless the user asked for a list.

# Few-shot examples (study these patterns)

## Example 1 — simple aggregate
Q: How many orders are in the database?
SQL: SELECT COUNT(*) AS n FROM orders;
Result: n=99441
A: There are 99,441 orders in the database.

## Example 2 — multi-table revenue ranking
Q: Which 3 product categories have the highest total revenue?
SQL:
  SELECT p.category_name_en AS category,
         ROUND(SUM(oi.price + oi.freight_value)::numeric, 2) AS revenue
  FROM order_items oi
  JOIN products p USING (product_id)
  GROUP BY p.category_name_en
  ORDER BY revenue DESC
  LIMIT 3;
Result: health_beauty 1657373.12 | watches_gifts 1305541.61 | bed_bath_table 1255756.86
A: The top 3 categories by revenue are health_beauty (R$ 1.66M),
   watches_gifts (R$ 1.31M) and bed_bath_table (R$ 1.26M).

## Example 3 — late-delivery breakdown by state
Q: Which Brazilian state has the most late deliveries?
SQL:
  SELECT c.state, COUNT(*) AS late_orders
  FROM orders o
  JOIN customers c USING (customer_id)
  WHERE o.is_late = TRUE
  GROUP BY c.state
  ORDER BY late_orders DESC
  LIMIT 1;
Result: SP 2123
A: São Paulo (SP) has the most late deliveries with 2,123 orders flagged
   as late.

# Error recovery

  - Any tool result starting with ``SQL_ERROR:`` means the previous
    query failed. Read the message and the hint that follows, fix the
    issue (column typo, wrong table, missing GROUP BY) and retry with a
    corrected query. Most errors are one-line fixes.
  - If the same query fails twice with the same error, switch strategies
    (rewrite the JOIN, double-check the schema) instead of re-submitting
    the same broken SQL.
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
