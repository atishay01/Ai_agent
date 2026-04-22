"""
LangChain SQL agent: converts natural-language questions into SQL
against the Olist Postgres DB, with extra tools for currency conversion,
state lookup and exact arithmetic.

Phase 3 extensions:
  * Per-query token accounting via ``TokenUsageCallback`` → ``METRICS``.
  * In-process LRU cache for identical questions (stateless calls only).
  * Optional ``session_id`` for multi-turn follow-ups — prior turns are
    prepended to the prompt so the agent can resolve pronouns like
    "and for São Paulo?".
  * Returns serialised intermediate steps so the UI can render a trace.
"""

from __future__ import annotations

from collections import defaultdict, deque

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

from cache import CACHE
from callbacks import TokenUsageCallback
from config import settings
from db import get_connection_string
from logging_setup import logger
from metrics import METRICS
from sql_guardrail import UnsafeSQLError, validate_sql
from web_tools import WEB_TOOLS

_log = logger.bind(component="agent")

# Keep the last 3 Q/A pairs per session (6 entries).
_HISTORY_MAX = 6
_history: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=_HISTORY_MAX))


class SafeSQLDatabase(SQLDatabase):
    """SQLDatabase that runs every query through the guardrail first.

    Any attempt to execute non-SELECT SQL (INSERT/UPDATE/DROP/etc.)
    is blocked before it reaches Postgres and is returned to the agent
    as an error string — which the LLM then incorporates into its
    reasoning instead of hitting the database.
    """

    def run(self, command, *args, **kwargs):  # type: ignore[override]
        try:
            validate_sql(command)
        except UnsafeSQLError as exc:
            _log.bind(sql=command[:200]).warning("guardrail blocked query: {}", exc)
            return f"ERROR: query blocked by guardrail ({exc})."
        _log.bind(sql=command[:200]).debug("executing sql")
        return super().run(command, *args, **kwargs)

    def run_no_throw(self, command, *args, **kwargs):  # type: ignore[override]
        try:
            validate_sql(command)
        except UnsafeSQLError as exc:
            _log.bind(sql=command[:200]).warning("guardrail blocked query: {}", exc)
            return f"ERROR: query blocked by guardrail ({exc})."
        return super().run_no_throw(command, *args, **kwargs)


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
            max_iterations=20,
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
    history = _history.get(session_id)
    if not history:
        return question
    prior = "\n".join(history)
    return f"Prior conversation:\n{prior}\n\nCurrent question: {question}"


def clear_session(session_id: str) -> None:
    """Forget all prior turns for a session."""
    _history.pop(session_id, None)


def ask(question: str, session_id: str | None = None) -> dict:
    """Run a single question through the agent.

    * Stateless calls (``session_id is None``) hit the LRU cache first.
    * Sessioned calls skip the cache — prior context is part of the key
      but we keep the cache simple and only memoize stateless lookups.
    """
    if session_id is None:
        cached = CACHE.get(question)
        if cached is not None:
            cached["cached"] = True
            _log.bind(question=question).info("cache hit")
            return cached

    cb = TokenUsageCallback()
    prompt_input = _build_input(question, session_id)

    try:
        result = get_agent().invoke(
            {"input": prompt_input},
            config={"callbacks": [cb]},
        )
    except Exception:
        METRICS.record_query(cb.prompt_tokens, cb.completion_tokens, failed=True)
        raise

    answer = result.get("output", "No answer.")
    steps = _serialize_steps(result.get("intermediate_steps"))

    METRICS.record_query(cb.prompt_tokens, cb.completion_tokens, failed=False)

    if session_id:
        _history[session_id].append(f"User: {question}")
        _history[session_id].append(f"Assistant: {answer}")

    response = {
        "question": question,
        "answer": answer,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "steps": steps,
        "cached": False,
    }

    if session_id is None:
        CACHE.set(question, response)

    return response


if __name__ == "__main__":
    import json

    print(json.dumps(ask("How many orders are in the database?"), indent=2))
