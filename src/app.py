"""
Streamlit UI for the Olist data chatbot.

Two views in one app:
  * Chat      — natural-language questions answered by the LangChain SQL agent
                via the FastAPI /query endpoint.
  * Dashboard — business-KPI view that queries Postgres directly (no LLM).
"""

from __future__ import annotations

import ast
import contextlib
import os
import re
import uuid
from datetime import date, datetime

import pandas as pd
import requests
import streamlit as st

import dashboard

API_URL = os.environ.get("API_URL", "http://localhost:8000/query")
SESSION_DELETE_URL = os.environ.get("SESSION_DELETE_URL", "http://localhost:8000/session")


# ---------------------------------------------------------------------
# Chart auto-detection from agent intermediate steps.
#
# When the agent runs a SELECT, LangChain returns the result as a
# string-repr of a list of tuples, e.g. "[('2017-01', 134567.89), ...]".
# If that result has a date-or-text first column and a numeric second
# column, we render it as a line or bar chart inline with the answer.
# ---------------------------------------------------------------------
def _parse_observation(obs: str) -> list[tuple] | None:
    """Try to recover a list of tuples from the agent's tool observation.

    LangChain's SQLDatabase returns rows as ``str(list_of_tuples)``. We
    re-parse defensively — any failure means we silently skip charting.
    """
    if not obs or not obs.strip().startswith("["):
        return None
    try:
        parsed = ast.literal_eval(obs.strip())
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    # Every row should be a tuple/list of the same length.
    if not all(isinstance(r, tuple | list) for r in parsed):
        return None
    widths = {len(r) for r in parsed}
    if len(widths) != 1 or widths == {1}:
        return None
    return [tuple(r) for r in parsed]


def _looks_like_date(value) -> bool:
    if isinstance(value, date | datetime):
        return True
    if isinstance(value, str):
        # 2017-01, 2017-01-15, 2017
        return bool(re.match(r"^\d{4}(-\d{2})?(-\d{2})?$", value.strip()))
    return False


def _last_sql_result(steps: list[dict]) -> tuple[str | None, list[tuple] | None]:
    """Return (sql_text, parsed_rows) from the most recent SQL step."""
    for step in reversed(steps or []):
        tool = (step.get("tool") or "").lower()
        if "sql" not in tool and "query" not in tool:
            continue
        rows = _parse_observation(step.get("observation", ""))
        if rows:
            return step.get("tool_input", ""), rows
    return None, None


def render_chart_if_useful(steps: list[dict]) -> None:
    """If the agent's last SQL result is chartable, render a chart.

    Heuristics, kept conservative (false-positive charts annoy more
    than they help):

      * line chart  -> first column looks like a date/month, ≥ 3 rows,
                       second column is numeric
      * bar chart   -> first column is text, ≤ 20 rows, second column
                       is numeric
      * otherwise   -> no chart
    """
    _, rows = _last_sql_result(steps)
    if not rows or len(rows) < 2:
        return

    first_col = [r[0] for r in rows]
    second_col = [r[1] for r in rows]

    # Second column must be numeric for either chart type.
    if not all(isinstance(v, int | float) for v in second_col):
        return

    df = pd.DataFrame(rows).iloc[:, :2]
    df.columns = ["x", "y"]

    # Time-series? First column dates → line chart.
    if len(rows) >= 3 and all(_looks_like_date(v) for v in first_col):
        df = df.sort_values("x")
        st.line_chart(df.set_index("x"), height=260)
        return

    # Categorical ranking? Text labels with bounded count → bar chart.
    if len(rows) <= 20 and all(isinstance(v, str) for v in first_col):
        st.bar_chart(df.set_index("x"), height=260, horizontal=True)
        return

    # Otherwise leave the trace panel as the only visualisation.


# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Olist Data Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🛒 Olist E-Commerce Data Assistant")
st.markdown(
    "Ask business questions in plain English about "
    "**~570,000 rows** of Brazilian e-commerce data, or open the "
    "**Dashboard** tab for a BI-style KPI view. "
    "Powered by LangChain + Groq + PostgreSQL, with live currency "
    "conversion and Wikipedia lookups."
)
st.divider()

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Try a sample question")

    st.caption("🗃️ **Database queries**")
    db_samples = [
        "How many orders are in the database?",
        "Top 5 product categories by revenue?",
        "Which state has the most late deliveries?",
        "Top 3 payment types by total value?",
        "Which sellers received the most 5-star reviews?",
    ]
    for s in db_samples:
        if st.button(s, key=f"db_{s}", use_container_width=True):
            st.session_state["pending_question"] = s

    st.caption("🌐 **With live web data**")
    web_samples = [
        "Total revenue in US dollars?",
        "What is the capital of state RJ?",
        "Top 5 states by revenue. Show full names and capitals.",
    ]
    for s in web_samples:
        if st.button(s, key=f"web_{s}", use_container_width=True):
            st.session_state["pending_question"] = s

    st.divider()
    if st.button("🔄 Clear chat", use_container_width=True):
        st.session_state.messages = []
        # Rotate the session id so the backend drops prior-turn context too.
        old_sid = st.session_state.get("session_id")
        if old_sid:
            with contextlib.suppress(requests.RequestException):
                requests.delete(f"{SESSION_DELETE_URL}/{old_sid}", timeout=5)
        st.session_state["session_id"] = uuid.uuid4().hex
        st.rerun()

    with st.expander("📊 Tables loaded (8)", expanded=False):
        st.code(
            "customers        99,441\n"
            "sellers           3,095\n"
            "products         32,951\n"
            "geolocation      19,015\n"
            "orders           99,441\n"
            "order_items     112,650\n"
            "order_payments  103,886\n"
            "order_reviews    99,224\n"
            "----------------------\n"
            "TOTAL           569,703",
            language="text",
        )

    st.divider()
    st.caption(
        "🔗 [GitHub](https://github.com/atishay01/Ai_agent)  ·  "
        "Built with Python, FastAPI, LangChain"
    )

# ---------------------------------------------------------------------
# Tabs: Chat vs Dashboard
# ---------------------------------------------------------------------
tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Dashboard"])

# =====================================================================
# Chat tab (existing behaviour)
# =====================================================================
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex

    if not st.session_state.messages:
        st.info(
            "👋 Welcome! Pick a sample question from the sidebar, "
            "or type your own question below. "
            "I can query the database and look up external data when needed."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                meta = msg.get("meta") or {}
                if meta:
                    badges = []
                    if meta.get("cached"):
                        badges.append("🟢 cached")
                    if meta.get("latency_ms") is not None:
                        badges.append(f"⏱ {meta['latency_ms']} ms")
                    tt = (meta.get("prompt_tokens") or 0) + (meta.get("completion_tokens") or 0)
                    if tt:
                        badges.append(f"🔢 {tt} tokens")
                    if badges:
                        st.caption("  ·  ".join(badges))
                steps = msg.get("steps") or []
                # Auto-chart: render a line/bar inline if the last SQL
                # result has a chartable shape. Silent no-op otherwise.
                if steps:
                    render_chart_if_useful(steps)
                if steps:
                    with st.expander(
                        f"🔍 Trace ({len(steps)} step{'s' if len(steps) != 1 else ''})",
                        expanded=False,
                    ):
                        for i, s in enumerate(steps, 1):
                            tool = s.get("tool", "")
                            st.markdown(f"**Step {i} — `{tool}`**")
                            lang = (
                                "sql"
                                if "sql" in tool.lower() or "query" in tool.lower()
                                else "text"
                            )
                            st.code(s.get("tool_input", ""), language=lang)
                            st.caption("Result:")
                            st.code((s.get("observation") or "")[:500], language="text")

    typed = st.chat_input("Ask a question about the Olist data...")
    question = st.session_state.pop("pending_question", None) or typed

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                data: dict = {}
                try:
                    r = requests.post(
                        API_URL,
                        json={
                            "question": question,
                            "session_id": st.session_state["session_id"],
                        },
                        timeout=180,
                    )
                    r.raise_for_status()
                    data = r.json()
                    answer = data.get("answer", "No answer.")
                except requests.RequestException as e:
                    answer = (
                        f"**Could not reach the backend.**\n\n`{e}`\n\n"
                        "Make sure FastAPI is running: `python src/api.py`"
                    )
            st.markdown(answer)

            meta = {
                "cached": bool(data.get("cached")),
                "latency_ms": data.get("latency_ms"),
                "prompt_tokens": data.get("prompt_tokens", 0),
                "completion_tokens": data.get("completion_tokens", 0),
            }
            steps = data.get("steps") or []
            if meta["cached"] or meta["latency_ms"] is not None or steps:
                badges = []
                if meta["cached"]:
                    badges.append("🟢 cached")
                if meta["latency_ms"] is not None:
                    badges.append(f"⏱ {meta['latency_ms']} ms")
                tt = meta["prompt_tokens"] + meta["completion_tokens"]
                if tt:
                    badges.append(f"🔢 {tt} tokens")
                if badges:
                    st.caption("  ·  ".join(badges))
            # Auto-chart inline (live render) when the result is chartable.
            if steps:
                render_chart_if_useful(steps)
            if steps:
                with st.expander(
                    f"🔍 Trace ({len(steps)} step{'s' if len(steps) != 1 else ''})",
                    expanded=False,
                ):
                    for i, s in enumerate(steps, 1):
                        tool = s.get("tool", "")
                        st.markdown(f"**Step {i} — `{tool}`**")
                        lang = "sql" if "sql" in tool.lower() or "query" in tool.lower() else "text"
                        st.code(s.get("tool_input", ""), language=lang)
                        st.caption("Result:")
                        st.code((s.get("observation") or "")[:500], language="text")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "meta": meta,
                "steps": steps,
            }
        )

# =====================================================================
# Dashboard tab
# =====================================================================
with tab_dashboard:
    dashboard.render()
