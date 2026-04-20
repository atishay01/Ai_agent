"""
Streamlit UI for the Olist data chatbot.

Two views in one app:
  * Chat      — natural-language questions answered by the LangChain SQL agent
                via the FastAPI /query endpoint.
  * Dashboard — business-KPI view that queries Postgres directly (no LLM).
"""

from __future__ import annotations

import requests
import streamlit as st

import dashboard

API_URL = "http://localhost:8000/query"

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

    if not st.session_state.messages:
        st.info(
            "👋 Welcome! Pick a sample question from the sidebar, "
            "or type your own question below. "
            "I can query the database and look up external data when needed."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed = st.chat_input("Ask a question about the Olist data...")
    question = st.session_state.pop("pending_question", None) or typed

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    r = requests.post(API_URL, json={"question": question}, timeout=180)
                    r.raise_for_status()
                    answer = r.json()["answer"]
                except requests.RequestException as e:
                    answer = (
                        f"**Could not reach the backend.**\n\n`{e}`\n\n"
                        "Make sure FastAPI is running: `python src/api.py`"
                    )
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

# =====================================================================
# Dashboard tab
# =====================================================================
with tab_dashboard:
    dashboard.render()
