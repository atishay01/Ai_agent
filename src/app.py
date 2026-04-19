"""
Streamlit chat UI
=================

Minimal chat interface. User types a question, Streamlit POSTs it to
the FastAPI backend, backend calls the LangChain agent, answer comes
back and is shown in the chat transcript.

Run:  streamlit run src/app.py   (from project root)
"""

from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://localhost:8000/query"

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Olist Data Chatbot",
    page_icon=":bar_chart:",
    layout="centered",
)

st.title("Olist E-Commerce Data Assistant")
st.caption("Ask business questions in plain English. "
           "Powered by LangChain + Groq + PostgreSQL. "
           "Includes live currency conversion and Wikipedia scraping.")

# ---------------------------------------------------------------------
# Sidebar — sample questions + info
# ---------------------------------------------------------------------
with st.sidebar:
    st.subheader("Try asking:")
    samples = [
        "How many orders are in the database?",
        "Top 5 product categories by revenue?",
        "Which state has the most late deliveries?",
        "Top 3 payment types by total value?",
        "Which sellers received the most 5-star reviews?",
        "What is total revenue from all orders in US dollars?",   # web: currency
        "Top 5 states by revenue — show full names and capitals.", # web: scraper
        "Average order value in USD for state SP?",                # web: both
    ]
    for s in samples:
        if st.button(s, use_container_width=True):
            st.session_state["pending_question"] = s

    st.divider()
    st.markdown("**Tables loaded (8):**")
    st.code("customers         99,441\n"
            "sellers            3,095\n"
            "products          32,951\n"
            "geolocation       19,015\n"
            "orders            99,441\n"
            "order_items      112,650\n"
            "order_payments   103,886\n"
            "order_reviews     99,224\n"
            "-------------------------\n"
            "TOTAL            569,703",
            language="text")

# ---------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------
# Input — either a clicked sample or typed question
# ---------------------------------------------------------------------
typed = st.chat_input("Ask a question about Olist data...")
question = st.session_state.pop("pending_question", None) or typed

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(API_URL, json={"question": question}, timeout=120)
                r.raise_for_status()
                answer = r.json()["answer"]
            except requests.RequestException as e:
                answer = (f"**Error contacting backend:** {e}\n\n"
                          "Make sure the FastAPI server is running:\n"
                          "`python src/api.py`")
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
