"""
LangChain SQL agent: converts natural-language questions into SQL
against the Olist Postgres DB, with extra tools for currency conversion,
state lookup and exact arithmetic.
"""

import os

from dotenv import load_dotenv
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

from db import get_connection_string
from web_tools import WEB_TOOLS

load_dotenv()


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
"""


def _db():
    return SQLDatabase.from_uri(
        get_connection_string(),
        include_tables=[
            "customers", "sellers", "products", "geolocation",
            "orders", "order_items", "order_payments", "order_reviews",
        ],
        sample_rows_in_table_info=2,
    )


def _llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
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
            max_iterations=10,
        )
    return _agent


def ask(question: str) -> dict:
    """Run a single question through the agent."""
    result = get_agent().invoke({"input": question})
    return {"question": question,
            "answer":   result.get("output", "No answer.")}


if __name__ == "__main__":
    import json
    print(json.dumps(ask("How many orders are in the database?"), indent=2))
