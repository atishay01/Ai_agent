"""Database connection helper.

Centralises the Postgres connection so every script reads credentials
from a single typed `Settings` object (see `config.py`).

Two roles are exposed:

  * ``get_engine()`` / ``get_connection_string()`` — admin user.
    Used by the ETL because it CREATE/DROP/INSERTs.
  * ``get_agent_engine()`` / ``get_agent_connection_string()`` —
    read-only role. Used by the agent and the dashboard. If
    ``PG_USER_AGENT`` is unset, falls back to the admin URL so dev
    setups without the seed user still work.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import settings


# --- admin role (ETL only) ---------------------------------------------
def get_engine() -> Engine:
    """SQLAlchemy engine with write access. Use for the ETL."""
    return create_engine(settings.database_url, pool_pre_ping=True)


def get_connection_string() -> str:
    """Plain admin connection URL."""
    return settings.database_url


# --- read-only role (agent + dashboard) --------------------------------
def get_agent_engine() -> Engine:
    """SQLAlchemy engine with read-only role. Use for the dashboard."""
    return create_engine(settings.agent_database_url, pool_pre_ping=True)


def get_agent_connection_string() -> str:
    """Plain read-only URL — passed to LangChain's SQLDatabase."""
    return settings.agent_database_url
