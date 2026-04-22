"""Database connection helper.

Centralises the Postgres connection so every script reads credentials
from a single typed `Settings` object (see `config.py`).
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import settings


def get_engine() -> Engine:
    """Return a SQLAlchemy engine for the Olist database."""
    return create_engine(settings.database_url, pool_pre_ping=True)


def get_connection_string() -> str:
    """Plain connection URL (used by LangChain's SQLDatabase)."""
    return settings.database_url
