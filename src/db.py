"""Database connection helper.

Centralises the Postgres connection string so every script
reads credentials from .env the same way.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()


def get_engine() -> Engine:
    """Return a SQLAlchemy engine for the Olist database."""
    url = (
        f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
        f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
    )
    return create_engine(url, pool_pre_ping=True)


def get_connection_string() -> str:
    """Plain connection URL (used by LangChain's SQLDatabase)."""
    return (
        f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
        f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
    )
