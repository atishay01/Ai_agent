"""Typed application configuration.

Single source of truth for environment variables. Every module that
needed `os.getenv(...)` should import `settings` from here instead.

Values are validated at import time: missing/invalid config fails fast
with a clear error, rather than blowing up mid-request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All runtime configuration for the Olist chatbot."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Postgres (admin — used by the ETL) -------------------------
    pg_host: str = Field(default="localhost")
    pg_port: int = Field(default=5432)
    pg_user: str = Field(default="postgres")
    pg_password: SecretStr = Field(default=SecretStr(""))
    pg_database: str = Field(default="olist_db")

    # --- Postgres (read-only — used by the agent + dashboard) ------
    # Defence-in-depth: even if the SQL guardrail had a bug, this role
    # only has SELECT permission. Falls back to ``pg_user`` if unset so
    # dev/test setups keep working without seeding the role first.
    # Seed with: psql -U postgres -d olist_db -f src/seed_readonly_user.sql
    pg_user_agent: str = Field(default="")
    pg_password_agent: SecretStr = Field(default=SecretStr(""))

    # --- Groq / LLM -------------------------------------------------
    groq_api_key: SecretStr = Field(default=SecretStr(""))
    groq_model: str = Field(default="openai/gpt-oss-20b")

    # --- API --------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # --- CORS -------------------------------------------------------
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Auth / rate-limit ------------------------------------------
    # Empty string disables API-key auth (dev default). Production
    # deployments should set a strong random value.
    api_key: SecretStr = Field(default=SecretStr(""))
    # Rate limit for /query (per client IP). Format: "<count>/<window>",
    # e.g. "30/minute", "500/hour". Empty string disables rate limiting.
    rate_limit: str = Field(default="30/minute")

    # --- Agent runtime ---------------------------------------------
    # Hard cap on tool-calling iterations per question.
    agent_max_iterations: int = Field(default=20)
    # How many user/assistant *turns* to keep in per-session memory
    # (one turn = one user msg + one assistant msg → 2 history entries).
    session_history_turns: int = Field(default=6)
    # In-memory response cache size (entries).
    cache_max_size: int = Field(default=128)

    # --- SQL guardrail ----------------------------------------------
    # Auto-append `LIMIT <n>` to any SELECT that has no LIMIT, so a
    # buggy or coerced agent query can't dump whole tables.
    sql_max_rows: int = Field(default=1000)

    # --- State store (sessions + response cache) -------------------
    # Path to the SQLite file backing per-session memory and the LRU
    # response cache. Default is a file under the project's data/
    # directory so state survives restarts. Set to ":memory:" to keep
    # state in-process only (used by the test suite).
    state_db_path: str = Field(default="data/state.db")

    # --- Web tools --------------------------------------------------
    # How long to cache the BRL/USD exchange rate (seconds). Frankfurter
    # only updates daily, so anything <24h is fine.
    exchange_rate_ttl_seconds: int = Field(default=3600)

    # --- Cost estimation (placeholders for free Groq tier) ----------
    # USD per 1M tokens. Edit when switching to a priced LLM endpoint.
    prompt_cost_per_1m_usd: float = Field(default=0.05)
    completion_cost_per_1m_usd: float = Field(default=0.10)

    # ----------------------------------------------------------------
    @property
    def database_url(self) -> str:
        """SQLAlchemy connection URL — admin user (write access).

        Used by the ETL, which needs CREATE/DROP/INSERT.
        """
        return (
            f"postgresql+psycopg2://{self.pg_user}:"
            f"{self.pg_password.get_secret_value()}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def agent_database_url(self) -> str:
        """SQLAlchemy connection URL — read-only role.

        Used by the agent and the dashboard. If ``pg_user_agent`` is
        unset (dev mode, no seed), falls back to the admin URL so the
        project still runs.
        """
        if not self.pg_user_agent:
            return self.database_url
        return (
            f"postgresql+psycopg2://{self.pg_user_agent}:"
            f"{self.pg_password_agent.get_secret_value()}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


settings = get_settings()
