"""Settings smoke tests."""

from pydantic import SecretStr

from config import Settings


def test_settings_defaults_loadable() -> None:
    s = Settings(_env_file=None)
    assert s.api_port == 8000
    assert isinstance(s.pg_password, SecretStr)
    assert isinstance(s.groq_api_key, SecretStr)


def test_database_url_uses_secrets() -> None:
    s = Settings(
        _env_file=None,
        pg_user="alice",
        pg_password="s3cret",
        pg_host="db.example.com",
        pg_port=5433,
        pg_database="olist",
    )
    url = s.database_url
    assert "alice" in url
    assert "s3cret" in url
    assert "db.example.com:5433" in url
    assert url.startswith("postgresql+psycopg2://")


def test_agent_database_url_uses_readonly_creds_when_set() -> None:
    s = Settings(
        _env_file=None,
        pg_user="alice",
        pg_password="admin_secret",
        pg_user_agent="bob_readonly",
        pg_password_agent="ro_secret",
        pg_host="db.example.com",
        pg_port=5433,
        pg_database="olist",
    )
    url = s.agent_database_url
    assert "bob_readonly" in url
    assert "ro_secret" in url
    # Make sure the admin password did NOT bleed through.
    assert "admin_secret" not in url


def test_agent_database_url_falls_back_to_admin_when_unset() -> None:
    """Dev-mode convenience: no readonly role configured → use admin."""
    s = Settings(
        _env_file=None,
        pg_user="alice",
        pg_password="s3cret",
        pg_host="db.example.com",
        pg_port=5433,
        pg_database="olist",
    )
    # pg_user_agent left at default ("") → should fall back.
    assert s.agent_database_url == s.database_url


def test_secret_repr_hides_value() -> None:
    s = Settings(_env_file=None, pg_password="do-not-leak", groq_api_key="do-not-leak")
    # SecretStr shows '**********' in repr
    assert "do-not-leak" not in repr(s)
