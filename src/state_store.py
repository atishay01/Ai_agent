"""Process-shared SQLite store for session memory and response cache.

A single connection serves both ``ResponseCache`` (cache.py) and the
session-history helpers (session_history.py). The connection lives at
the path given by ``settings.state_db_path``:

  * Default ``data/state.db`` — survives process restarts and can be
    pointed at a docker volume so a container redeploy doesn't wipe
    chat history.
  * ``:memory:`` — used by the test suite. Note that an in-memory
    SQLite DB is bound to its connection, but since this module owns
    the *only* connection, all callers share the same DB.

Concurrency: every write acquires ``_lock`` so we never have two
threads commit at once. ``check_same_thread=False`` lets the FastAPI
worker pool reuse the connection. WAL mode is enabled on file paths
so concurrent readers don't block on a writer.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from config import PROJECT_ROOT, settings

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_seq_counter: int = 0
_seq_lock = threading.Lock()


SCHEMA = """
CREATE TABLE IF NOT EXISTS session_history (
    session_id TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    line       TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
);

CREATE TABLE IF NOT EXISTS response_cache (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    accessed_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_response_cache_accessed
    ON response_cache(accessed_at);
"""


def _resolve_path() -> str:
    p = settings.state_db_path
    if p == ":memory:":
        return p
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / p
    return str(path)


def _init_seq_counter(conn: sqlite3.Connection) -> None:
    """Resume the LRU counter from the high-water mark of accessed_at."""
    global _seq_counter
    row = conn.execute("SELECT COALESCE(MAX(accessed_at), 0) FROM response_cache").fetchone()
    _seq_counter = int(row[0])


def get_conn() -> sqlite3.Connection:
    """Lazy-init the shared SQLite connection (single instance per process)."""
    global _conn
    if _conn is not None:
        return _conn
    with _lock:
        if _conn is not None:
            return _conn
        path = _resolve_path()
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        if path != ":memory:":
            # WAL gives us non-blocking reads alongside a writer.
            conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        conn.commit()
        _init_seq_counter(conn)
        _conn = conn
        return _conn


@contextmanager
def transaction():
    """Acquire the connection under the write lock with auto-commit/rollback."""
    conn = get_conn()
    with _lock:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def next_seq() -> int:
    """Monotonically increasing access counter for LRU eviction order."""
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


def reset_for_tests() -> None:
    """Wipe both tables and reset the seq counter. Test isolation only."""
    global _seq_counter
    with transaction() as conn:
        conn.execute("DELETE FROM session_history")
        conn.execute("DELETE FROM response_cache")
    with _seq_lock:
        _seq_counter = 0


def close() -> None:
    """Close the connection (used by tests that swap the DB path)."""
    global _conn, _seq_counter
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
        _seq_counter = 0
