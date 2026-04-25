"""Per-session conversation history, persisted to SQLite.

Each entry is a single user/assistant line ("User: ..." or
"Assistant: ..."). When the count for a session exceeds
``settings.session_history_turns * 2``, the oldest entries are pruned
on the next ``append`` so memory is bounded and ordered by insert seq.

Storage lives in the same SQLite file as the response cache (see
``state_store``). A docker redeploy or API restart no longer wipes
multi-turn context.
"""

from __future__ import annotations

from config import settings
from state_store import transaction


def _max_entries() -> int:
    """One turn = user msg + assistant msg → 2 entries."""
    return settings.session_history_turns * 2


def append(session_id: str, line: str) -> None:
    """Append a line to a session, pruning the oldest if over the cap."""
    with transaction() as conn:
        next_seq = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM session_history " "WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO session_history(session_id, seq, line) VALUES (?, ?, ?)",
            (session_id, next_seq, line),
        )
        cap = _max_entries()
        count = conn.execute(
            "SELECT COUNT(*) FROM session_history WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        if count > cap:
            excess = count - cap
            conn.execute(
                "DELETE FROM session_history WHERE rowid IN ("
                "SELECT rowid FROM session_history WHERE session_id = ? "
                "ORDER BY seq ASC LIMIT ?)",
                (session_id, excess),
            )


def get(session_id: str) -> list[str]:
    with transaction() as conn:
        rows = conn.execute(
            "SELECT line FROM session_history WHERE session_id = ? ORDER BY seq ASC",
            (session_id,),
        ).fetchall()
    return [r["line"] for r in rows]


def clear(session_id: str) -> None:
    with transaction() as conn:
        conn.execute("DELETE FROM session_history WHERE session_id = ?", (session_id,))


def clear_all() -> None:
    with transaction() as conn:
        conn.execute("DELETE FROM session_history")
