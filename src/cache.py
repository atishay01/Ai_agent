"""SQLite-backed LRU cache for agent responses.

Keyed by a normalized form of the question (lower-cased, punctuation
stripped, whitespace collapsed). Storage is delegated to
``state_store`` so the cache survives process restarts — a redeploy
no longer forces every user to re-pay the LLM cost for FAQ-style
questions.

LRU semantics are implemented with a monotonically increasing
``accessed_at`` integer (``state_store.next_seq``). The counter is
seeded at startup from the maximum value already in the table so
ordering remains correct after a restart.
"""

from __future__ import annotations

import json
import string

from config import settings
from metrics import METRICS
from state_store import next_seq, transaction


class ResponseCache:
    """Bounded LRU map from normalized question → response dict, on disk."""

    _PUNCT_TABLE = str.maketrans("", "", string.punctuation)

    def __init__(self, max_size: int = 128) -> None:
        self._max = max_size

    @classmethod
    def _key(cls, question: str) -> str:
        cleaned = question.lower().translate(cls._PUNCT_TABLE)
        return " ".join(cleaned.split())

    def get(self, question: str) -> dict | None:
        k = self._key(question)
        with transaction() as conn:
            row = conn.execute("SELECT value FROM response_cache WHERE key = ?", (k,)).fetchone()
            if row is not None:
                # Promote on access — bump accessed_at so LRU stays correct.
                conn.execute(
                    "UPDATE response_cache SET accessed_at = ? WHERE key = ?",
                    (next_seq(), k),
                )
                METRICS.record_cache(hit=True)
                return json.loads(row["value"])
            METRICS.record_cache(hit=False)
            return None

    def set(self, question: str, value: dict) -> None:
        k = self._key(question)
        payload = json.dumps(value)
        seq = next_seq()
        with transaction() as conn:
            conn.execute(
                "INSERT INTO response_cache(key, value, accessed_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "value = excluded.value, accessed_at = excluded.accessed_at",
                (k, payload, seq),
            )
            count = conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0]
            if count > self._max:
                excess = count - self._max
                conn.execute(
                    "DELETE FROM response_cache WHERE key IN ("
                    "SELECT key FROM response_cache "
                    "ORDER BY accessed_at ASC LIMIT ?)",
                    (excess,),
                )

    def clear(self) -> None:
        with transaction() as conn:
            conn.execute("DELETE FROM response_cache")

    def __len__(self) -> int:
        with transaction() as conn:
            return conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0]


CACHE = ResponseCache(max_size=settings.cache_max_size)
