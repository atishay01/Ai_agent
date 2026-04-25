"""In-memory LRU cache for agent responses.

Keyed by a normalized form of the question (whitespace collapsed,
lower-cased, trailing punctuation stripped). Small, thread-safe,
in-process only — good enough to silence the free-tier Groq limit
when the user re-asks the same question, without requiring Redis.
"""

from __future__ import annotations

import string
from collections import OrderedDict
from threading import Lock

from config import settings
from metrics import METRICS


class ResponseCache:
    """Bounded LRU map from normalized question → response dict."""

    def __init__(self, max_size: int = 128) -> None:
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._max = max_size
        self._lock = Lock()

    _PUNCT_TABLE = str.maketrans("", "", string.punctuation)

    @classmethod
    def _key(cls, question: str) -> str:
        # Lowercase, strip punctuation, collapse whitespace. So
        # "How many orders?" and "how, many orders" hit the same slot.
        cleaned = question.lower().translate(cls._PUNCT_TABLE)
        return " ".join(cleaned.split())

    def get(self, question: str) -> dict | None:
        k = self._key(question)
        with self._lock:
            if k in self._store:
                self._store.move_to_end(k)
                METRICS.record_cache(hit=True)
                return dict(self._store[k])
            METRICS.record_cache(hit=False)
            return None

    def set(self, question: str, value: dict) -> None:
        k = self._key(question)
        with self._lock:
            self._store[k] = dict(value)
            self._store.move_to_end(k)
            if len(self._store) > self._max:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


CACHE = ResponseCache(max_size=settings.cache_max_size)
