"""Shared pytest fixtures.

Crucially, this file is imported *before* any test module, so it's the
right place to set deterministic env vars (API_KEY, RATE_LIMIT, etc.)
and make sure `src/` is on ``sys.path``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `src/` importable without relying on editable installs.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Deterministic test environment — applied before settings is first read.
os.environ.setdefault("API_KEY", "")
os.environ.setdefault("RATE_LIMIT", "1000/minute")
os.environ.setdefault("GROQ_API_KEY", "test-key-unused")
os.environ.setdefault("PG_PASSWORD", "dummy")
os.environ.setdefault("LOG_LEVEL", "WARNING")
