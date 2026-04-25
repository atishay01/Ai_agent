"""Centralised loguru configuration.

One call to `configure_logging()` sets up:
  - Pretty, colourised console output with timestamp, level and module.
  - A rotating JSON log file at `logs/app.log` — line-delimited, one
    JSON object per log record, ready to grep/jq/ship to Loki/Datadog.
  - An `interception handler` so that stdlib `logging` calls (used by
    uvicorn, langchain, sqlalchemy, etc.) are routed through loguru
    and formatted identically.

Import once at process start:

    from logging_setup import configure_logging
    configure_logging()
"""

from __future__ import annotations

import hashlib
import logging
import sys

from loguru import logger

from config import PROJECT_ROOT, settings

LOG_DIR = PROJECT_ROOT / "logs"


def redact(value: str | None) -> str:
    """Hash a sensitive value to a 12-char hex digest for log binding.

    User questions and session ids are not safe to write to disk in
    plaintext (PII risk, GDPR posture). Hashing preserves the ability
    to grep-correlate records that share a value across requests
    without persisting the value itself. ``None``/empty maps to ``"-"``
    so log lines stay parseable.
    """
    if not value:
        return "-"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


class _InterceptHandler(logging.Handler):
    """Route stdlib logging through loguru so every log looks the same."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


_configured = False


def configure_logging() -> None:
    """Idempotent logging setup. Safe to call multiple times."""
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()

    # Console (human-readable)
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> "
            "<level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )

    # File (JSON, one record per line; rotate at 10 MB; keep 7 files).
    # diagnose=False is critical: with the default True, loguru would
    # render local-variable values inline in tracebacks — including the
    # unredacted QueryRequest object (question text + session id) — and
    # bypass the redact() helper entirely.
    logger.add(
        LOG_DIR / "app.log",
        level=settings.log_level,
        rotation="10 MB",
        retention=7,
        serialize=True,
        enqueue=True,  # thread/process-safe
        backtrace=False,
        diagnose=False,
    )

    # Redirect stdlib logging -> loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "sqlalchemy.engine",
        "langchain",
        "langchain_community",
        "langchain_groq",
    ):
        lg = logging.getLogger(name)
        lg.handlers = [_InterceptHandler()]
        lg.propagate = False

    _configured = True
    logger.debug("logging configured (level={})", settings.log_level)


__all__ = ["configure_logging", "logger", "LOG_DIR", "redact"]
