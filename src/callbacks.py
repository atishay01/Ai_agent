"""LangChain callback handler that funnels token usage into METRICS."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from logging_setup import logger

_log = logger.bind(component="callbacks")


class TokenUsageCallback(BaseCallbackHandler):
    """Accumulates prompt/completion tokens across an agent invocation.

    The agent may call the LLM several times (tool-calling loop), so we
    sum usage from every ``on_llm_end`` event until the callback is
    discarded at the end of ``ask()``.
    """

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: D401
        try:
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        except Exception as exc:
            _log.warning("token accounting failed: {}", exc)
