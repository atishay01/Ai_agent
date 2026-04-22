"""Tests for TokenUsageCallback — covers every shape LangChain providers use."""

from __future__ import annotations

from types import SimpleNamespace


def _response(llm_output: dict | None = None, generations=None):
    """Build a duck-typed LLMResult for testing."""
    return SimpleNamespace(
        llm_output=llm_output or {},
        generations=generations or [],
    )


def test_reads_llm_output_token_usage() -> None:
    from callbacks import TokenUsageCallback

    cb = TokenUsageCallback()
    cb.on_llm_end(
        _response(llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    )
    assert cb.prompt_tokens == 10
    assert cb.completion_tokens == 5


def test_reads_llm_output_usage_fallback_key() -> None:
    from callbacks import TokenUsageCallback

    cb = TokenUsageCallback()
    cb.on_llm_end(_response(llm_output={"usage": {"prompt_tokens": 3, "completion_tokens": 4}}))
    assert cb.prompt_tokens == 3
    assert cb.completion_tokens == 4


def test_reads_usage_metadata_from_generations() -> None:
    """Groq/OpenAI chat path — usage sits on AIMessage.usage_metadata."""
    from callbacks import TokenUsageCallback

    msg = SimpleNamespace(
        usage_metadata={"input_tokens": 7, "output_tokens": 3},
        response_metadata={},
    )
    gen = SimpleNamespace(message=msg)
    cb = TokenUsageCallback()
    cb.on_llm_end(_response(generations=[[gen]]))
    assert cb.prompt_tokens == 7
    assert cb.completion_tokens == 3


def test_reads_response_metadata_token_usage() -> None:
    from callbacks import TokenUsageCallback

    msg = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"token_usage": {"prompt_tokens": 4, "completion_tokens": 2}},
    )
    gen = SimpleNamespace(message=msg)
    cb = TokenUsageCallback()
    cb.on_llm_end(_response(generations=[[gen]]))
    assert cb.prompt_tokens == 4
    assert cb.completion_tokens == 2


def test_llm_output_wins_over_generations() -> None:
    """Don't double-count: if llm_output has usage, skip per-generation."""
    from callbacks import TokenUsageCallback

    msg = SimpleNamespace(
        usage_metadata={"input_tokens": 99, "output_tokens": 99},
        response_metadata={},
    )
    gen = SimpleNamespace(message=msg)
    cb = TokenUsageCallback()
    cb.on_llm_end(
        _response(
            llm_output={"token_usage": {"prompt_tokens": 1, "completion_tokens": 2}},
            generations=[[gen]],
        )
    )
    assert cb.prompt_tokens == 1
    assert cb.completion_tokens == 2


def test_accumulates_across_multiple_calls() -> None:
    from callbacks import TokenUsageCallback

    cb = TokenUsageCallback()
    cb.on_llm_end(
        _response(llm_output={"token_usage": {"prompt_tokens": 5, "completion_tokens": 1}})
    )
    cb.on_llm_end(
        _response(llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 2}})
    )
    assert cb.prompt_tokens == 15
    assert cb.completion_tokens == 3


def test_handles_missing_usage_gracefully() -> None:
    from callbacks import TokenUsageCallback

    cb = TokenUsageCallback()
    cb.on_llm_end(_response())
    assert cb.prompt_tokens == 0
    assert cb.completion_tokens == 0
