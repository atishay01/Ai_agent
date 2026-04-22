"""Golden-set evaluation harness for the Olist SQL agent.

Usage
-----

    # From project root:
    python eval/run_eval.py                  # run full suite
    python eval/run_eval.py --only orders_count,top_revenue_category
    python eval/run_eval.py --golden eval/golden.yaml
    python eval/run_eval.py --json report.json

Exits non-zero if any case fails, so this script is safe to drop into CI
once Postgres + Groq are reachable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass
class CaseResult:
    id: str
    question: str
    passed: bool
    answer: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    missing: list[str] = field(default_factory=list)
    slow: bool = False
    error: str | None = None


def _normalize(text: str) -> str:
    # Collapse all Unicode whitespace (incl. narrow no-break space U+202F)
    # to single ASCII spaces so "São Paulo" matches "São\u202fPaulo".
    return " ".join(text.split()).lower()


def score_case(expected: list[str], answer: str) -> list[str]:
    """Return the list of expected substrings NOT found in answer (case-insensitive)."""
    hay = _normalize(answer)
    return [needle for needle in expected if _normalize(needle) not in hay]


def run_case(case: dict, ask_fn: Callable[..., dict]) -> CaseResult:
    question = case["question"]
    expected = list(case.get("contains") or [])
    max_latency = case.get("max_latency_ms")

    start = time.perf_counter()
    try:
        result = ask_fn(question, session_id=f"eval-{uuid.uuid4().hex[:8]}")
    except Exception as exc:
        elapsed = int((time.perf_counter() - start) * 1000)
        return CaseResult(
            id=case["id"],
            question=question,
            passed=False,
            answer="",
            latency_ms=elapsed,
            prompt_tokens=0,
            completion_tokens=0,
            missing=expected,
            error=repr(exc),
        )
    elapsed = int((time.perf_counter() - start) * 1000)

    answer = result.get("answer", "")
    missing = score_case(expected, answer)
    slow = bool(max_latency and elapsed > max_latency)
    passed = not missing

    return CaseResult(
        id=case["id"],
        question=question,
        passed=passed,
        answer=answer,
        latency_ms=elapsed,
        prompt_tokens=result.get("prompt_tokens", 0),
        completion_tokens=result.get("completion_tokens", 0),
        missing=missing,
        slow=slow,
    )


def load_golden(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or []
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list at the top level")
    return data


def _fmt_report(results: list[CaseResult]) -> str:
    lines = ["", "=" * 72, "GOLDEN EVALUATION REPORT", "=" * 72]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        slow_tag = " [SLOW]" if r.slow else ""
        lines.append(
            f"[{status}] {r.id:30s}  {r.latency_ms:>6d} ms  "
            f"{r.prompt_tokens + r.completion_tokens:>5d} tok{slow_tag}"
        )
        if not r.passed:
            if r.error:
                lines.append(f"        error: {r.error}")
            if r.missing:
                lines.append(f"        missing: {r.missing!r}")
            preview = (r.answer or "").replace("\n", " ")[:120]
            lines.append(f"        answer: {preview!r}")

    n = len(results)
    passed = sum(1 for r in results if r.passed)
    slow = sum(1 for r in results if r.slow)
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in results)
    avg_latency = (sum(r.latency_ms for r in results) // n) if n else 0

    # Cheap placeholder cost per 1M tokens — matches src/metrics.py defaults.
    cost_usd = total_tokens / 1_000_000 * (0.05 + 0.10) / 2

    lines.append("-" * 72)
    lines.append(
        f"Passed: {passed}/{n} ({(passed / n * 100 if n else 0):.1f}%)    "
        f"Slow: {slow}    Avg latency: {avg_latency} ms    "
        f"Total tokens: {total_tokens} (~${cost_usd:.4f})"
    )
    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(__file__).resolve().parent / "golden.yaml",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of case ids to run.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write machine-readable results JSON to this path.",
    )
    args = parser.parse_args(argv)

    cases = load_golden(args.golden)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        cases = [c for c in cases if c.get("id") in wanted]
        if not cases:
            print(f"No cases matched --only={args.only}", file=sys.stderr)
            return 2

    from agent import ask  # import lazily so --help doesn't need Postgres/Groq

    results = [run_case(c, ask) for c in cases]
    print(_fmt_report(results))

    if args.json:
        args.json.write_text(
            json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
