"""
FastAPI backend
===============

Thin HTTP layer over the LangChain SQL agent. Streamlit (or any client)
POSTs a natural-language question to /query and gets JSON back.

Run:  uvicorn src.api:app --reload   (from project root)
Or:   python src/api.py              (direct, dev mode)
"""

import time
import uuid
from datetime import UTC, datetime
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent import ask, clear_session
from config import settings
from logging_setup import configure_logging, logger
from metrics import METRICS

configure_logging()
log = logger.bind(component="api")


# ---------------------------------------------------------------------
# Rate limiter — keyed by X-API-Key when present (per-user), otherwise
# by client IP. Behind a load balancer or NAT, IP-only keying lets all
# clients sharing an egress IP exhaust the shared budget; per-key
# limiting isolates each authenticated caller.
# ---------------------------------------------------------------------
def rate_limit_key(request: Request) -> str:
    """Prefer the API key as the rate-limit identity; fall back to remote IP.

    The key is namespaced (``apikey:`` / ``ip:``) so a client with a
    real key can never collide with the literal IP string of another
    client. Keys are not echoed into logs by slowapi — they only flow
    into the in-memory bucket map.
    """
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"apikey:{api_key}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=rate_limit_key, default_limits=[])

# ---------------------------------------------------------------------
app = FastAPI(
    title="Olist AI Data-Query Chatbot",
    description="Ask business questions about the Olist e-commerce database.",
    version="1.3.0",
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Return 429 with a helpful message instead of slowapi's terse default."""
    trace_id = getattr(request.state, "trace_id", "-")
    log.bind(trace_id=trace_id).warning("rate limit hit: {}", exc.detail)
    return _json_error(
        status.HTTP_429_TOO_MANY_REQUESTS,
        f"Rate limit exceeded: {exc.detail}. Slow down and retry.",
        trace_id,
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Auth dependency — optional X-API-Key header.
# ---------------------------------------------------------------------
def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Verify the X-API-Key header matches settings.api_key.

    If no key is configured (dev mode), auth is skipped entirely.
    """
    configured = settings.api_key.get_secret_value()
    if not configured:
        return
    if x_api_key != configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )


# ---------------------------------------------------------------------
# Helper: consistent JSON error envelope
# ---------------------------------------------------------------------
def _json_error(status_code: int, message: str, trace_id: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={"error": message, "trace_id": trace_id},
        headers={"x-trace-id": trace_id},
    )


# ---------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------
class QueryRequest(BaseModel):
    """Single natural-language question to run through the agent."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["What are the top 5 product categories by revenue?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Opaque session key for multi-turn follow-ups. "
        "Leave unset for a one-shot, cacheable question.",
    )


class QueryResponse(BaseModel):
    """Agent answer plus observability metadata."""

    question: str = Field(..., description="Echo of the submitted question.")
    answer: str = Field(..., description="Agent's natural-language answer.")
    trace_id: str = Field(..., description="Unique ID for correlating this request across logs.")
    latency_ms: int = Field(..., ge=0, description="Server-side processing time in milliseconds.")
    timestamp: datetime = Field(..., description="UTC timestamp when the response was produced.")
    prompt_tokens: int = Field(default=0, ge=0, description="Prompt tokens consumed.")
    completion_tokens: int = Field(default=0, ge=0, description="Completion tokens produced.")
    cached: bool = Field(default=False, description="True if served from the response cache.")
    steps: list[dict] = Field(
        default_factory=list,
        description="Intermediate agent tool calls (tool, tool_input, observation).",
    )


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------
# Middleware: attach a trace id + log the request
# ---------------------------------------------------------------------
@app.middleware("http")
async def trace_and_log(request: Request, call_next):
    trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex[:12]
    request.state.trace_id = trace_id
    bound = logger.bind(trace_id=trace_id, path=request.url.path, method=request.method)
    bound.info("request received")
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        bound.exception("unhandled error")
        raise
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    bound.bind(status=response.status_code, latency_ms=elapsed_ms).info("request completed")
    response.headers["x-trace-id"] = trace_id
    response.headers["x-latency-ms"] = str(elapsed_ms)
    return response


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(status="ok", version=app.version)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Cheap liveness probe."""
    return HealthResponse(status="healthy", version=app.version)


_QUERY_RATE_LIMIT = settings.rate_limit or "1000000/minute"


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(_QUERY_RATE_LIMIT)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    """Forward the question to the LangChain SQL agent."""
    trace_id = getattr(request.state, "trace_id", uuid.uuid4().hex[:12])
    bound = log.bind(trace_id=trace_id, question=req.question, session_id=req.session_id)
    bound.info("agent invocation start")

    start = time.perf_counter()
    try:
        result = ask(req.question, session_id=req.session_id)
    except Exception as exc:
        bound.exception("agent invocation failed")
        msg = str(exc).lower()
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        if "rate_limit" in msg or "429" in msg:
            return QueryResponse(
                question=req.question,
                answer=(
                    "⚠️ The LLM's free-tier daily token limit was reached. "
                    "This resets hourly. Please wait a few minutes and try again, "
                    "or switch to a smaller model in .env (e.g. "
                    "GROQ_MODEL=llama-3.1-8b-instant)."
                ),
                trace_id=trace_id,
                latency_ms=elapsed_ms,
                timestamp=datetime.now(UTC),
            )
        if "connection" in msg or "timed out" in msg:
            return QueryResponse(
                question=req.question,
                answer=(
                    "⚠️ Could not reach the LLM provider. "
                    "Check your internet connection and try again."
                ),
                trace_id=trace_id,
                latency_ms=elapsed_ms,
                timestamp=datetime.now(UTC),
            )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    bound.bind(latency_ms=elapsed_ms, answer_len=len(result["answer"])).info("agent invocation ok")
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        trace_id=trace_id,
        latency_ms=elapsed_ms,
        timestamp=datetime.now(UTC),
        prompt_tokens=result.get("prompt_tokens", 0),
        completion_tokens=result.get("completion_tokens", 0),
        cached=result.get("cached", False),
        steps=result.get("steps", []),
    )


@app.get("/metrics")
def metrics() -> dict:
    """Counters for queries, tokens, cost estimate, cache hit rate."""
    return METRICS.snapshot()


@app.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str) -> None:
    """Forget prior-turn context for a given session id."""
    clear_session(session_id)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=settings.api_host, port=settings.api_port, reload=False)
