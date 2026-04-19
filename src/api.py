"""
FastAPI backend
===============

Thin HTTP layer over the LangChain SQL agent. Streamlit (or any client)
POSTs a natural-language question to /query and gets JSON back.

Run:  uvicorn src.api:app --reload   (from project root)
Or:   python src/api.py              (direct, dev mode)
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import ask

# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("olist-api")

# ---------------------------------------------------------------------
app = FastAPI(
    title="Olist AI Data-Query Chatbot",
    description="Ask business questions about the Olist e-commerce database.",
    version="1.0.0",
)

# Streamlit runs on a different port; allow it to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # demo only — tighten in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          examples=["What are the top 5 product categories by revenue?"])


class QueryResponse(BaseModel):
    question: str
    answer:   str


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "olist-chatbot", "status": "ok"}


@app.get("/health")
def health():
    """Cheap liveness probe."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Forward the question to the LangChain SQL agent."""
    log.info("Q: %s", req.question)
    try:
        result = ask(req.question)
    except Exception as exc:
        log.exception("agent failed")
        msg = str(exc).lower()
        # Friendly message for the common failure modes
        if "rate_limit" in msg or "429" in msg:
            return QueryResponse(
                question=req.question,
                answer=("⚠️ The LLM's free-tier daily token limit was reached. "
                        "This resets hourly. Please wait a few minutes and try again, "
                        "or switch to a smaller model in .env (e.g. "
                        "GROQ_MODEL=llama-3.1-8b-instant)."))
        if "connection" in msg or "timed out" in msg:
            return QueryResponse(
                question=req.question,
                answer="⚠️ Could not reach the LLM provider. Check your internet connection and try again.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    log.info("A: %s", result["answer"][:120])
    return QueryResponse(**result)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
