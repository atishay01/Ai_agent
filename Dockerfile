# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the Olist AI chatbot. One image runs either the
# FastAPI backend (default) or the Streamlit UI — compose picks which by
# overriding the command.

# ---- Stage 1: build wheels in a throwaway layer -----------------------
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---- Stage 2: slim runtime -------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r app && useradd -r -g app app

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=app:app src/ ./src/
COPY --chown=app:app pyproject.toml ./

USER app

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Default: FastAPI backend. Compose overrides this for the UI service.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--app-dir", "/app/src"]
