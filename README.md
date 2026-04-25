# Olist E-Commerce Data Chatbot

[![CI](https://github.com/atishay01/Ai_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/atishay01/Ai_agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-SQL%20Agent-1C3C3C?logo=langchain)](https://python.langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered data-query app over the public
[Brazilian E-Commerce (Olist)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
dataset. **Two interfaces** on the same Postgres warehouse:

- **💬 Chat** — users ask business questions in plain English; a LangChain
  SQL agent writes the SQL, queries Postgres, optionally calls web APIs
  for currency / geographic info, and returns a readable answer.
- **📊 Dashboard** — a BI-style KPI view that queries Postgres directly
  (no LLM) for fast, deterministic loads. Total revenue, monthly trend,
  top categories / states, payment mix, review-score distribution.

The split is deliberate: the chatbot is for non-analysts who want ad-hoc
answers; the dashboard is for analysts who want shape-at-a-glance.

## Architecture

```
9 CSVs ─► ETL (Pandas) ─► PostgreSQL (8 tables, ~570K rows)
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
       LangChain SQL Agent (Groq)     Dashboard SQL (cached)
       ├── SQL Toolkit                ├── KPIs (revenue, AOV, on-time)
       ├── get_usd_brl_rate           ├── Monthly trend
       ├── lookup_brazilian_state     ├── Category / state breakdowns
       └── calculate                  └── Payment mix, reviews
                 │                             │
                 ▼                             ▼
              FastAPI  ────────►  Streamlit UI (tabbed)
               :8000                    :8501
                                  💬 Chat  |  📊 Dashboard
```

## Tech stack

| Layer         | Choice                                       |
|---------------|----------------------------------------------|
| Language      | Python 3.11                                  |
| Warehouse     | PostgreSQL 16                                |
| ETL           | Pandas, SQLAlchemy                           |
| LLM           | Groq (`openai/gpt-oss-20b`)                  |
| Agent         | LangChain `create_sql_agent` + custom tools  |
| Backend       | FastAPI + slowapi rate-limiting              |
| Frontend      | Streamlit (Chat + Dashboard tabs)            |
| Web data      | Requests, BeautifulSoup                      |
| Tests         | pytest (90+ tests, mocked externals)         |
| Lint / format | ruff + black, enforced in CI                 |
| Packaging     | Multi-stage Dockerfile + docker-compose      |

## Data model

8 tables in 3NF:

- `customers`, `sellers`, `products`, `geolocation` — dimensions
- `orders`, `order_items`, `order_payments`, `order_reviews` — facts

Foreign keys and indexes on all join columns. Full DDL in
[`src/schema.sql`](src/schema.sql).

## Production features

| Feature                  | Where it lives                                            |
|--------------------------|-----------------------------------------------------------|
| **Per-session memory**   | `src/agent.py` keeps the last 6 turns per `session_id`.   |
| **LRU response cache**   | `src/cache.py` — 128-entry bounded, normalized-question key. |
| **Token + cost tracking**| `src/callbacks.py` reads the three token-usage shapes ChatGroq emits. |
| **Prometheus-ish metrics** | `GET /metrics` → queries, failures, tokens, cache hits, $ estimate. |
| **Session reset API**    | `DELETE /session/{session_id}` drops server-side history. |
| **Rate limiting**        | slowapi on `/query`, 30 req/min default (configurable).   |
| **Structured logging**   | Loguru, JSON-friendly format, `logs/app.log` ships outside the image. |
| **Health check**         | `/health` endpoint used by the Docker `HEALTHCHECK`.      |
| **Trace panel**          | Streamlit renders `intermediate_steps` with SQL syntax highlighting. |
| **Golden eval suite**    | `eval/run_eval.py` runs 9 cases end-to-end, prints pass/fail + latency + $. |
| **CI**                   | GitHub Actions: ruff + black + pytest on every push.      |

Latest end-to-end eval run (dockerized stack):

```
Passed: 9/9 (100.0%)    Avg latency: 4321 ms    Total tokens: 35,473 (~$0.0027)
```

## Quick start — Docker (recommended)

One command brings up Postgres + FastAPI + Streamlit:

```
cp .env.example .env         # then edit GROQ_API_KEY + PG_PASSWORD
docker compose up --build
```

Seed the database (one-off, after `olist_db` is healthy):

```
docker compose run --rm api python src/etl.py
```

Open http://localhost:8501 for the UI and http://localhost:8000/health
for the API. Tear down with `docker compose down -v`.

## Manual setup (without Docker)

1. Install Python 3.11, PostgreSQL 16, and Git.
2. `git clone https://github.com/atishay01/Ai_agent.git && cd Ai_agent`
3. `py -3.11 -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
4. Download the 9 Olist CSVs from
   [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
   into `data/raw/`.
5. `psql -U postgres -c "CREATE DATABASE olist_db;"`
6. Copy `.env.example` to `.env` and fill in `PG_PASSWORD` + `GROQ_API_KEY`.
7. `python src/etl.py` — loads ~570K rows across 8 tables in ~90 s.
8. In two terminals:
   ```
   python src/api.py              # FastAPI on :8000
   streamlit run src/app.py       # Streamlit on :8501
   ```
9. Open http://localhost:8501. The **📊 Dashboard** tab works even
   without FastAPI — it queries Postgres directly.

## Sample questions

- How many orders are in the database?
- Top 5 product categories by revenue?
- Which state has the most late deliveries?
- What is the total revenue in US dollars?
- What is the capital of state RJ?
- Top 5 states by revenue — show full names and capitals.

## Design notes

- **Prompt design**: the agent gets a concise system prompt listing the
  schema and a few column-usage rules (e.g. `category_name_en` for
  category filters, `is_late = TRUE` for late deliveries). This trims
  the LLM's search space and produces much more reliable SQL than
  a schema-only prompt.

- **LLM math drift**: multiplying a 7-digit BRL total by an exchange
  rate gives LLMs trouble — they often drift by 0.1–1%. The
  `calculate` tool evaluates arithmetic through Python's `ast` module,
  so currency conversions are exact.

- **Web data**: the agent can fetch the live BRL/USD rate from
  Frankfurter and scrape the "States of Brazil" Wikipedia page for
  full state names and capitals — demonstrating that answers can
  come from outside the database when needed.

- **Cache key normalization**: the LRU cache keys questions by their
  lower-cased, whitespace-collapsed form so "How many orders?" and
  "how   many orders?" hit the same slot. Cache hits skip the LLM
  entirely and return in <5 ms, which the /metrics endpoint surfaces.

- **Token accounting**: ChatGroq emits usage metadata in three
  different shapes across model versions (`LLMResult.llm_output`,
  `AIMessage.usage_metadata`, `AIMessage.response_metadata`). The
  callback handler reads all three with a priority order so the
  counters are never silently zero.

## Testing

```
pytest -q                        # full suite (~90 tests, fully mocked)
pytest tests/test_agent.py -v    # one module
```

External services (Postgres, Groq, Wikipedia, Frankfurter) are mocked
with `unittest.mock` — the suite runs offline and CI doesn't need
any real credentials. Ruff + Black run in the same CI job.

## Evaluation

A small golden set of questions with expected-substring assertions
(and per-case latency budgets):

```
# inside the container:
docker compose exec api python eval/run_eval.py

# subset:
python eval/run_eval.py --only orders_count,top_revenue_category

# machine-readable:
python eval/run_eval.py --json report.json
```

Exits non-zero on any failure, so it slots into CI once Postgres +
Groq are reachable from the runner.

## Limitations

- The free Groq tier has a daily token cap; heavy use falls back to a
  friendly error message from FastAPI.
- Wikipedia scraping is structure-dependent; the scraper falls back to
  a hardcoded IBGE state table if the page layout changes or the
  network is unreachable, so state-name and capital lookups keep
  working — but new states/territories would need a code update.
- Frankfurter (BRL/USD rate) is cached for one hour and falls back to
  the last-known-good rate if the API is down, so currency answers are
  resilient to transient outages but may go stale during long ones.
- In-process state: per-session memory and the LRU response cache live
  in one Python process. Multi-worker deployments fragment this state
  and a restart loses it — a Redis-backed store would be the next
  step. Memory window is also bounded (default 6 turns, configurable
  via `SESSION_HISTORY_TURNS`).
- Static dataset: the Olist data is a 2016–2018 snapshot. Questions
  like "this month's revenue" can't be answered against live data.
- Runs locally only. For production, Postgres, FastAPI, and Streamlit
  would each need their own host (e.g. Supabase + Render + Streamlit
  Cloud).

## Project layout

```
.
├── data/raw/              # 9 Olist CSVs (gitignored)
├── queries/               # sample SQL for pgAdmin demos
├── src/
│   ├── schema.sql         # DDL
│   ├── db.py              # Postgres connection helper
│   ├── etl.py             # Pandas ETL pipeline
│   ├── web_tools.py       # currency, Wikipedia, calculator
│   ├── agent.py           # LangChain SQL agent + per-session memory
│   ├── callbacks.py       # TokenUsageCallback — reads 3 LLM usage shapes
│   ├── cache.py           # Bounded LRU response cache
│   ├── metrics.py         # Thread-safe counters for /metrics
│   ├── api.py             # FastAPI backend (/query, /metrics, /session, /health)
│   ├── app.py             # Streamlit UI — Chat + Dashboard tabs
│   └── dashboard.py       # Dashboard queries + rendering (direct Postgres)
├── tests/                 # 90+ pytest cases; external services mocked
├── eval/
│   ├── golden.yaml        # 9-case golden set
│   └── run_eval.py        # CLI harness, prints report + exits non-zero on fail
├── .github/workflows/ci.yml   # ruff + black + pytest on every push
├── Dockerfile             # Multi-stage (builder + slim runtime, non-root user)
├── docker-compose.yml     # db + api + ui with healthchecks
├── .dockerignore
├── .env.example
├── .gitignore
└── requirements.txt
```
