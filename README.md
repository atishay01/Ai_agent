# Olist E-Commerce Data Chatbot

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

| Layer       | Choice                                         |
|-------------|------------------------------------------------|
| Language    | Python 3.11                                    |
| Warehouse   | PostgreSQL 16                                  |
| ETL         | Pandas, SQLAlchemy                             |
| LLM         | Groq (`openai/gpt-oss-20b`)                    |
| Agent       | LangChain `create_sql_agent` + custom tools    |
| Backend     | FastAPI                                        |
| Frontend    | Streamlit                                      |
| Web data    | Requests, BeautifulSoup                        |

## Data model

8 tables in 3NF:

- `customers`, `sellers`, `products`, `geolocation` — dimensions
- `orders`, `order_items`, `order_payments`, `order_reviews` — facts

Foreign keys and indexes on all join columns. Full DDL in
[`src/schema.sql`](src/schema.sql).

## Setup

1. **Install** Python 3.11, PostgreSQL 16, and Git.
2. **Clone**
   ```
   git clone https://github.com/atishay01/Ai_agent.git
   cd Ai_agent
   ```
3. **Create venv and install deps**
   ```
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Download** the 9 Olist CSVs from
   [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
   into `data/raw/`.
5. **Create the database**
   ```
   psql -U postgres -c "CREATE DATABASE olist_db;"
   ```
6. **Configure secrets** — copy `.env.example` to `.env` and fill in
   your Postgres password and a free Groq API key.
7. **Run the ETL pipeline**
   ```
   python src/etl.py
   ```
   Loads ~570K rows across 8 tables in about 90 seconds.
8. **Start the backend and UI** (two terminals)
   ```
   python src/api.py              # FastAPI on :8000
   streamlit run src/app.py       # Streamlit on :8501
   ```
9. Open http://localhost:8501. Use the **💬 Chat** tab to ask
   natural-language questions, or the **📊 Dashboard** tab for the
   BI-style KPI view. The Dashboard tab works even without FastAPI
   running — it queries Postgres directly.

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

## Limitations

- Only tested manually; no automated test suite.
- The free Groq tier has a daily token cap; heavy use falls back to a
  friendly error message from FastAPI.
- Wikipedia scraping is structure-dependent; if the target page layout
  changes, the scraper may need updating.
- Runs locally only. To deploy, Postgres, FastAPI and Streamlit would
  each need their own host (e.g. Supabase + Render + Streamlit Cloud).

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
│   ├── agent.py           # LangChain SQL agent
│   ├── api.py             # FastAPI backend
│   ├── app.py             # Streamlit UI — Chat + Dashboard tabs
│   └── dashboard.py       # Dashboard queries + rendering (direct Postgres, no LLM)
├── .env.example
├── .gitignore
└── requirements.txt
```
