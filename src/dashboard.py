"""Business-KPI dashboard for the Olist e-commerce dataset.

Rendered as a Streamlit tab alongside the chatbot. Queries Postgres directly
(not through the LLM agent) so the charts load fast and deterministically.

Dataset is historical (2016-09 to 2018-10), so KPIs are framed as
"latest month in data" + "all-time" rather than real current-month figures.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import text

from db import get_agent_engine


# ---------------------------------------------------------------------
# Query helpers — cached so the dashboard loads fast on re-renders.
# ---------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _query(sql: str) -> pd.DataFrame:
    # Read-only role — dashboard never writes.
    with get_agent_engine().connect() as conn:
        return pd.read_sql(text(sql), conn)


def _kpis() -> dict:
    row = _query(
        """
        SELECT
            (SELECT ROUND(SUM(price + freight_value)::numeric, 2)
                FROM order_items)                                    AS total_revenue,
            (SELECT COUNT(*) FROM orders)                            AS total_orders,
            (SELECT COUNT(DISTINCT customer_unique_id) FROM customers) AS total_customers,
            (SELECT ROUND(AVG(delivery_days)::numeric, 1)
                FROM orders WHERE delivery_days IS NOT NULL)         AS avg_delivery_days,
            (SELECT MAX(order_purchase_timestamp) FROM orders)       AS latest_ts
        """
    ).iloc[0]
    return row.to_dict()


def _latest_month_revenue() -> tuple[str, float, float]:
    """Revenue for the last full month in data + month-over-month delta (%)."""
    df = _query(
        """
        SELECT
            DATE_TRUNC('month', o.order_purchase_timestamp)::date AS month,
            SUM(oi.price + oi.freight_value)::numeric             AS revenue
        FROM order_items oi
        JOIN orders o USING (order_id)
        WHERE o.order_purchase_timestamp IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    )
    if len(df) < 2:
        return ("n/a", 0.0, 0.0)
    # Drop the partial last month (trailing edge of dataset) so MoM is fair.
    # Olist data ends 2018-10-17 — that month is incomplete.
    df = df.iloc[:-1]
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    delta_pct = ((latest.revenue - prev.revenue) / prev.revenue * 100.0) if prev.revenue else 0.0
    return (str(latest.month), float(latest.revenue), float(delta_pct))


def _monthly_revenue() -> pd.DataFrame:
    return _query(
        """
        SELECT
            DATE_TRUNC('month', o.order_purchase_timestamp)::date AS month,
            ROUND(SUM(oi.price + oi.freight_value)::numeric, 2)   AS revenue_brl,
            COUNT(DISTINCT o.order_id)                            AS orders
        FROM order_items oi
        JOIN orders o USING (order_id)
        WHERE o.order_purchase_timestamp IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    )


def _revenue_by_category(limit: int = 10) -> pd.DataFrame:
    return _query(
        f"""
        SELECT
            COALESCE(p.category_name_en, 'uncategorised') AS category,
            ROUND(SUM(oi.price + oi.freight_value)::numeric, 2) AS revenue_brl
        FROM order_items oi
        JOIN products p USING (product_id)
        GROUP BY 1
        ORDER BY revenue_brl DESC
        LIMIT {limit}
        """
    )


def _revenue_by_state(limit: int = 10) -> pd.DataFrame:
    return _query(
        f"""
        SELECT
            c.state,
            ROUND(SUM(oi.price + oi.freight_value)::numeric, 2) AS revenue_brl,
            COUNT(DISTINCT o.order_id)                          AS orders
        FROM order_items oi
        JOIN orders    o USING (order_id)
        JOIN customers c USING (customer_id)
        GROUP BY 1
        ORDER BY revenue_brl DESC
        LIMIT {limit}
        """
    )


def _payment_mix() -> pd.DataFrame:
    return _query(
        """
        SELECT
            payment_type,
            COUNT(*)                                            AS payments,
            ROUND(SUM(payment_value)::numeric, 2)               AS total_value_brl
        FROM order_payments
        WHERE payment_type IS NOT NULL
        GROUP BY 1
        ORDER BY total_value_brl DESC
        """
    )


def _review_score_distribution() -> pd.DataFrame:
    return _query(
        """
        SELECT review_score, COUNT(*) AS reviews
        FROM order_reviews
        WHERE review_score IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    )


def _on_time_rate() -> float:
    row = _query(
        """
        SELECT
            100.0 * SUM(CASE WHEN is_late = FALSE THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(*), 0) AS on_time_pct
        FROM orders
        WHERE is_late IS NOT NULL
        """
    ).iloc[0]
    return float(row.on_time_pct or 0.0)


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------
def _fmt_brl(value: float) -> str:
    if value >= 1_000_000:
        return f"R$ {value/1_000_000:.2f}M"
    if value >= 1_000:
        return f"R$ {value/1_000:.1f}K"
    return f"R$ {value:,.2f}"


def render() -> None:
    """Render the Dashboard tab. Called from app.py inside a `with tab:` block."""
    st.subheader("Business KPIs — Olist marketplace")
    st.caption(
        "Dataset is historical (2016-09 → 2018-10). 'Latest month' means the "
        "most recent full month in the data, not the calendar current month."
    )

    # --- KPI row -----------------------------------------------------
    try:
        k = _kpis()
        latest_month, latest_rev, mom_delta = _latest_month_revenue()
        on_time = _on_time_rate()
    except Exception as e:  # pragma: no cover — surfaced to user
        st.error(f"Could not load KPIs from Postgres: {e}")
        st.info(
            "Make sure Postgres is running and `.env` has valid credentials. "
            "Run `python src/etl.py` if the schema is empty."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total revenue (all time)", _fmt_brl(float(k["total_revenue"] or 0)))
    c2.metric("Orders", f"{int(k['total_orders']):,}")
    c3.metric("Unique customers", f"{int(k['total_customers']):,}")
    c4.metric("Avg delivery", f"{float(k['avg_delivery_days'] or 0):.1f} days")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Latest full month revenue",
        _fmt_brl(latest_rev),
        delta=f"{mom_delta:+.1f}% vs prior month",
        help=f"Latest full month in data: {latest_month}",
    )
    c6.metric("On-time delivery rate", f"{on_time:.1f}%")
    aov = (float(k["total_revenue"] or 0) / int(k["total_orders"])) if k["total_orders"] else 0.0
    c7.metric("Avg order value", _fmt_brl(aov))
    c8.metric(
        "Data snapshot",
        str(k["latest_ts"])[:10] if k["latest_ts"] else "n/a",
        help="Most recent order_purchase_timestamp in the database.",
    )

    st.divider()

    # --- Revenue trend ----------------------------------------------
    st.markdown("#### Monthly revenue trend")
    monthly = _monthly_revenue()
    if not monthly.empty:
        chart_df = monthly.set_index("month")[["revenue_brl"]]
        chart_df.columns = ["Revenue (BRL)"]
        st.line_chart(chart_df, height=280)
        st.caption(
            f"Peak month: **{monthly.loc[monthly.revenue_brl.idxmax(), 'month']}** "
            f"at **{_fmt_brl(float(monthly.revenue_brl.max()))}**. "
            "Note the ramp through 2017 followed by flattening — classic marketplace "
            "growth curve."
        )

    # --- Category × State side by side ------------------------------
    col_cat, col_state = st.columns(2)

    with col_cat:
        st.markdown("#### Top 10 categories by revenue")
        cat = _revenue_by_category(10)
        if not cat.empty:
            cat_display = cat.rename(
                columns={"category": "Category", "revenue_brl": "Revenue (BRL)"}
            ).set_index("Category")
            st.bar_chart(cat_display, height=340, horizontal=True)

    with col_state:
        st.markdown("#### Top 10 states by revenue")
        state = _revenue_by_state(10)
        if not state.empty:
            state_display = state.rename(
                columns={"state": "State", "revenue_brl": "Revenue (BRL)"}
            ).set_index("State")[["Revenue (BRL)"]]
            st.bar_chart(state_display, height=340, horizontal=True)
            if "SP" in state["state"].values:
                sp_share = (
                    state.loc[state["state"] == "SP", "revenue_brl"].iloc[0]
                    / float(k["total_revenue"] or 1)
                    * 100
                )
                st.caption(
                    f"São Paulo (SP) alone accounts for **{sp_share:.1f}%** of total "
                    "revenue — marketplace is heavily São Paulo-centric."
                )

    st.divider()

    # --- Payment mix + Reviews --------------------------------------
    col_pay, col_rev = st.columns(2)

    with col_pay:
        st.markdown("#### Payment method split")
        pay = _payment_mix()
        if not pay.empty:
            pay_display = pay.rename(
                columns={
                    "payment_type": "Payment type",
                    "payments": "Payments",
                    "total_value_brl": "Total value (BRL)",
                }
            )
            st.dataframe(pay_display, hide_index=True, use_container_width=True)
            top = pay.iloc[0]
            pay_share = float(top.total_value_brl) / pay.total_value_brl.sum() * 100
            st.caption(
                f"**{top.payment_type}** dominates at **{pay_share:.1f}%** of value. "
                "Credit card + installment behaviour is the single biggest lever for "
                "payment-ops optimisation."
            )

    with col_rev:
        st.markdown("#### Review score distribution")
        rv = _review_score_distribution()
        if not rv.empty:
            rv_display = rv.rename(
                columns={"review_score": "Score", "reviews": "Reviews"}
            ).set_index("Score")
            st.bar_chart(rv_display, height=340)
            total = rv.reviews.sum()
            five_star = (
                rv.loc[rv.review_score == 5, "reviews"].sum() if 5 in rv.review_score.values else 0
            )
            one_star = (
                rv.loc[rv.review_score == 1, "reviews"].sum() if 1 in rv.review_score.values else 0
            )
            st.caption(
                f"**{five_star / total * 100:.1f}%** of reviews are 5-star, "
                f"but **{one_star / total * 100:.1f}%** are 1-star — a bimodal "
                "distribution typical of marketplace reviews."
            )

    st.divider()
    st.caption(
        "All figures in **BRL** (Brazilian Reais). Queries hit Postgres directly "
        "(not the LLM agent) and are cached for 1 hour via `@st.cache_data`. "
        "Source SQL lives in `src/dashboard.py`."
    )
