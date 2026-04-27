-- =====================================================================
-- Olist semantic layer — pre-aggregated business views
-- =====================================================================
--
-- These views are the project's single source of truth for the metrics
-- both the agent and the dashboard care about: revenue, on-time
-- delivery rate, category performance, state performance.
--
-- Why views, not ad-hoc SUMs:
--   * The agent stops re-deriving 'SUM(price + freight_value)' every
--     time someone asks about revenue. One canonical formula, one
--     consistent answer.
--   * The dashboard becomes 'SELECT * FROM mart_*' instead of six
--     ad-hoc joins. Faster cold-render, cleaner code.
--   * If the formula ever changes (e.g. exclude cancelled orders), it
--     changes in exactly one place.
--
-- Created automatically at the end of ETL (etl.py:apply_marts) so a
-- fresh deploy always has them.
-- =====================================================================

DROP VIEW IF EXISTS mart_state_performance;
DROP VIEW IF EXISTS mart_revenue_by_category;
DROP VIEW IF EXISTS mart_revenue_by_month;


-- ---------------------------------------------------------------------
-- mart_revenue_by_month
--
-- One row per (year, month). Revenue, order count, and unique-customer
-- count for that month. Use for time-series charts and growth trend
-- questions.
-- ---------------------------------------------------------------------
CREATE VIEW mart_revenue_by_month AS
SELECT
    DATE_TRUNC('month', o.order_purchase_timestamp)::date  AS month,
    ROUND(SUM(oi.price + oi.freight_value)::numeric, 2)    AS revenue_brl,
    COUNT(DISTINCT o.order_id)                             AS orders,
    COUNT(DISTINCT c.customer_unique_id)                   AS unique_customers
FROM order_items oi
JOIN orders     o USING (order_id)
JOIN customers  c USING (customer_id)
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY 1
ORDER BY 1;


-- ---------------------------------------------------------------------
-- mart_revenue_by_category
--
-- One row per English category name. Revenue, line-item count, average
-- review score for products in that category. NULL categories are
-- bucketed as 'uncategorised' so the view always has a complete set.
-- ---------------------------------------------------------------------
CREATE VIEW mart_revenue_by_category AS
SELECT
    COALESCE(p.category_name_en, 'uncategorised')         AS category,
    ROUND(SUM(oi.price + oi.freight_value)::numeric, 2)   AS revenue_brl,
    COUNT(*)                                              AS items_sold,
    ROUND(AVG(r.review_score)::numeric, 2)                AS avg_review_score
FROM order_items oi
JOIN products      p USING (product_id)
LEFT JOIN order_reviews r USING (order_id)
GROUP BY COALESCE(p.category_name_en, 'uncategorised');


-- ---------------------------------------------------------------------
-- mart_state_performance
--
-- One row per Brazilian state (2-letter code). Order count, revenue,
-- on-time-delivery percentage, average delivery time. Use for
-- geo-comparison questions and the dashboard's state-ranking chart.
-- ---------------------------------------------------------------------
CREATE VIEW mart_state_performance AS
SELECT
    c.state                                                  AS state,
    COUNT(DISTINCT o.order_id)                               AS orders,
    ROUND(SUM(oi.price + oi.freight_value)::numeric, 2)      AS revenue_brl,
    ROUND(
        100.0 * SUM(CASE WHEN o.is_late = FALSE THEN 1 ELSE 0 END)
              / NULLIF(COUNT(*) FILTER (WHERE o.is_late IS NOT NULL), 0)
        , 2)                                                 AS on_time_pct,
    ROUND(AVG(o.delivery_days)
        FILTER (WHERE o.delivery_days IS NOT NULL)::numeric
        , 1)                                                 AS avg_delivery_days
FROM orders     o
JOIN customers  c  USING (customer_id)
JOIN order_items oi USING (order_id)
GROUP BY c.state
ORDER BY revenue_brl DESC;


-- ---------------------------------------------------------------------
-- The read-only agent role (created in seed_readonly_user.sql) inherits
-- SELECT on these views via ALTER DEFAULT PRIVILEGES. Belt-and-braces
-- explicit grant in case the role pre-existed.
-- ---------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'olist_agent') THEN
        GRANT SELECT ON
            mart_revenue_by_month,
            mart_revenue_by_category,
            mart_state_performance
        TO olist_agent;
    END IF;
END
$$;
