-- =====================================================================
-- 02. Business questions — the SQL the LangChain agent generates.
-- Use these to show the interviewer that the agent produces real,
-- sensible SQL (or run them manually if the agent misbehaves).
-- =====================================================================

-- Q1: How many orders?
SELECT COUNT(*) AS total_orders FROM orders;


-- Q2: Top 5 product categories by revenue (JOIN across 3 tables)
SELECT
    p.category_name_en,
    ROUND(SUM(oi.price + oi.freight_value)::numeric, 2) AS revenue_brl
FROM order_items oi
JOIN products p ON p.product_id = oi.product_id
GROUP BY p.category_name_en
ORDER BY revenue_brl DESC
LIMIT 5;


-- Q3: State with the most late deliveries (uses derived is_late column)
SELECT
    c.state,
    COUNT(*) AS late_delivery_count
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.is_late = TRUE
GROUP BY c.state
ORDER BY late_delivery_count DESC
LIMIT 5;


-- Q4: Top 3 payment types by total value
SELECT
    payment_type,
    COUNT(*)  AS num_payments,
    ROUND(SUM(payment_value)::numeric, 2) AS total_value_brl
FROM order_payments
GROUP BY payment_type
ORDER BY total_value_brl DESC;


-- Q5: Sellers with the most 5-star reviews (4-way JOIN)
SELECT
    oi.seller_id,
    COUNT(*) AS five_star_count
FROM order_reviews r
JOIN orders      o  ON o.order_id = r.order_id
JOIN order_items oi ON oi.order_id = o.order_id
WHERE r.review_score = 5
GROUP BY oi.seller_id
ORDER BY five_star_count DESC
LIMIT 5;


-- Q6: Revenue by state (the query behind "Top states in USD")
SELECT
    c.state,
    ROUND(SUM(oi.price + oi.freight_value)::numeric, 2) AS revenue_brl
FROM order_items oi
JOIN orders    o ON o.order_id = oi.order_id
JOIN customers c ON c.customer_id = o.customer_id
GROUP BY c.state
ORDER BY revenue_brl DESC
LIMIT 5;
