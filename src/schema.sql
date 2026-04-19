-- =====================================================================
-- Olist E-Commerce Data Warehouse — Schema (PostgreSQL 16)
--
-- 8 tables total (4 dimension + 4 fact/associative):
--   customers      (dim)     products       (dim)
--   sellers        (dim)     geolocation    (dim, lookup by zip)
--   orders         (fact)    order_items    (fact, line items)
--   order_payments (fact)    order_reviews  (fact)
--
-- Design notes:
--   * 3NF relational model. Olist's natural string IDs kept as PKs
--     for traceability back to the raw CSVs.
--   * Foreign keys enforced; ETL loads tables in dependency order.
--   * Indexes on all join keys for fast agent-generated SQL.
-- =====================================================================

-- Drop in reverse-FK order so no constraint errors.
DROP TABLE IF EXISTS order_reviews   CASCADE;
DROP TABLE IF EXISTS order_payments  CASCADE;
DROP TABLE IF EXISTS order_items     CASCADE;
DROP TABLE IF EXISTS orders          CASCADE;
DROP TABLE IF EXISTS sellers         CASCADE;
DROP TABLE IF EXISTS products        CASCADE;
DROP TABLE IF EXISTS customers       CASCADE;
DROP TABLE IF EXISTS geolocation     CASCADE;

-- --------------------------------------------------------------------
-- customers — one row per customer order record
-- --------------------------------------------------------------------
CREATE TABLE customers (
    customer_id           VARCHAR(64) PRIMARY KEY,
    customer_unique_id    VARCHAR(64) NOT NULL,
    zip_code_prefix       VARCHAR(10),
    city                  VARCHAR(80),
    state                 CHAR(2)
);
CREATE INDEX idx_customers_unique_id ON customers (customer_unique_id);
CREATE INDEX idx_customers_state     ON customers (state);
CREATE INDEX idx_customers_zip       ON customers (zip_code_prefix);

-- --------------------------------------------------------------------
-- sellers — supply-side dimension
-- --------------------------------------------------------------------
CREATE TABLE sellers (
    seller_id         VARCHAR(64) PRIMARY KEY,
    zip_code_prefix   VARCHAR(10),
    city              VARCHAR(80),
    state             CHAR(2)
);
CREATE INDEX idx_sellers_state ON sellers (state);
CREATE INDEX idx_sellers_zip   ON sellers (zip_code_prefix);

-- --------------------------------------------------------------------
-- products — master catalog with English category names
-- --------------------------------------------------------------------
CREATE TABLE products (
    product_id            VARCHAR(64) PRIMARY KEY,
    category_name_pt      VARCHAR(80),
    category_name_en      VARCHAR(80),
    product_weight_g      INTEGER,
    product_length_cm     INTEGER,
    product_height_cm     INTEGER,
    product_width_cm      INTEGER
);
CREATE INDEX idx_products_category_en ON products (category_name_en);

-- --------------------------------------------------------------------
-- geolocation — one representative lat/lng per zip_code_prefix.
-- Raw CSV has ~1M rows with many duplicate zips; ETL aggregates
-- to one row per zip (mean lat/lng, modal city/state).
-- --------------------------------------------------------------------
CREATE TABLE geolocation (
    zip_code_prefix   VARCHAR(10) PRIMARY KEY,
    lat               NUMERIC(10, 6),
    lng               NUMERIC(10, 6),
    city              VARCHAR(80),
    state             CHAR(2)
);
CREATE INDEX idx_geo_state ON geolocation (state);

-- --------------------------------------------------------------------
-- orders — order headers with derived ETL columns
-- --------------------------------------------------------------------
CREATE TABLE orders (
    order_id                       VARCHAR(64) PRIMARY KEY,
    customer_id                    VARCHAR(64) NOT NULL
                                   REFERENCES customers (customer_id),
    order_status                   VARCHAR(20),
    order_purchase_timestamp       TIMESTAMP,
    order_approved_at              TIMESTAMP,
    order_delivered_carrier_date   TIMESTAMP,
    order_delivered_customer_date  TIMESTAMP,
    order_estimated_delivery_date  TIMESTAMP,
    delivery_days                  INTEGER,
    is_late                        BOOLEAN
);
CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_orders_status   ON orders (order_status);
CREATE INDEX idx_orders_purchase ON orders (order_purchase_timestamp);

-- --------------------------------------------------------------------
-- order_items — composite PK. Now FKs to sellers too.
-- --------------------------------------------------------------------
CREATE TABLE order_items (
    order_id               VARCHAR(64) NOT NULL
                           REFERENCES orders (order_id),
    order_item_id          INTEGER     NOT NULL,
    product_id             VARCHAR(64) NOT NULL
                           REFERENCES products (product_id),
    seller_id              VARCHAR(64)
                           REFERENCES sellers (seller_id),
    shipping_limit_date    TIMESTAMP,
    price                  NUMERIC(10, 2),
    freight_value          NUMERIC(10, 2),
    PRIMARY KEY (order_id, order_item_id)
);
CREATE INDEX idx_items_product ON order_items (product_id);
CREATE INDEX idx_items_seller  ON order_items (seller_id);

-- --------------------------------------------------------------------
-- order_payments — an order can have multiple payment rows
-- (e.g. split voucher + credit card). PK is (order_id, payment_sequential).
-- --------------------------------------------------------------------
CREATE TABLE order_payments (
    order_id              VARCHAR(64) NOT NULL
                          REFERENCES orders (order_id),
    payment_sequential    INTEGER     NOT NULL,
    payment_type          VARCHAR(20),
    payment_installments  INTEGER,
    payment_value         NUMERIC(10, 2),
    PRIMARY KEY (order_id, payment_sequential)
);
CREATE INDEX idx_payments_type ON order_payments (payment_type);

-- --------------------------------------------------------------------
-- order_reviews — 1..N with orders. Composite PK (review_id, order_id)
-- because a review_id can span multiple orders in raw data.
-- --------------------------------------------------------------------
CREATE TABLE order_reviews (
    review_id                VARCHAR(64) NOT NULL,
    order_id                 VARCHAR(64) NOT NULL
                             REFERENCES orders (order_id),
    review_score             INTEGER,
    review_comment_title     TEXT,
    review_comment_message   TEXT,
    review_creation_date     TIMESTAMP,
    review_answer_timestamp  TIMESTAMP,
    PRIMARY KEY (review_id, order_id)
);
CREATE INDEX idx_reviews_score ON order_reviews (review_score);
CREATE INDEX idx_reviews_order ON order_reviews (order_id);
