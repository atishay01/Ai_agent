"""
Olist ETL Pipeline — full 8-table load
======================================

Stages: EXTRACT -> PROFILE -> CLEAN -> TRANSFORM -> LOAD -> VALIDATE

Run:  python src/etl.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine

# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
SCHEMA_FILE  = PROJECT_ROOT / "src"  / "schema.sql"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def profile(df: pd.DataFrame, name: str) -> None:
    log(f"  {name}: shape={df.shape}  "
        f"nulls={int(df.isna().sum().sum())}  "
        f"duplicates={int(df.duplicated().sum())}")


# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------
def apply_schema(engine) -> None:
    log("Applying schema...")
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(text(sql))
    log("  schema applied OK")


# ---------------------------------------------------------------------
# customers
# ---------------------------------------------------------------------
def build_customers() -> pd.DataFrame:
    log("EXTRACT customers")
    df = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")
    profile(df, "raw customers")

    df = df.rename(columns={"customer_zip_code_prefix": "zip_code_prefix",
                            "customer_city":            "city",
                            "customer_state":           "state"})
    df["city"]  = df["city"].astype(str).str.strip().str.title()
    df["state"] = df["state"].astype(str).str.upper().str[:2]
    df["zip_code_prefix"] = df["zip_code_prefix"].astype(str).str.zfill(5)
    df = df.drop_duplicates(subset=["customer_id"])
    profile(df, "clean customers")
    return df[["customer_id", "customer_unique_id",
               "zip_code_prefix", "city", "state"]]


# ---------------------------------------------------------------------
# sellers (new)
# ---------------------------------------------------------------------
def build_sellers() -> pd.DataFrame:
    log("EXTRACT sellers")
    df = pd.read_csv(RAW_DIR / "olist_sellers_dataset.csv")
    profile(df, "raw sellers")

    df = df.rename(columns={"seller_zip_code_prefix": "zip_code_prefix",
                            "seller_city":            "city",
                            "seller_state":           "state"})
    df["city"]  = df["city"].astype(str).str.strip().str.title()
    df["state"] = df["state"].astype(str).str.upper().str[:2]
    df["zip_code_prefix"] = df["zip_code_prefix"].astype(str).str.zfill(5)
    df = df.drop_duplicates(subset=["seller_id"])
    profile(df, "clean sellers")
    return df[["seller_id", "zip_code_prefix", "city", "state"]]


# ---------------------------------------------------------------------
# products (+ English translation merge)
# ---------------------------------------------------------------------
def build_products() -> pd.DataFrame:
    log("EXTRACT products + category translation")
    products = pd.read_csv(RAW_DIR / "olist_products_dataset.csv")
    trans    = pd.read_csv(RAW_DIR / "product_category_name_translation.csv")
    profile(products, "raw products")
    profile(trans,    "raw translation")

    df = products.merge(trans, on="product_category_name", how="left")

    dim_cols = ["product_weight_g", "product_length_cm",
                "product_height_cm", "product_width_cm"]
    for col in dim_cols:
        df[col] = df.groupby("product_category_name")[col].transform(
            lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].astype("Int64")

    df = df.rename(columns={"product_category_name":         "category_name_pt",
                            "product_category_name_english": "category_name_en"})
    df = df.drop_duplicates(subset=["product_id"])
    profile(df, "clean products")
    return df[["product_id", "category_name_pt", "category_name_en",
               "product_weight_g", "product_length_cm",
               "product_height_cm", "product_width_cm"]]


# ---------------------------------------------------------------------
# geolocation (new) — aggregate ~1M rows down to 1 row per zip
# ---------------------------------------------------------------------
def build_geolocation() -> pd.DataFrame:
    log("EXTRACT geolocation (large ~1M row CSV)")
    df = pd.read_csv(RAW_DIR / "olist_geolocation_dataset.csv")
    profile(df, "raw geolocation")

    df = df.rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix",
                            "geolocation_lat":             "lat",
                            "geolocation_lng":             "lng",
                            "geolocation_city":            "city",
                            "geolocation_state":           "state"})
    df["zip_code_prefix"] = df["zip_code_prefix"].astype(str).str.zfill(5)

    # Aggregate: mean lat/lng, modal city/state per zip_code_prefix
    def _mode_first(s: pd.Series):
        m = s.mode()
        return m.iloc[0] if not m.empty else None

    agg = (df.groupby("zip_code_prefix")
             .agg(lat=("lat", "mean"),
                  lng=("lng", "mean"),
                  city=("city",  _mode_first),
                  state=("state", _mode_first))
             .reset_index())

    agg["lat"]   = agg["lat"].round(6)
    agg["lng"]   = agg["lng"].round(6)
    agg["city"]  = agg["city"].astype(str).str.strip().str.title()
    agg["state"] = agg["state"].astype(str).str.upper().str[:2]
    profile(agg, "clean geolocation (one row per zip)")
    return agg


# ---------------------------------------------------------------------
# orders (with derived columns)
# ---------------------------------------------------------------------
def build_orders() -> pd.DataFrame:
    log("EXTRACT orders")
    df = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")
    profile(df, "raw orders")

    ts_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in ts_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["delivery_days"] = (df["order_delivered_customer_date"]
                           - df["order_purchase_timestamp"]).dt.days.astype("Int64")
    df["is_late"] = (df["order_delivered_customer_date"]
                     > df["order_estimated_delivery_date"])
    df["is_late"] = df["is_late"].fillna(False).astype(bool)

    df = df.drop_duplicates(subset=["order_id"])
    profile(df, "clean orders")
    return df[["order_id", "customer_id", "order_status",
               *ts_cols, "delivery_days", "is_late"]]


# ---------------------------------------------------------------------
# order_items
# ---------------------------------------------------------------------
def build_order_items() -> pd.DataFrame:
    log("EXTRACT order_items")
    df = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
    profile(df, "raw order_items")

    df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"],
                                               errors="coerce")
    df["price"]         = df["price"].astype(float).round(2)
    df["freight_value"] = df["freight_value"].astype(float).round(2)
    df = df.drop_duplicates(subset=["order_id", "order_item_id"])
    profile(df, "clean order_items")
    return df[["order_id", "order_item_id", "product_id", "seller_id",
               "shipping_limit_date", "price", "freight_value"]]


# ---------------------------------------------------------------------
# order_payments (new)
# ---------------------------------------------------------------------
def build_order_payments() -> pd.DataFrame:
    log("EXTRACT order_payments")
    df = pd.read_csv(RAW_DIR / "olist_order_payments_dataset.csv")
    profile(df, "raw order_payments")

    df["payment_value"] = df["payment_value"].astype(float).round(2)
    df["payment_installments"] = df["payment_installments"].astype("Int64")
    df = df.drop_duplicates(subset=["order_id", "payment_sequential"])
    profile(df, "clean order_payments")
    return df[["order_id", "payment_sequential", "payment_type",
               "payment_installments", "payment_value"]]


# ---------------------------------------------------------------------
# order_reviews (new)
# ---------------------------------------------------------------------
def build_order_reviews() -> pd.DataFrame:
    log("EXTRACT order_reviews")
    df = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv")
    profile(df, "raw order_reviews")

    df["review_creation_date"]    = pd.to_datetime(df["review_creation_date"],
                                                   errors="coerce")
    df["review_answer_timestamp"] = pd.to_datetime(df["review_answer_timestamp"],
                                                   errors="coerce")
    df["review_score"] = df["review_score"].astype("Int64")

    # Defensive dedup on composite PK
    df = df.drop_duplicates(subset=["review_id", "order_id"])
    profile(df, "clean order_reviews")
    return df[["review_id", "order_id", "review_score",
               "review_comment_title", "review_comment_message",
               "review_creation_date", "review_answer_timestamp"]]


# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------
def load(engine, name: str, df: pd.DataFrame) -> None:
    log(f"LOAD {name} ({len(df):,} rows)")
    df.to_sql(name, engine, if_exists="append", index=False,
              method="multi", chunksize=5000)


# ---------------------------------------------------------------------
# VALIDATE
# ---------------------------------------------------------------------
def validate(engine, expected: dict[str, int]) -> None:
    log("VALIDATE row counts")
    with engine.connect() as conn:
        for table, exp in expected.items():
            got = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            status = "OK" if got == exp else "MISMATCH"
            log(f"  {table:16s} expected={exp:>9,}  actual={got:>9,}  [{status}]")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    engine = get_engine()

    apply_schema(engine)

    # Build all dataframes
    customers      = build_customers()
    sellers        = build_sellers()
    products       = build_products()
    geolocation    = build_geolocation()
    orders         = build_orders()
    order_items    = build_order_items()
    order_payments = build_order_payments()
    order_reviews  = build_order_reviews()

    # Defensive FK filtering
    orders         = orders[orders["customer_id"].isin(customers["customer_id"])]
    order_items    = order_items[
        order_items["order_id"].isin(orders["order_id"])
        & order_items["product_id"].isin(products["product_id"])
        & order_items["seller_id"].isin(sellers["seller_id"])
    ]
    order_payments = order_payments[order_payments["order_id"].isin(orders["order_id"])]
    order_reviews  = order_reviews[order_reviews["order_id"].isin(orders["order_id"])]

    # Load in dependency order
    load(engine, "customers",      customers)
    load(engine, "sellers",        sellers)
    load(engine, "products",       products)
    load(engine, "geolocation",    geolocation)
    load(engine, "orders",         orders)
    load(engine, "order_items",    order_items)
    load(engine, "order_payments", order_payments)
    load(engine, "order_reviews",  order_reviews)

    validate(engine, {
        "customers":      len(customers),
        "sellers":        len(sellers),
        "products":       len(products),
        "geolocation":    len(geolocation),
        "orders":         len(orders),
        "order_items":    len(order_items),
        "order_payments": len(order_payments),
        "order_reviews":  len(order_reviews),
    })

    log(f"ETL complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
