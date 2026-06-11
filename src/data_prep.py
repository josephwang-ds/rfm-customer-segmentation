"""Load and clean the Online Retail II dataset (2009-12-01 to 2011-12-09).

Reads data/raw/online_retail_ii_combined.parquet (built from the UCI
"Online Retail II" workbook), applies the standard cleaning rules, and
writes a clean transaction-level table to data/processed/transactions_clean.parquet.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_ii_combined.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_PATH}")

    df = pd.read_parquet(RAW_PATH)
    df = df.rename(columns={"Customer ID": "CustomerID"})

    n_raw = len(df)

    clean = df[
        df["CustomerID"].notna()
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & (~df["Invoice"].astype(str).str.startswith("C"))
    ].copy()

    clean["CustomerID"] = clean["CustomerID"].astype(int)
    clean["Amount"] = clean["Quantity"] * clean["Price"]
    clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"])

    n_clean = len(clean)
    n_customers = clean["CustomerID"].nunique()
    n_invoices = clean["Invoice"].nunique()
    revenue = clean["Amount"].sum()
    date_min, date_max = clean["InvoiceDate"].min(), clean["InvoiceDate"].max()

    print(f"Raw rows: {n_raw:,}")
    print(f"Clean rows: {n_clean:,} ({n_clean/n_raw:.1%} of raw)")
    print(f"Customers: {n_customers:,}")
    print(f"Invoices: {n_invoices:,}")
    print(f"Revenue: GBP {revenue:,.0f}")
    print(f"Date range: {date_min.date()} to {date_max.date()}")
    print(f"Countries: {clean['Country'].nunique()}")
    print(clean["Country"].value_counts().head(8))

    clean.to_parquet(OUTPUT_DIR / "transactions_clean.parquet")

    summary = pd.DataFrame(
        [
            {
                "metric": "raw_rows",
                "value": n_raw,
            },
            {"metric": "clean_rows", "value": n_clean},
            {"metric": "customers", "value": n_customers},
            {"metric": "invoices", "value": n_invoices},
            {"metric": "revenue_gbp", "value": round(revenue, 2)},
            {"metric": "date_min", "value": str(date_min.date())},
            {"metric": "date_max", "value": str(date_max.date())},
            {"metric": "countries", "value": clean["Country"].nunique()},
        ]
    )
    summary.to_csv(OUTPUT_DIR / "data_summary.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
