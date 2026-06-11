"""DEPRECATED: superseded by rfm_segments.py + clv_model.py + churn_model.py.

Kept for reference only. This was the original 1-year Online Retail / SQLite
pipeline; the current project uses the 2-year Online Retail II dataset with
a calibration/holdout split (see src/rfm_segments.py).
"""

"""Build RFM customer segments from the local Online Retail SQLite database."""

from __future__ import annotations

import sqlite3
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DB = Path(os.environ.get("RFM_DB_PATH", PROJECT_ROOT / "data" / "raw" / "rfm.db"))
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
    warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_DB.exists():
        raise FileNotFoundError(
            f"Source database not found: {SOURCE_DB}. "
            "Place rfm.db under data/raw/ or set RFM_DB_PATH=/absolute/path/to/rfm.db."
        )

    with sqlite3.connect(SOURCE_DB) as con:
        df = pd.read_sql_query(
            """
            SELECT
              InvoiceNo,
              Quantity,
              InvoiceDate,
              UnitPrice,
              CustomerID,
              Country,
              Quantity * UnitPrice AS Amount
            FROM sales_data
            """,
            con,
            parse_dates=["InvoiceDate"],
        )

    clean = df[
        df["CustomerID"].notna()
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (~df["InvoiceNo"].astype(str).str.startswith("C"))
    ].copy()

    reference_date = clean["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        clean.groupby("CustomerID")
        .agg(
            recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            frequency=("InvoiceNo", "nunique"),
            monetary=("Amount", "sum"),
            first_order=("InvoiceDate", "min"),
            last_order=("InvoiceDate", "max"),
            country=("Country", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        )
        .reset_index()
    )

    features = pd.DataFrame(
        {
            "recency": np.log1p(rfm["recency"]),
            "frequency": np.log1p(rfm["frequency"]),
            "monetary": np.log1p(rfm["monetary"]),
        }
    )
    scaled = StandardScaler().fit_transform(features)

    diagnostics = []
    for k in range(2, 8):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled)
        diagnostics.append(
            {
                "k": k,
                "silhouette": silhouette_score(scaled, labels),
                "inertia": model.inertia_,
            }
        )

    model = KMeans(n_clusters=4, random_state=42, n_init=50)
    rfm["cluster"] = model.fit_predict(scaled)

    summary = (
        rfm.groupby("cluster")
        .agg(
            customers=("CustomerID", "count"),
            recency_median_days=("recency", "median"),
            frequency_median_orders=("frequency", "median"),
            monetary_median_gbp=("monetary", "median"),
            revenue=("monetary", "sum"),
        )
        .reset_index()
    )
    summary["customer_pct"] = summary["customers"] / len(rfm) * 100
    summary["revenue_pct"] = summary["revenue"] / rfm["monetary"].sum() * 100

    rfm.to_csv(OUTPUT_DIR / "rfm_customers.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUTPUT_DIR / "kmeans_diagnostics.csv", index=False)
    print(f"Wrote {len(rfm):,} customers to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
