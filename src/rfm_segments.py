"""Build RFM features (calibration period) and K-means segments.

Also writes the lifetimes calibration/holdout summary table that
src/clv_model.py and src/churn_model.py reuse, so the whole pipeline
shares one train/validate split:

  - Calibration period: 2009-12-01 -> 2011-06-09 (~18 months)
  - Holdout period:     2011-06-09 -> 2011-12-10 (~6 months)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import pandas as pd
from lifetimes.utils import calibration_and_holdout_data
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CALIBRATION_END = pd.Timestamp("2011-06-09")
OBSERVATION_END = pd.Timestamp("2011-12-10")

SEGMENT_NAMES = {
    # filled in dynamically based on RFM ranking, see label_segments()
}


def label_segments(rfm: pd.DataFrame, summary: pd.DataFrame) -> dict[int, str]:
    """Assign business-readable names to clusters based on RFM medians."""
    s = summary.set_index("cluster")
    rev_rank = s["monetary_median_gbp"].rank(ascending=False)
    rec_rank = s["recency_median_days"].rank(ascending=True)
    freq_rank = s["frequency_median_orders"].rank(ascending=False)

    names = {}
    for cluster in s.index:
        recency = s.loc[cluster, "recency_median_days"]
        frequency = s.loc[cluster, "frequency_median_orders"]

        if recency > 150:
            # Lapsed customers: split by whether they ever showed repeat behavior
            if frequency >= 2:
                names[cluster] = "At-Risk"
            else:
                names[cluster] = "Hibernating"
        elif rev_rank[cluster] <= 1 and rec_rank[cluster] <= 2:
            names[cluster] = "Champions"
        elif rev_rank[cluster] <= 2 and freq_rank[cluster] <= 2:
            names[cluster] = "Loyal High-Value"
        else:
            names[cluster] = "Recent Low-Spend"
    return names


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
    warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

    clean = pd.read_parquet(PROCESSED_DIR / "transactions_clean.parquet")

    # ---- 1. Calibration / holdout split via lifetimes ----
    cal_holdout = calibration_and_holdout_data(
        transactions=clean,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        calibration_period_end=CALIBRATION_END,
        observation_period_end=OBSERVATION_END,
        freq="D",
        monetary_value_col="Amount",
    )
    cal_holdout.to_csv(PROCESSED_DIR / "lifetimes_summary.csv")
    print(f"Calibration/holdout summary: {len(cal_holdout):,} customers")
    print(f"  Calibration: {clean['InvoiceDate'].min().date()} -> {CALIBRATION_END.date()}")
    print(f"  Holdout:     {CALIBRATION_END.date()} -> {OBSERVATION_END.date()}")

    # ---- 2. RFM features computed directly from calibration-period transactions ----
    cal_txn = clean[clean["InvoiceDate"] <= CALIBRATION_END].copy()
    ref_date = CALIBRATION_END + pd.Timedelta(days=1)

    rfm = (
        cal_txn.groupby("CustomerID")
        .agg(
            recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
            frequency=("Invoice", "nunique"),
            monetary=("Amount", "sum"),
            first_order=("InvoiceDate", "min"),
            last_order=("InvoiceDate", "max"),
            country=("Country", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        )
        .reset_index()
    )
    print(f"\nRFM table (calibration period only): {len(rfm):,} customers")

    # ---- 3. K-means on log1p + standardized RFM ----
    features = pd.DataFrame(
        {
            "recency": np.log1p(rfm["recency"]),
            "frequency": np.log1p(rfm["frequency"]),
            "monetary": np.log1p(rfm["monetary"].clip(lower=0)),
        }
    )
    scaled = StandardScaler().fit_transform(features)

    diagnostics = []
    for k in range(2, 8):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled)
        diagnostics.append(
            {"k": k, "silhouette": silhouette_score(scaled, labels), "inertia": model.inertia_}
        )
    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(PROCESSED_DIR / "kmeans_diagnostics.csv", index=False)
    print("\nK-means diagnostics:")
    print(diag_df)

    K = 5
    model = KMeans(n_clusters=K, random_state=42, n_init=50)
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

    names = label_segments(rfm, summary)
    summary["segment_name"] = summary["cluster"].map(names)
    rfm["segment_name"] = rfm["cluster"].map(names)

    summary = summary.sort_values("revenue_pct", ascending=False).reset_index(drop=True)

    print(f"\nSegment summary (k={K}):")
    print(summary[["cluster", "segment_name", "customers", "customer_pct", "revenue_pct",
                    "recency_median_days", "frequency_median_orders", "monetary_median_gbp"]])

    rfm.to_csv(PROCESSED_DIR / "rfm_customers.csv", index=False)
    summary.to_csv(PROCESSED_DIR / "cluster_summary.csv", index=False)
    print(f"\nSaved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
