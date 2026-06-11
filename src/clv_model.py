"""Predict 6-month CLV with BG/NBD (purchase frequency) + Gamma-Gamma (order value).

Uses the calibration/holdout summary written by src/rfm_segments.py:
  - Fit BG/NBD on calibration-period frequency/recency/T
  - Fit Gamma-Gamma on calibration-period frequency/monetary_value (repeat purchasers)
  - Predict expected # purchases and average order value over the holdout window
  - Validate predictions against ACTUAL holdout-period revenue
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CALIBRATION_END = pd.Timestamp("2011-06-09")
OBSERVATION_END = pd.Timestamp("2011-12-10")


def main() -> None:
    warnings.filterwarnings("ignore")

    summary = pd.read_csv(PROCESSED_DIR / "lifetimes_summary.csv", index_col=0)
    rfm = pd.read_csv(PROCESSED_DIR / "rfm_customers.csv")
    clean = pd.read_parquet(PROCESSED_DIR / "transactions_clean.parquet")

    holdout_days = (OBSERVATION_END - CALIBRATION_END).days
    print(f"Holdout window: {holdout_days} days")

    # ---- 1. BG/NBD: predicted purchase frequency ----
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(summary["frequency_cal"], summary["recency_cal"], summary["T_cal"])
    print("\nBG/NBD fitted parameters:")
    print(bgf.summary)

    summary["predicted_purchases_6m"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        holdout_days, summary["frequency_cal"], summary["recency_cal"], summary["T_cal"]
    )

    # ---- 2. Gamma-Gamma: predicted average order value ----
    repeat_buyers = summary[(summary["frequency_cal"] > 0) & (summary["monetary_value_cal"] > 0)]
    print(f"\nGamma-Gamma fit on {len(repeat_buyers):,} repeat buyers (cal period)")
    print(
        "Frequency / monetary_value correlation (should be low):",
        round(repeat_buyers[["frequency_cal", "monetary_value_cal"]].corr().iloc[0, 1], 3),
    )

    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(repeat_buyers["frequency_cal"], repeat_buyers["monetary_value_cal"])
    print("\nGamma-Gamma fitted parameters:")
    print(ggf.summary)

    summary["predicted_avg_order_value"] = ggf.conditional_expected_average_profit(
        summary["frequency_cal"], summary["monetary_value_cal"]
    )
    summary["predicted_clv_6m"] = summary["predicted_purchases_6m"] * summary["predicted_avg_order_value"]

    # ---- 3. Actual holdout revenue (ground truth) ----
    holdout_txn = clean[
        (clean["InvoiceDate"] > CALIBRATION_END) & (clean["InvoiceDate"] <= OBSERVATION_END)
    ]
    actual_revenue = holdout_txn.groupby("CustomerID")["Amount"].sum().rename("actual_holdout_revenue")
    actual_orders = holdout_txn.groupby("CustomerID")["Invoice"].nunique().rename("actual_holdout_orders")

    summary = summary.join(actual_revenue, how="left").join(actual_orders, how="left")
    summary["actual_holdout_revenue"] = summary["actual_holdout_revenue"].fillna(0.0)
    summary["actual_holdout_orders"] = summary["actual_holdout_orders"].fillna(0).astype(int)

    # ---- 4. Merge with RFM segment labels ----
    out = summary.reset_index().rename(columns={"index": "CustomerID"})
    out = out.merge(rfm[["CustomerID", "segment_name", "country"]], on="CustomerID", how="left")
    out["segment_name"] = out["segment_name"].fillna("New (no cal. history)")

    # ---- 5. Validation metrics ----
    pred_purch_corr = out["predicted_purchases_6m"].corr(out["frequency_holdout"])
    pred_clv_corr = out["predicted_clv_6m"].corr(out["actual_holdout_revenue"])
    mae_purch = (out["predicted_purchases_6m"] - out["frequency_holdout"]).abs().mean()

    total_pred_clv = out["predicted_clv_6m"].sum()
    total_actual_rev = out["actual_holdout_revenue"].sum()

    print("\n--- Validation (calibration -> holdout) ---")
    print(f"Predicted vs actual purchase count: corr = {pred_purch_corr:.3f}, MAE = {mae_purch:.3f}")
    print(f"Predicted vs actual CLV/revenue: corr = {pred_clv_corr:.3f}")
    print(f"Total predicted 6m CLV (all customers): GBP {total_pred_clv:,.0f}")
    print(f"Total actual holdout revenue:           GBP {total_actual_rev:,.0f}")
    print(f"Portfolio-level error: {(total_pred_clv - total_actual_rev) / total_actual_rev:+.1%}")

    # Decile validation: rank by predicted CLV, check actual revenue per decile
    out["clv_decile"] = pd.qcut(out["predicted_clv_6m"].rank(method="first"), 10, labels=False)
    decile_check = out.groupby("clv_decile").agg(
        customers=("CustomerID", "count"),
        avg_predicted_clv=("predicted_clv_6m", "mean"),
        avg_actual_revenue=("actual_holdout_revenue", "mean"),
    )
    print("\nDecile check (decile 9 = highest predicted CLV):")
    print(decile_check)

    out.to_csv(PROCESSED_DIR / "clv_predictions.csv", index=False)

    metrics = pd.DataFrame(
        [
            {"metric": "holdout_days", "value": holdout_days},
            {"metric": "pred_purchase_corr", "value": round(pred_purch_corr, 4)},
            {"metric": "pred_purchase_mae", "value": round(mae_purch, 4)},
            {"metric": "pred_clv_corr", "value": round(pred_clv_corr, 4)},
            {"metric": "total_predicted_clv_6m_gbp", "value": round(total_pred_clv, 2)},
            {"metric": "total_actual_holdout_revenue_gbp", "value": round(total_actual_rev, 2)},
            {"metric": "portfolio_error_pct", "value": round((total_pred_clv - total_actual_rev) / total_actual_rev, 4)},
        ]
    )
    metrics.to_csv(PROCESSED_DIR / "clv_validation_metrics.csv", index=False)
    decile_check.to_csv(PROCESSED_DIR / "clv_decile_check.csv")

    print(f"\nSaved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
