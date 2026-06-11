"""Churn risk model: will a customer make zero purchases in the holdout window?

Features come entirely from the calibration period (no leakage). The label
is whether the customer made any purchase in the 6-month holdout window
(frequency_holdout == 0 -> churned). We compare:
  - A BTYD heuristic: 1 - BG/NBD P(alive) at end of calibration
  - ML classifiers: Logistic Regression, Random Forest, Gradient Boosting
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURES = [
    "recency_days",
    "frequency_orders",
    "monetary_gbp",
    "avg_order_value",
    "tenure_days",
    "recency_ratio",
    "predicted_purchases_6m",
    "predicted_clv_6m",
]


def main() -> None:
    warnings.filterwarnings("ignore")

    clv = pd.read_csv(PROCESSED_DIR / "clv_predictions.csv")

    df = clv.copy()
    df["churned"] = (df["frequency_holdout"] == 0).astype(int)
    df["recency_days"] = df["recency_cal"]
    df["frequency_orders"] = df["frequency_cal"] + 1  # +1 = first purchase, lifetimes excludes it
    df["monetary_gbp"] = df["monetary_value_cal"] * df["frequency_cal"] + df["monetary_value_cal"].where(
        df["frequency_cal"] == 0, 0
    )
    # Simpler & robust: just use total cal spend approximated from monetary_value_cal * frequency
    # (monetary_value_cal is avg of repeat txns). For a cleaner total spend, fall back to RFM table.
    rfm = pd.read_csv(PROCESSED_DIR / "rfm_customers.csv")[["CustomerID", "monetary", "frequency", "recency"]]
    rfm = rfm.rename(columns={"monetary": "monetary_gbp_rfm", "frequency": "frequency_orders_rfm", "recency": "recency_days_rfm"})
    df = df.merge(rfm, on="CustomerID", how="left")

    df["recency_days"] = df["recency_days_rfm"].fillna(df["recency_days"])
    df["frequency_orders"] = df["frequency_orders_rfm"].fillna(df["frequency_orders"])
    df["monetary_gbp"] = df["monetary_gbp_rfm"].fillna(df["monetary_gbp"])

    df["avg_order_value"] = df["monetary_gbp"] / df["frequency_orders"].replace(0, 1)
    df["tenure_days"] = df["T_cal"]
    df["recency_ratio"] = df["recency_days"] / df["tenure_days"].replace(0, 1)

    print(f"Total customers: {len(df):,}")
    print(f"Churn rate (no purchase in holdout): {df['churned'].mean():.1%}")

    # ---- Baseline: BG/NBD probability alive ----
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(df["frequency_cal"], df["recency_cal"], df["T_cal"])
    df["p_alive"] = bgf.conditional_probability_alive(df["frequency_cal"], df["recency_cal"], df["T_cal"])
    df["btyd_churn_risk"] = 1 - df["p_alive"]
    baseline_auc = roc_auc_score(df["churned"], df["btyd_churn_risk"])
    print(f"\nBaseline (BTYD 1 - P(alive)) AUC: {baseline_auc:.3f}")

    # ---- ML classifiers ----
    X = df[FEATURES].fillna(0)
    y = df["churned"]
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    logreg = LogisticRegression(max_iter=2000)
    logreg.fit(X_train_scaled, y_train)
    proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]
    results["logistic_regression"] = roc_auc_score(y_test, proba_lr)

    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    proba_rf = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = roc_auc_score(y_test, proba_rf)

    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    proba_gb = gb.predict_proba(X_test)[:, 1]
    results["gradient_boosting"] = roc_auc_score(y_test, proba_gb)

    btyd_test_auc = roc_auc_score(y_test, df.loc[idx_test, "btyd_churn_risk"])
    results["btyd_baseline"] = btyd_test_auc

    print("\nModel AUC on held-out test customers:")
    for name, auc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:20s} {auc:.3f}")

    best_name = max(
        {k: v for k, v in results.items() if k != "btyd_baseline"}, key=results.get
    )
    print(f"\nBest model: {best_name}")

    best_model = {"logistic_regression": logreg, "random_forest": rf, "gradient_boosting": gb}[best_name]
    if best_name == "logistic_regression":
        full_proba = best_model.predict_proba(scaler.transform(X))[:, 1]
        importance = pd.Series(np.abs(best_model.coef_[0]), index=FEATURES).sort_values(ascending=False)
    else:
        full_proba = best_model.predict_proba(X)[:, 1]
        importance = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    print("\nFeature importance:")
    print(importance)

    df["churn_risk_score"] = full_proba
    df["in_test_set"] = df.index.isin(idx_test)

    out_cols = [
        "CustomerID",
        "segment_name",
        "country",
        "recency_days",
        "frequency_orders",
        "monetary_gbp",
        "avg_order_value",
        "tenure_days",
        "predicted_clv_6m",
        "p_alive",
        "btyd_churn_risk",
        "churn_risk_score",
        "churned",
        "in_test_set",
    ]
    df[out_cols].to_csv(PROCESSED_DIR / "churn_predictions.csv", index=False)

    metrics = pd.DataFrame(
        [{"model": k, "auc": round(v, 4)} for k, v in results.items()]
    )
    metrics.to_csv(PROCESSED_DIR / "churn_model_metrics.csv", index=False)
    importance.rename("importance").to_csv(PROCESSED_DIR / "churn_feature_importance.csv")

    print(f"\nSaved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
