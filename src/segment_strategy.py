"""Segment x CLV x churn-risk strategy: experiment design + cost-adjusted ROI simulation.

For each RFM segment we define a targeted intervention, compute the
required A/B test sample size (power analysis on the "made another
purchase in the next 6 months" metric), and simulate the cost-adjusted
ROI of a full-segment rollout assuming the intervention closes part of
the observed churn gap.

All assumptions (MDE, cost per customer, uplift) are stated explicitly
in STRATEGY below so they can be challenged/adjusted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ALPHA = 0.05
POWER = 0.80

# segment_name -> (intervention, cost_per_customer_gbp, assumed_mde_in_retention_rate)
STRATEGY = {
    "Champions": (
        "VIP early-access + loyalty perks (retain & deepen, not win back)",
        3.0,
        0.02,
    ),
    "Loyal High-Value": (
        "Personalized cross-sell/bundle email based on purchase history",
        2.0,
        0.05,
    ),
    "At-Risk": (
        "Win-back campaign: 15% discount + personalized email",
        8.0,
        0.08,
    ),
    "Recent Low-Spend": (
        "3-email onboarding/nurture series to build repeat-purchase habit",
        3.0,
        0.06,
    ),
    "Hibernating": (
        "Low-cost automated reactivation email (single send)",
        1.0,
        0.04,
    ),
}


def required_sample_size(baseline_rate: float, mde: float) -> int:
    """Per-arm sample size for a two-proportion test (alpha=0.05, power=0.80)."""
    p1, p2 = baseline_rate, min(baseline_rate + mde, 0.999)
    effect_size = proportion_effectsize(p2, p1)
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=effect_size, alpha=ALPHA, power=POWER, ratio=1.0)
    return int(np.ceil(n))


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "churn_predictions.csv")

    seg = (
        df.groupby("segment_name")
        .agg(
            customers=("CustomerID", "count"),
            avg_clv_6m=("predicted_clv_6m", "mean"),
            total_clv_6m=("predicted_clv_6m", "sum"),
            avg_churn_risk=("churn_risk_score", "mean"),
            actual_churn_rate=("churned", "mean"),
        )
        .reset_index()
    )
    seg["clv_at_risk_6m"] = seg["total_clv_6m"] * seg["avg_churn_risk"]
    seg["baseline_retention_rate"] = 1 - seg["actual_churn_rate"]

    rows = []
    for _, r in seg.iterrows():
        name = r["segment_name"]
        intervention, cost_per_cust, mde = STRATEGY[name]

        n_per_arm = required_sample_size(r["baseline_retention_rate"], mde)
        feasible = n_per_arm <= r["customers"] / 2

        customers = r["customers"]
        campaign_cost = customers * cost_per_cust
        incremental_retained_customers = customers * mde
        incremental_revenue = incremental_retained_customers * r["avg_clv_6m"]
        net_value = incremental_revenue - campaign_cost
        roi = net_value / campaign_cost

        rows.append(
            {
                "segment_name": name,
                "customers": int(customers),
                "avg_clv_6m_gbp": round(r["avg_clv_6m"], 2),
                "actual_churn_rate": round(r["actual_churn_rate"], 3),
                "clv_at_risk_6m_gbp": round(r["clv_at_risk_6m"], 2),
                "intervention": intervention,
                "cost_per_customer_gbp": cost_per_cust,
                "assumed_retention_uplift_pp": mde * 100,
                "required_n_per_arm": n_per_arm,
                "ab_test_feasible": feasible,
                "campaign_cost_gbp": round(campaign_cost, 2),
                "incremental_revenue_gbp": round(incremental_revenue, 2),
                "net_value_gbp": round(net_value, 2),
                "roi": round(roi, 2),
            }
        )

    out = pd.DataFrame(rows).sort_values("net_value_gbp", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(out[["segment_name", "customers", "avg_clv_6m_gbp", "actual_churn_rate",
                "clv_at_risk_6m_gbp", "cost_per_customer_gbp", "assumed_retention_uplift_pp",
                "required_n_per_arm", "ab_test_feasible", "roi", "net_value_gbp"]])

    print(f"\nTotal campaign cost: GBP {out['campaign_cost_gbp'].sum():,.0f}")
    print(f"Total incremental revenue (if MDE achieved): GBP {out['incremental_revenue_gbp'].sum():,.0f}")
    print(f"Total net value: GBP {out['net_value_gbp'].sum():,.0f}")
    print(f"Blended ROI: {out['net_value_gbp'].sum() / out['campaign_cost_gbp'].sum():.2f}")

    out.to_csv(PROCESSED_DIR / "segment_strategy.csv", index=False)
    print(f"\nSaved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
