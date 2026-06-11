"""Streamlit portfolio demo: RFM segmentation, CLV prediction & churn risk.

Reads pre-computed model outputs from data/processed/*.csv (committed to the
repo) so the public app is reproducible without raw data, the `lifetimes`
package, or re-running the modeling pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

SEGMENT_ORDER = ["Champions", "Loyal High-Value", "Recent Low-Spend", "At-Risk", "Hibernating"]
SEGMENT_COLORS = {
    "Champions": "#2E7D32",
    "Loyal High-Value": "#1565C0",
    "Recent Low-Spend": "#F9A825",
    "At-Risk": "#EF6C00",
    "Hibernating": "#9E9E9E",
}

st.set_page_config(
    page_title="RFM Segmentation, CLV & Churn Risk",
    page_icon="\U0001F6CD️",
    layout="wide",
)


@st.cache_data
def load_data():
    data = {}
    for name in [
        "data_summary",
        "cluster_summary",
        "rfm_customers",
        "clv_predictions",
        "clv_validation_metrics",
        "clv_decile_check",
        "churn_predictions",
        "churn_model_metrics",
        "churn_feature_importance",
        "segment_strategy",
    ]:
        path = PROCESSED_DIR / f"{name}.csv"
        if path.exists():
            data[name] = pd.read_csv(path)
    return data


DATA = load_data()
MISSING = [k for k in ["cluster_summary", "rfm_customers", "clv_predictions",
                        "churn_predictions", "segment_strategy"] if k not in DATA]

st.title("RFM Segmentation, CLV Prediction & Churn Risk")
st.caption(
    "UCI Online Retail II (2009-12 to 2011-12) -> RFM segments -> 6-month CLV "
    "(BG/NBD + Gamma-Gamma) -> churn risk model -> segment strategy & ROI simulation"
)

if MISSING:
    st.error(
        "Missing pre-computed data files: "
        + ", ".join(MISSING)
        + ". Run the pipeline in src/ (see README) to generate data/processed/*.csv."
    )
    st.stop()

cluster_summary = DATA["cluster_summary"].set_index("segment_name").loc[SEGMENT_ORDER].reset_index()
rfm = DATA["rfm_customers"]
clv = DATA["clv_predictions"]
churn = DATA["churn_predictions"]
strategy = DATA["segment_strategy"]

tab_overview, tab_segments, tab_clv, tab_churn, tab_strategy, tab_method = st.tabs(
    ["Overview", "Segments", "CLV Model", "Churn Risk", "Strategy & ROI Simulator", "Methodology"]
)

# ------------------------------------------------------------------
# Overview
# ------------------------------------------------------------------
with tab_overview:
    summary = DATA.get("data_summary")
    if summary is not None:
        s = summary.set_index("metric")["value"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers analyzed", f"{int(float(s['customers'])):,}")
        c2.metric("Clean transactions", f"{int(float(s['clean_rows'])):,}")
        c3.metric("Revenue (cleaned)", f"£{float(s['revenue_gbp']):,.0f}")
        c4.metric("Date range", f"{s['date_min']} → {s['date_max']}")

    clv_metrics = DATA.get("clv_validation_metrics")
    churn_metrics = DATA.get("churn_model_metrics")
    c1, c2, c3, c4 = st.columns(4)
    if clv_metrics is not None:
        m = clv_metrics.set_index("metric")["value"]
        c1.metric("CLV pred-vs-actual corr.", f"{float(m['pred_clv_corr']):.2f}")
        c2.metric("Portfolio CLV error", f"{float(m['portfolio_error_pct']):+.1%}")
    if churn_metrics is not None:
        best_auc = churn_metrics[churn_metrics["model"] != "btyd_baseline"]["auc"].max()
        btyd_auc = churn_metrics.loc[churn_metrics["model"] == "btyd_baseline", "auc"].iloc[0]
        c3.metric("Churn model AUC", f"{best_auc:.2f}", help="Best ML classifier (logistic regression)")
        c4.metric("BTYD baseline AUC", f"{btyd_auc:.2f}", help="1 - P(alive) from BG/NBD")

    blended_roi = strategy["incremental_revenue_gbp"].sum() - strategy["campaign_cost_gbp"].sum()
    blended_roi_x = (strategy["incremental_revenue_gbp"].sum() - strategy["campaign_cost_gbp"].sum()) / strategy["campaign_cost_gbp"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total campaign cost (sim.)", f"£{strategy['campaign_cost_gbp'].sum():,.0f}")
    c2.metric("Simulated net value", f"£{blended_roi:,.0f}")
    c3.metric("Blended ROI", f"{blended_roi_x:.1f}x")

    st.markdown(
        """
**The story in one line:** segment customers by RFM, predict what each segment is worth
over the next 6 months (CLV), predict who's about to go quiet (churn risk), then size a
campaign budget against the CLV that's actually at risk — and check whether each
segment is even big enough to run a valid A/B test.
"""
    )

    col1, col2 = st.columns(2)
    fig1 = FIG_DIR / "fig1_rfm_segments.png"
    fig2 = FIG_DIR / "fig2_pareto.png"
    if fig1.exists():
        col1.image(str(fig1), caption="Segments: frequency vs. monetary, and revenue concentration")
    if fig2.exists():
        col2.image(str(fig2), caption="25% of customers drive 80% of revenue")

# ------------------------------------------------------------------
# Segments
# ------------------------------------------------------------------
with tab_segments:
    st.subheader("Segment summary (calibration period)")
    display_cols = {
        "segment_name": "Segment",
        "customers": "Customers",
        "customer_pct": "% of customers",
        "revenue_pct": "% of revenue",
        "recency_median_days": "Median recency (days)",
        "frequency_median_orders": "Median frequency (orders)",
        "monetary_median_gbp": "Median monetary (£)",
    }
    st.dataframe(
        cluster_summary[list(display_cols.keys())].rename(columns=display_cols).set_index("Segment"),
        use_container_width=True,
    )

    st.subheader("Explore customers by segment")
    selected = st.selectbox("Segment", SEGMENT_ORDER)
    seg_strategy_row = strategy[strategy["segment_name"] == selected].iloc[0]

    c1, c2, c3 = st.columns(3)
    seg_row = cluster_summary[cluster_summary["segment_name"] == selected].iloc[0]
    c1.metric("Customers", f"{int(seg_row['customers']):,}")
    c2.metric("Revenue share", f"{seg_row['revenue_pct']:.1f}%")
    c3.metric("Avg 6m CLV", f"£{seg_strategy_row['avg_clv_6m_gbp']:,.0f}")

    st.info(f"**Recommended intervention:** {seg_strategy_row['intervention']}")

    seg_rfm = rfm[rfm["segment_name"] == selected].copy()
    seg_rfm["log_frequency"] = np.log1p(seg_rfm["frequency"])
    seg_rfm["log_monetary"] = np.log1p(seg_rfm["monetary"])

    st.caption(f"Recency / Frequency / Monetary distributions for {selected} ({len(seg_rfm):,} customers)")
    c1, c2, c3 = st.columns(3)
    c1.bar_chart(seg_rfm["recency"].value_counts(bins=15).sort_index())
    c2.bar_chart(seg_rfm["frequency"].value_counts(bins=15).sort_index())
    c3.bar_chart(seg_rfm["monetary"].value_counts(bins=15).sort_index())

    with st.expander("Show raw customer-level data"):
        st.dataframe(
            seg_rfm[["CustomerID", "recency", "frequency", "monetary", "country"]]
            .sort_values("monetary", ascending=False)
            .head(200),
            use_container_width=True,
        )

# ------------------------------------------------------------------
# CLV Model
# ------------------------------------------------------------------
with tab_clv:
    st.subheader("BG/NBD + Gamma-Gamma: 6-month CLV prediction")

    clv_metrics = DATA.get("clv_validation_metrics")
    if clv_metrics is not None:
        m = clv_metrics.set_index("metric")["value"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Purchase count corr.", f"{float(m['pred_purchase_corr']):.2f}")
        c2.metric("Purchase count MAE", f"{float(m['pred_purchase_mae']):.2f}")
        c3.metric("CLV correlation", f"{float(m['pred_clv_corr']):.2f}")
        c4.metric("Portfolio error", f"{float(m['portfolio_error_pct']):+.1%}")

    fig3 = FIG_DIR / "fig3_clv_validation.png"
    if fig3.exists():
        st.image(str(fig3), caption="Predicted vs. actual CLV, by decile and per customer")

    st.subheader("Look up a customer")
    cust_id = st.selectbox("CustomerID", sorted(clv["CustomerID"].unique()), index=0)
    row = clv[clv["CustomerID"] == cust_id].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segment", row.get("segment_name", "n/a"))
    c2.metric("Predicted 6m purchases", f"{row['predicted_purchases_6m']:.2f}")
    c3.metric("Predicted 6m CLV", f"£{row['predicted_clv_6m']:,.0f}")
    c4.metric("Actual holdout revenue", f"£{row['actual_holdout_revenue']:,.0f}")

# ------------------------------------------------------------------
# Churn Risk
# ------------------------------------------------------------------
with tab_churn:
    st.subheader("Churn risk: ML classifier vs. BTYD heuristic")

    churn_metrics = DATA.get("churn_model_metrics")
    if churn_metrics is not None:
        st.dataframe(churn_metrics.set_index("model"), use_container_width=True)

    fig4 = FIG_DIR / "fig4_churn_roc.png"
    if fig4.exists():
        col1, col2 = st.columns([2, 1])
        col1.image(str(fig4), caption="ROC curve: logistic regression vs. BTYD 1-P(alive)")
        with col2:
            st.markdown(
                """
**Why does the BTYD heuristic fail?**

BG/NBD assigns `P(alive) = 1.0` to every customer with zero
repeat purchases in the calibration period — there's no
"dropout" evidence for a one-time buyer.

But empirically, **one-time buyers churn at ~74%**, vs. ~36%
for repeat buyers. The heuristic is backwards for exactly the
group it can't model: new/one-time customers.
"""
            )

    fi = DATA.get("churn_feature_importance")
    if fi is not None:
        fi.columns = ["feature", "importance"]
        st.subheader("Feature importance (best model)")
        st.bar_chart(fi.set_index("feature")["importance"])

# ------------------------------------------------------------------
# Strategy & ROI Simulator
# ------------------------------------------------------------------
with tab_strategy:
    st.subheader("Segment strategy: adjust assumptions and see the ROI change")
    st.caption(
        "Customer counts, churn rates, and avg CLV come from the model. "
        "Cost-per-customer and retention-uplift assumptions are adjustable — "
        "try the values from reports/experiment_design.md or your own."
    )

    sim_rows = []
    for _, r in strategy.iterrows():
        seg = r["segment_name"]
        with st.expander(f"{seg} — {r['customers']:,} customers, avg 6m CLV £{r['avg_clv_6m_gbp']:,.0f}", expanded=(seg == "At-Risk")):
            c1, c2 = st.columns(2)
            cost = c1.slider(
                f"Cost per customer (£) — {seg}",
                min_value=0.0, max_value=20.0, value=float(r["cost_per_customer_gbp"]), step=0.5,
                key=f"cost_{seg}",
            )
            uplift_pp = c2.slider(
                f"Assumed retention uplift (pp) — {seg}",
                min_value=0.0, max_value=20.0, value=float(r["assumed_retention_uplift_pp"]), step=0.5,
                key=f"uplift_{seg}",
            )
            campaign_cost = r["customers"] * cost
            incremental_revenue = r["customers"] * (uplift_pp / 100) * r["avg_clv_6m_gbp"]
            net_value = incremental_revenue - campaign_cost
            roi = net_value / campaign_cost if campaign_cost > 0 else float("nan")

            c1, c2, c3 = st.columns(3)
            c1.metric("Campaign cost", f"£{campaign_cost:,.0f}")
            c2.metric("Incremental revenue", f"£{incremental_revenue:,.0f}")
            c3.metric("Net value / ROI", f"£{net_value:,.0f} ({roi:.1f}x)" if campaign_cost > 0 else "n/a")

            sim_rows.append(
                {
                    "segment_name": seg,
                    "customers": r["customers"],
                    "campaign_cost": campaign_cost,
                    "incremental_revenue": incremental_revenue,
                    "net_value": net_value,
                    "roi": roi,
                }
            )

    sim_df = pd.DataFrame(sim_rows)
    st.subheader("Totals across all segments (with your adjustments)")
    total_cost = sim_df["campaign_cost"].sum()
    total_incr = sim_df["incremental_revenue"].sum()
    total_net = total_incr - total_cost
    blended = total_net / total_cost if total_cost > 0 else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total campaign cost", f"£{total_cost:,.0f}")
    c2.metric("Total net value", f"£{total_net:,.0f}")
    c3.metric("Blended ROI", f"{blended:.1f}x" if total_cost > 0 else "n/a")

    st.bar_chart(sim_df.set_index("segment_name")[["net_value"]])

# ------------------------------------------------------------------
# Methodology
# ------------------------------------------------------------------
with tab_method:
    st.markdown(
        """
### Pipeline

1. **`src/data_prep.py`** — load & clean Online Retail II (805,549 transactions, 5,878 customers, £17.74M revenue)
2. **`src/rfm_segments.py`** — calibration/holdout split (18mo / 6mo), RFM features from calibration period, K-means (k=5)
3. **`src/clv_model.py`** — BG/NBD (purchase timing) + Gamma-Gamma (order value) → 6-month CLV, validated against actual holdout revenue
4. **`src/churn_model.py`** — churn label = zero purchases in holdout; logistic regression / random forest / gradient boosting vs. BTYD `1-P(alive)` baseline
5. **`src/segment_strategy.py`** — CLV at risk = avg CLV × churn probability; per-segment intervention, A/B test sample size (statsmodels power analysis), and ROI simulation
6. **`src/make_figures.py`** — generates `reports/figures/*.png`

### Calibration / Holdout split

- Calibration: 2009-12-01 → 2011-06-09 (~18 months) — used to compute features and train models
- Holdout: 2011-06-09 → 2011-12-10 (~6 months) — used only to validate predictions

### Limitations

- Portfolio-level CLV is conservative by ~16.6% (typical BG/NBD bias on seasonal data).
- Retention-uplift assumptions in the simulator are illustrative — replace with measured effects from a pilot.
- This app reads pre-computed `data/processed/*.csv`; it does not re-run `lifetimes` live.

Full write-up: see `README.md` and `reports/experiment_design.md` in the
[GitHub repository](https://github.com/josephwang-ds/rfm-customer-segmentation).
"""
    )
