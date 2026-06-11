"""Generate the figure set for the project report / README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

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


def fig_rfm_segments(rfm: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for seg in SEGMENT_ORDER:
        d = rfm[rfm["segment_name"] == seg]
        ax.scatter(
            d["frequency"], d["monetary"].clip(lower=1),
            s=14, alpha=0.5, label=seg, color=SEGMENT_COLORS[seg],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (orders, calibration period)")
    ax.set_ylabel("Monetary (GBP, calibration period)")
    ax.set_title("Customer Segments: Frequency vs. Monetary Value")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    summary = pd.read_csv(PROCESSED_DIR / "cluster_summary.csv").set_index("segment_name").loc[SEGMENT_ORDER]
    x = np.arange(len(SEGMENT_ORDER))
    width = 0.38
    ax.bar(x - width / 2, summary["customer_pct"], width, label="% of customers",
           color=[SEGMENT_COLORS[s] for s in SEGMENT_ORDER], alpha=0.5)
    ax.bar(x + width / 2, summary["revenue_pct"], width, label="% of revenue",
           color=[SEGMENT_COLORS[s] for s in SEGMENT_ORDER], alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(SEGMENT_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Share (%)")
    ax.set_title("Segment Size vs. Revenue Contribution")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_rfm_segments.png", dpi=150)
    plt.close(fig)


def fig_pareto(rfm: pd.DataFrame) -> None:
    df = rfm.sort_values("monetary", ascending=False).reset_index(drop=True)
    df["cum_revenue_pct"] = df["monetary"].cumsum() / df["monetary"].sum() * 100
    df["cum_customer_pct"] = (df.index + 1) / len(df) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["cum_customer_pct"], df["cum_revenue_pct"], color="#1565C0", lw=2)
    ax.plot([0, 100], [0, 100], "--", color="grey", lw=1, label="Equality line")

    # annotate the 80/20 point
    idx80 = (df["cum_revenue_pct"] >= 80).idxmax()
    cust80 = df.loc[idx80, "cum_customer_pct"]
    ax.axhline(80, color="#EF6C00", lw=1, ls=":")
    ax.axvline(cust80, color="#EF6C00", lw=1, ls=":")
    ax.annotate(
        f"{cust80:.0f}% of customers\n-> 80% of revenue",
        xy=(cust80, 80), xytext=(cust80 + 12, 55),
        arrowprops=dict(arrowstyle="->", color="#EF6C00"),
        fontsize=9, color="#EF6C00",
    )

    ax.set_xlabel("Cumulative % of customers (ranked by spend)")
    ax.set_ylabel("Cumulative % of revenue")
    ax.set_title("Revenue Concentration (Pareto Curve)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_pareto.png", dpi=150)
    plt.close(fig)


def fig_clv_validation(clv: pd.DataFrame) -> None:
    decile = pd.read_csv(PROCESSED_DIR / "clv_decile_check.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x = np.arange(len(decile))
    width = 0.38
    ax.bar(x - width / 2, decile["avg_predicted_clv"], width, label="Predicted 6m CLV", color="#1565C0")
    ax.bar(x + width / 2, decile["avg_actual_revenue"], width, label="Actual holdout revenue", color="#2E7D32")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i}" for i in decile["clv_decile"]])
    ax.set_xlabel("Predicted-CLV decile (D9 = highest)")
    ax.set_ylabel("GBP (avg per customer)")
    ax.set_title("CLV Model Validation: Predicted vs. Actual by Decile")
    ax.legend(fontsize=9)

    ax = axes[1]
    sample = clv.sample(min(2000, len(clv)), random_state=42)
    ax.scatter(
        sample["predicted_clv_6m"].clip(lower=1),
        sample["actual_holdout_revenue"].clip(lower=1),
        s=8, alpha=0.3, color="#1565C0",
    )
    lims = [1, max(sample["predicted_clv_6m"].max(), sample["actual_holdout_revenue"].max())]
    ax.plot(lims, lims, "--", color="grey", lw=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted 6m CLV (GBP, log scale)")
    ax.set_ylabel("Actual holdout revenue (GBP, log scale)")
    ax.set_title("Predicted vs. Actual Revenue (per customer)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_clv_validation.png", dpi=150)
    plt.close(fig)


def fig_churn_roc(churn: pd.DataFrame) -> None:
    test = churn[churn["in_test_set"]]

    fig, ax = plt.subplots(figsize=(6, 6))
    for col, label, color in [
        ("churn_risk_score", "Logistic Regression (best model)", "#1565C0"),
        ("btyd_churn_risk", "BTYD heuristic (1 - P(alive))", "#9E9E9E"),
    ]:
        fpr, tpr, _ = roc_curve(test["churned"], test[col])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "--", color="lightgrey", lw=1, label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Churn Risk Model: ROC Curve (held-out test customers)")
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_churn_roc.png", dpi=150)
    plt.close(fig)


def fig_segment_strategy(strategy: pd.DataFrame) -> None:
    df = strategy.set_index("segment_name").loc[
        [s for s in SEGMENT_ORDER if s in strategy["segment_name"].values]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    norm = plt.Normalize(0, 1)
    heat = pd.DataFrame(
        {
            "Avg CLV (6m, normalized)": df["avg_clv_6m_gbp"] / df["avg_clv_6m_gbp"].max(),
            "Churn risk": df["actual_churn_rate"],
            "CLV at risk (normalized)": df["clv_at_risk_6m_gbp"] / df["clv_at_risk_6m_gbp"].max(),
        }
    )
    im = ax.imshow(heat.values, cmap="OrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=15, ha="right", fontsize=9)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=9)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Segment Risk/Value Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    x = np.arange(len(df))
    ax.bar(x, df["net_value_gbp"], color=[SEGMENT_COLORS[s] for s in df.index])
    for i, (roi, val) in enumerate(zip(df["roi"], df["net_value_gbp"])):
        ax.text(i, val + 1000, f"ROI {roi:.1f}x", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=20, ha="right")
    ax.set_ylabel("Simulated net value (GBP, 6m)")
    ax.set_title("Segment Strategy: Simulated Net Value & ROI")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_segment_strategy.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rfm = pd.read_csv(PROCESSED_DIR / "rfm_customers.csv")
    clv = pd.read_csv(PROCESSED_DIR / "clv_predictions.csv")
    churn = pd.read_csv(PROCESSED_DIR / "churn_predictions.csv")
    strategy = pd.read_csv(PROCESSED_DIR / "segment_strategy.csv")

    fig_rfm_segments(rfm)
    fig_pareto(rfm)
    fig_clv_validation(clv)
    fig_churn_roc(churn)
    fig_segment_strategy(strategy)

    print(f"Saved 5 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
