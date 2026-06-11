"""Streamlit portfolio demo: RFM segmentation, CLV prediction & churn risk.

Bilingual (English / 中文). Reads pre-computed model outputs from
data/processed/*.csv (committed to the repo) so the public app is reproducible
without raw data, the `lifetimes` package, or re-running the modeling pipeline.
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
SEGMENT_ZH = {
    "Champions": "冠军客户",
    "Loyal High-Value": "高价值忠诚客户",
    "Recent Low-Spend": "新近低消费客户",
    "At-Risk": "流失风险客户",
    "Hibernating": "沉睡客户",
}
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

with st.sidebar:
    lang = st.radio("语言 / Language", ["English", "中文"], horizontal=True)

ZH = lang == "中文"


def L(en: str, zh: str) -> str:
    """Return the string for the selected language."""
    return zh if ZH else en


def seg_label(name: str) -> str:
    return f"{name}（{SEGMENT_ZH[name]}）" if ZH else name


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

st.title(L("RFM Segmentation, CLV Prediction & Churn Risk",
           "RFM 客户分群、CLV 预测与流失风险"))
st.caption(
    L(
        "UCI Online Retail II (2009-12 to 2011-12) -> RFM segments -> 6-month CLV "
        "(BG/NBD + Gamma-Gamma) -> churn risk model -> segment strategy & ROI simulation",
        "UCI Online Retail II 数据集（2009-12 至 2011-12）-> RFM 分群 -> 6 个月 CLV 预测"
        "（BG/NBD + Gamma-Gamma）-> 流失风险模型 -> 分群策略与 ROI 模拟",
    )
)

if MISSING:
    st.error(
        L(
            "Missing pre-computed data files: " + ", ".join(MISSING)
            + ". Run the pipeline in src/ (see README) to generate data/processed/*.csv.",
            "缺少预计算数据文件：" + ", ".join(MISSING)
            + "。请运行 src/ 中的管道（见 README）生成 data/processed/*.csv。",
        )
    )
    st.stop()

cluster_summary = DATA["cluster_summary"].set_index("segment_name").loc[SEGMENT_ORDER].reset_index()
rfm = DATA["rfm_customers"]
clv = DATA["clv_predictions"]
churn = DATA["churn_predictions"]
strategy = DATA["segment_strategy"]

tab_labels = (
    ["总览", "客户分群", "CLV 模型", "流失风险", "策略与 ROI 模拟", "方法论"]
    if ZH
    else ["Overview", "Segments", "CLV Model", "Churn Risk", "Strategy & ROI Simulator", "Methodology"]
)
tab_overview, tab_segments, tab_clv, tab_churn, tab_strategy, tab_method = st.tabs(tab_labels)

# ------------------------------------------------------------------
# Overview
# ------------------------------------------------------------------
with tab_overview:
    with st.expander(
        L("Why this project? — the motivation", "为什么做这个项目？— 项目动机"), expanded=True
    ):
        st.markdown(
            L(
                """
**The problem I wanted to solve.** Most CRM/growth teams spend their campaign budget
evenly across all customers because they can't answer three questions: *who is worth
protecting, who is worth winning back, and is the spend even worth it?* Meanwhile, most
portfolio RFM projects stop at clustering — pretty segments, no decisions.

**What this project does differently:**

1. **From segments to money.** Each segment gets a 6-month CLV prediction (BG/NBD +
   Gamma-Gamma), so "Champions" isn't a label — it's £X of future revenue at stake.
2. **From description to prediction.** A churn model (AUC 0.81) identifies who is about
   to go quiet — and I show *why* the textbook BTYD "P(alive)" heuristic fails (AUC 0.44)
   on one-time buyers.
3. **From prediction to decision.** Segment × CLV × churn risk → a cost-adjusted ROI
   simulation (£17.7K spend → £127.8K simulated net value) with per-segment A/B test
   sample-size requirements, so the plan is actually testable.
4. **Honest validation.** An 18-month calibration / 6-month holdout split — every claim
   is checked against what customers *actually did*, not in-sample fit.

It uses only transaction history — the one dataset every retailer already has.
""",
                """
**我想解决的问题。** 大多数 CRM/增长团队把营销预算平均撒向所有客户，因为他们回答不了
三个问题：*谁值得保护？谁值得挽回？这笔钱花得值不值？* 而大多数作品集里的 RFM 项目
止步于聚类——分群很漂亮，却不产生决策。

**这个项目的不同之处：**

1. **从分群到金额。** 每个分群都有 6 个月 CLV 预测（BG/NBD + Gamma-Gamma），
   "冠军客户"不只是标签，而是一笔可量化的未来收入。
2. **从描述到预测。** 流失模型（AUC 0.81）识别谁即将沉默——并且我展示了教科书式
   BTYD "P(alive)" 启发式为什么在一次性购买客户上失效（AUC 仅 0.44）。
3. **从预测到决策。** 分群 × CLV × 流失风险 → 成本调整后的 ROI 模拟
   （£1.77 万投入 → 模拟净值 £12.78 万），并给出每个分群的 A/B 测试样本量要求，
   让方案真正可验证。
4. **诚实的验证。** 18 个月校准期 / 6 个月留出期——所有结论都用客户*实际行为*检验，
   而不是样本内拟合。

整个项目只用交易流水数据——这是每个零售商都已经拥有的数据。
""",
            )
        )

    summary = DATA.get("data_summary")
    if summary is not None:
        s = summary.set_index("metric")["value"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(L("Customers analyzed", "分析客户数"), f"{int(float(s['customers'])):,}")
        c2.metric(L("Clean transactions", "清洗后交易数"), f"{int(float(s['clean_rows'])):,}")
        c3.metric(L("Revenue (cleaned)", "收入（清洗后）"), f"£{float(s['revenue_gbp']):,.0f}")
        c4.metric(L("Date range", "时间范围"), f"{s['date_min']} → {s['date_max']}")

    clv_metrics = DATA.get("clv_validation_metrics")
    churn_metrics = DATA.get("churn_model_metrics")
    c1, c2, c3, c4 = st.columns(4)
    if clv_metrics is not None:
        m = clv_metrics.set_index("metric")["value"]
        c1.metric(L("CLV pred-vs-actual corr.", "CLV 预测-实际相关性"), f"{float(m['pred_clv_corr']):.2f}")
        c2.metric(L("Portfolio CLV error", "组合 CLV 误差"), f"{float(m['portfolio_error_pct']):+.1%}")
    if churn_metrics is not None:
        best_auc = churn_metrics[churn_metrics["model"] != "btyd_baseline"]["auc"].max()
        btyd_auc = churn_metrics.loc[churn_metrics["model"] == "btyd_baseline", "auc"].iloc[0]
        c3.metric(L("Churn model AUC", "流失模型 AUC"), f"{best_auc:.2f}",
                  help=L("Best ML classifier (logistic regression)", "最佳机器学习分类器（逻辑回归）"))
        c4.metric(L("BTYD baseline AUC", "BTYD 基线 AUC"), f"{btyd_auc:.2f}",
                  help=L("1 - P(alive) from BG/NBD", "BG/NBD 模型的 1 - P(alive)"))

    blended_roi = strategy["incremental_revenue_gbp"].sum() - strategy["campaign_cost_gbp"].sum()
    blended_roi_x = blended_roi / strategy["campaign_cost_gbp"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric(L("Total campaign cost (sim.)", "总营销成本（模拟）"), f"£{strategy['campaign_cost_gbp'].sum():,.0f}")
    c2.metric(L("Simulated net value", "模拟净值"), f"£{blended_roi:,.0f}")
    c3.metric(L("Blended ROI", "综合 ROI"), f"{blended_roi_x:.1f}x")

    st.markdown(
        L(
            """
**The story in one line:** segment customers by RFM, predict what each segment is worth
over the next 6 months (CLV), predict who's about to go quiet (churn risk), then size a
campaign budget against the CLV that's actually at risk — and check whether each
segment is even big enough to run a valid A/B test.
""",
            """
**一句话讲清这个项目：** 用 RFM 给客户分群，预测每个分群未来 6 个月的价值（CLV），
预测谁即将流失（流失风险），再把营销预算对准真正"处于风险中的 CLV"——
并检验每个分群的样本量是否足以支撑一个有效的 A/B 测试。
""",
        )
    )

    col1, col2 = st.columns(2)
    fig1 = FIG_DIR / "fig1_rfm_segments.png"
    fig2 = FIG_DIR / "fig2_pareto.png"
    if fig1.exists():
        col1.image(str(fig1), caption=L("Segments: frequency vs. monetary, and revenue concentration",
                                        "分群：频次 vs. 金额，以及收入集中度"))
    if fig2.exists():
        col2.image(str(fig2), caption=L("25% of customers drive 80% of revenue",
                                        "25% 的客户贡献 80% 的收入"))

# ------------------------------------------------------------------
# Segments
# ------------------------------------------------------------------
with tab_segments:
    st.subheader(L("Segment summary (calibration period)", "分群概览（校准期）"))
    display_cols = {
        "segment_name": L("Segment", "分群"),
        "customers": L("Customers", "客户数"),
        "customer_pct": L("% of customers", "客户占比 %"),
        "revenue_pct": L("% of revenue", "收入占比 %"),
        "recency_median_days": L("Median recency (days)", "近度中位数（天）"),
        "frequency_median_orders": L("Median frequency (orders)", "频次中位数（订单）"),
        "monetary_median_gbp": L("Median monetary (£)", "金额中位数（£）"),
    }
    seg_table = cluster_summary[list(display_cols.keys())].rename(columns=display_cols)
    if ZH:
        seg_table[display_cols["segment_name"]] = seg_table[display_cols["segment_name"]].map(seg_label)
    st.dataframe(seg_table.set_index(display_cols["segment_name"]), use_container_width=True)

    st.subheader(L("Explore customers by segment", "按分群浏览客户"))
    selected = st.selectbox(L("Segment", "分群"), SEGMENT_ORDER, format_func=seg_label)
    seg_strategy_row = strategy[strategy["segment_name"] == selected].iloc[0]

    c1, c2, c3 = st.columns(3)
    seg_row = cluster_summary[cluster_summary["segment_name"] == selected].iloc[0]
    c1.metric(L("Customers", "客户数"), f"{int(seg_row['customers']):,}")
    c2.metric(L("Revenue share", "收入占比"), f"{seg_row['revenue_pct']:.1f}%")
    c3.metric(L("Avg 6m CLV", "平均 6 个月 CLV"), f"£{seg_strategy_row['avg_clv_6m_gbp']:,.0f}")

    st.info(f"**{L('Recommended intervention', '推荐干预策略')}:** {seg_strategy_row['intervention']}")

    seg_rfm = rfm[rfm["segment_name"] == selected].copy()
    seg_rfm["log_frequency"] = np.log1p(seg_rfm["frequency"])
    seg_rfm["log_monetary"] = np.log1p(seg_rfm["monetary"])

    st.caption(
        L(
            f"Recency / Frequency / Monetary distributions for {selected} ({len(seg_rfm):,} customers)",
            f"{seg_label(selected)} 的近度 / 频次 / 金额分布（{len(seg_rfm):,} 名客户）",
        )
    )
    c1, c2, c3 = st.columns(3)
    c1.bar_chart(seg_rfm["recency"].value_counts(bins=15).sort_index())
    c2.bar_chart(seg_rfm["frequency"].value_counts(bins=15).sort_index())
    c3.bar_chart(seg_rfm["monetary"].value_counts(bins=15).sort_index())

    with st.expander(L("Show raw customer-level data", "查看客户级原始数据")):
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
    st.subheader(L("BG/NBD + Gamma-Gamma: 6-month CLV prediction",
                   "BG/NBD + Gamma-Gamma：6 个月 CLV 预测"))

    clv_metrics = DATA.get("clv_validation_metrics")
    if clv_metrics is not None:
        m = clv_metrics.set_index("metric")["value"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(L("Purchase count corr.", "购买次数相关性"), f"{float(m['pred_purchase_corr']):.2f}")
        c2.metric(L("Purchase count MAE", "购买次数 MAE"), f"{float(m['pred_purchase_mae']):.2f}")
        c3.metric(L("CLV correlation", "CLV 相关性"), f"{float(m['pred_clv_corr']):.2f}")
        c4.metric(L("Portfolio error", "组合误差"), f"{float(m['portfolio_error_pct']):+.1%}")

    fig3 = FIG_DIR / "fig3_clv_validation.png"
    if fig3.exists():
        st.image(str(fig3), caption=L("Predicted vs. actual CLV, by decile and per customer",
                                      "预测 vs. 实际 CLV：按十分位与单客户"))

    st.subheader(L("Look up a customer", "查询单个客户"))
    cust_id = st.selectbox("CustomerID", sorted(clv["CustomerID"].unique()), index=0)
    row = clv[clv["CustomerID"] == cust_id].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    seg_name = row.get("segment_name", "n/a")
    c1.metric(L("Segment", "分群"), seg_label(seg_name) if seg_name in SEGMENT_ZH else seg_name)
    c2.metric(L("Predicted 6m purchases", "预测 6 个月购买次数"), f"{row['predicted_purchases_6m']:.2f}")
    c3.metric(L("Predicted 6m CLV", "预测 6 个月 CLV"), f"£{row['predicted_clv_6m']:,.0f}")
    c4.metric(L("Actual holdout revenue", "留出期实际收入"), f"£{row['actual_holdout_revenue']:,.0f}")

# ------------------------------------------------------------------
# Churn Risk
# ------------------------------------------------------------------
with tab_churn:
    st.subheader(L("Churn risk: ML classifier vs. BTYD heuristic",
                   "流失风险：机器学习分类器 vs. BTYD 启发式"))

    churn_metrics = DATA.get("churn_model_metrics")
    if churn_metrics is not None:
        st.dataframe(churn_metrics.set_index("model"), use_container_width=True)

    fig4 = FIG_DIR / "fig4_churn_roc.png"
    if fig4.exists():
        col1, col2 = st.columns([2, 1])
        col1.image(str(fig4), caption=L("ROC curve: logistic regression vs. BTYD 1-P(alive)",
                                        "ROC 曲线：逻辑回归 vs. BTYD 1-P(alive)"))
        with col2:
            st.markdown(
                L(
                    """
**Why does the BTYD heuristic fail?**

BG/NBD assigns `P(alive) = 1.0` to every customer with zero
repeat purchases in the calibration period — there's no
"dropout" evidence for a one-time buyer.

But empirically, **one-time buyers churn at ~74%**, vs. ~36%
for repeat buyers. The heuristic is backwards for exactly the
group it can't model: new/one-time customers.
""",
                    """
**BTYD 启发式为什么失效？**

对于校准期内没有复购的客户，BG/NBD 会给出
`P(alive) = 1.0`——一次性购买客户没有任何
"流失"证据可供模型学习。

但实际数据中，**一次性购买客户的流失率约 74%**，
而复购客户仅约 36%。这个启发式恰恰在它无法建模的
群体（新客户/一次性客户）上完全判反了。
""",
                )
            )

    fi = DATA.get("churn_feature_importance")
    if fi is not None:
        fi.columns = ["feature", "importance"]
        st.subheader(L("Feature importance (best model)", "特征重要性（最佳模型）"))
        st.bar_chart(fi.set_index("feature")["importance"])

# ------------------------------------------------------------------
# Strategy & ROI Simulator
# ------------------------------------------------------------------
with tab_strategy:
    st.subheader(L("Segment strategy: adjust assumptions and see the ROI change",
                   "分群策略：调整假设，实时查看 ROI 变化"))
    st.caption(
        L(
            "Customer counts, churn rates, and avg CLV come from the model. "
            "Cost-per-customer and retention-uplift assumptions are adjustable — "
            "try the values from reports/experiment_design.md or your own.",
            "客户数、流失率、平均 CLV 来自模型输出。单客户成本与留存提升假设可调——"
            "可以试试 reports/experiment_design.md 中的取值，或你自己的假设。",
        )
    )

    sim_rows = []
    for _, r in strategy.iterrows():
        seg = r["segment_name"]
        exp_title = L(
            f"{seg} — {r['customers']:,} customers, avg 6m CLV £{r['avg_clv_6m_gbp']:,.0f}",
            f"{seg_label(seg)} — {r['customers']:,} 名客户，平均 6 个月 CLV £{r['avg_clv_6m_gbp']:,.0f}",
        )
        with st.expander(exp_title, expanded=(seg == "At-Risk")):
            c1, c2 = st.columns(2)
            cost = c1.slider(
                L(f"Cost per customer (£) — {seg}", f"单客户成本（£）— {seg_label(seg)}"),
                min_value=0.0, max_value=20.0, value=float(r["cost_per_customer_gbp"]), step=0.5,
                key=f"cost_{seg}",
            )
            uplift_pp = c2.slider(
                L(f"Assumed retention uplift (pp) — {seg}",
                  f"假设留存提升（百分点）— {seg_label(seg)}"),
                min_value=0.0, max_value=20.0, value=float(r["assumed_retention_uplift_pp"]), step=0.5,
                key=f"uplift_{seg}",
            )
            campaign_cost = r["customers"] * cost
            incremental_revenue = r["customers"] * (uplift_pp / 100) * r["avg_clv_6m_gbp"]
            net_value = incremental_revenue - campaign_cost
            roi = net_value / campaign_cost if campaign_cost > 0 else float("nan")

            c1, c2, c3 = st.columns(3)
            c1.metric(L("Campaign cost", "营销成本"), f"£{campaign_cost:,.0f}")
            c2.metric(L("Incremental revenue", "增量收入"), f"£{incremental_revenue:,.0f}")
            c3.metric(L("Net value / ROI", "净值 / ROI"),
                      f"£{net_value:,.0f} ({roi:.1f}x)" if campaign_cost > 0 else "n/a")

            sim_rows.append(
                {
                    "segment_name": seg_label(seg),
                    "customers": r["customers"],
                    "campaign_cost": campaign_cost,
                    "incremental_revenue": incremental_revenue,
                    "net_value": net_value,
                    "roi": roi,
                }
            )

    sim_df = pd.DataFrame(sim_rows)
    st.subheader(L("Totals across all segments (with your adjustments)",
                   "所有分群合计（含你的调整）"))
    total_cost = sim_df["campaign_cost"].sum()
    total_incr = sim_df["incremental_revenue"].sum()
    total_net = total_incr - total_cost
    blended = total_net / total_cost if total_cost > 0 else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric(L("Total campaign cost", "总营销成本"), f"£{total_cost:,.0f}")
    c2.metric(L("Total net value", "总净值"), f"£{total_net:,.0f}")
    c3.metric(L("Blended ROI", "综合 ROI"), f"{blended:.1f}x" if total_cost > 0 else "n/a")

    st.bar_chart(sim_df.set_index("segment_name")[["net_value"]])

# ------------------------------------------------------------------
# Methodology
# ------------------------------------------------------------------
with tab_method:
    st.markdown(
        L(
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
""",
            """
### 数据与建模管道

1. **`src/data_prep.py`** — 加载并清洗 Online Retail II（805,549 条交易，5,878 名客户，£1,774 万收入）
2. **`src/rfm_segments.py`** — 校准期/留出期划分（18 个月 / 6 个月），基于校准期计算 RFM 特征，K-means 聚类（k=5）
3. **`src/clv_model.py`** — BG/NBD（购买时机）+ Gamma-Gamma（订单金额）→ 6 个月 CLV，用留出期实际收入验证
4. **`src/churn_model.py`** — 流失标签 = 留出期零购买；逻辑回归 / 随机森林 / 梯度提升 vs. BTYD `1-P(alive)` 基线
5. **`src/segment_strategy.py`** — 风险 CLV = 平均 CLV × 流失概率；每个分群的干预策略、A/B 测试样本量（statsmodels 功效分析）与 ROI 模拟
6. **`src/make_figures.py`** — 生成 `reports/figures/*.png`

### 校准期 / 留出期划分

- 校准期：2009-12-01 → 2011-06-09（约 18 个月）— 用于计算特征和训练模型
- 留出期：2011-06-09 → 2011-12-10（约 6 个月）— 仅用于验证预测

### 局限性

- 组合层面 CLV 偏保守约 16.6%（BG/NBD 在季节性数据上的典型偏差）。
- 模拟器中的留存提升假设仅为示意——应替换为试点实验中实测的效果。
- 本应用读取预计算的 `data/processed/*.csv`，不会实时重跑 `lifetimes`。

完整说明见 [GitHub 仓库](https://github.com/josephwang-ds/rfm-customer-segmentation)
中的 `README.md` 与 `reports/experiment_design.md`。
""",
        )
    )
