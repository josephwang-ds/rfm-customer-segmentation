# RFM, CLV & Churn Risk for Retail CRM

This demo answers one practical CRM question:

> **If campaign budget is limited, which customers are worth protecting or winning back?**

It uses two years of UK e-commerce transactions to build a simple decision chain:

```text
RFM segmentation -> 6-month CLV -> churn risk -> CRM action
```

The most important story is not the clustering. It is the model validation lesson:

> A textbook BTYD `1 - P(alive)` churn heuristic scored **0.44 AUC**, worse than random, because it misread one-time buyers. A supervised churn model fixed the issue and reached **0.81 AUC**.

## Live Demo

- Streamlit demo: https://josephwang-rfm-segmentation.streamlit.app/
- GitHub repository: https://github.com/josephwang-ds/rfm-customer-segmentation

## What This Project Shows

Many CRM teams segment customers but still struggle to decide where to spend money. This project keeps the workflow focused:

1. **Segment customers** with RFM so the team knows who they are.
2. **Estimate future value** with 6-month CLV.
3. **Predict churn risk** using a validated supervised model.
4. **Translate scores into actions**: protect, cross-sell, win back, onboard, or low-cost reactivate.

## Dataset

Source: UCI Online Retail II, a public UK e-commerce transaction dataset.

After cleaning:

- **805,549** transaction rows
- **5,878** customers
- **£17.74M** revenue
- 18-month calibration window + 6-month holdout window

The holdout design matters because it mirrors real CRM work: score customers today, then check what they actually do in the next two quarters.

## Core Results

### 1. RFM Finds the Customer Structure

Five customer groups:

- **Champions**: recent, frequent, high-value buyers
- **Loyal High-Value**: repeat buyers with growth potential
- **At-Risk**: used to buy repeatedly, now quiet
- **Recent Low-Spend**: newer or lower-value customers to nurture
- **Hibernating**: mostly low-value, long-inactive customers

Headline result:

- **9.6%** of customers, the Champions, drive **55.8%** of revenue.
- Roughly **25%** of customers drive **80%** of revenue.

### 2. CLV Separates Future Value

Method:

- BG/NBD predicts future purchase count.
- Gamma-Gamma predicts average order value.
- Together they estimate 6-month CLV.

Validation:

- Predicted 6-month CLV has **0.84 correlation** with actual holdout revenue.

Production caveat:

- Portfolio-level CLV is conservative on this seasonal dataset, so a real deployment should add seasonal calibration.

### 3. The Key Story: BTYD Fails for Churn

I tested the classic BTYD churn shortcut:

```text
churn risk = 1 - P(alive)
```

It performed poorly:

- BTYD heuristic AUC: **0.44**
- Logistic regression churn model AUC: **0.81**

Why it failed:

- BG/NBD often gives one-time buyers `P(alive)=1.0` because it has not observed a repeat-purchase gap.
- But in the holdout, one-time buyers are actually the highest-risk group.
- One-time buyer churn rate: about **73.9%**
- Repeat-buyer churn rate: about **36.2%**

This is the main lesson: **a model can be useful for CLV while still being the wrong tool for a specific churn classification task.**

## CRM Action Layer

The demo turns segment + CLV + churn risk into simple actions:

| Segment | Action |
| --- | --- |
| Champions | VIP retention, avoid unnecessary discounting |
| Loyal High-Value | Personalized cross-sell / bundle recommendation |
| At-Risk | Win-back test with discount or personalized email |
| Recent Low-Spend | Onboarding / second-purchase nurture |
| Hibernating | Low-cost automated reactivation |

The ROI section is intentionally framed as a **simulation**, not proven business impact:

- Simulated campaign spend: **£17.7K**
- Simulated net value: **£127.8K**
- Simulated blended ROI: **7.2x**

In production, these assumptions should be replaced by measured uplift from A/B tests.

## How To Talk About This Demo

Short version:

> I built a CRM decision workflow on 805K retail transactions: RFM segmentation, 6-month CLV, churn prediction, and ROI simulation. The key story is model validation: the textbook BTYD `1 - P(alive)` churn heuristic scored only 0.44 AUC because it misread one-time buyers, so I replaced it with a supervised churn classifier and reached 0.81 AUC.

What not to over-explain:

- K-means tuning details
- Gamma-Gamma formulas
- Every segment's ROI table
- The full dashboard architecture

Keep the story simple:

```text
Budget allocation problem
-> RFM / CLV / churn decision chain
-> BTYD churn heuristic fails
-> diagnose one-time buyers
-> supervised churn model fixes it
-> CRM actions + A/B test next step
```

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/data_prep.py
python src/rfm_segments.py
python src/clv_model.py
python src/churn_model.py
python src/segment_strategy.py
streamlit run app/streamlit_app.py
```

## Limits

- Public dataset, not company data.
- ROI is simulated, not measured campaign lift.
- Churn uses transaction behavior only; a production model should add campaign exposure, product category, channel, and engagement features.
- BTYD is not “bad”; it is useful for CLV here, but `1 - P(alive)` is not reliable as a churn classifier for this business label.
