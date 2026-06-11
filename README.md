# RFM Customer Segmentation for Experiment-Ready CRM

This project turns retail transaction history into a customer segmentation workflow that can be used before A/B testing. The core idea is simple: do not treat every customer as the same population before designing a campaign. Use RFM behavior to understand who is active, loyal, high-value, or at risk, then design segment-specific experiments.

## Business Question

Before launching a CRM campaign, which customer groups should receive different treatments, and how should the experiment be structured?

## Data

Source dataset: UCI Online Retail transaction data, also available locally in the original RFM project folder as `rfm.db`.

The local SQLite table used for this analysis:

- Table: `sales_data`
- Date range: `2010-12-01` to `2011-12-09`
- Raw rows: `406,829`
- Distinct customers: `4,372`
- Distinct invoices: `22,190`
- Distinct products: `3,684`

To reproduce locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option A: place the SQLite file at data/raw/rfm.db
python src/build_rfm.py

# Option B: point to an existing local database
RFM_DB_PATH="/absolute/path/to/rfm.db" python src/build_rfm.py
```

## Cleaning Rules

- Remove cancelled invoices where `InvoiceNo` starts with `C`
- Remove rows with `Quantity <= 0`
- Remove rows with `UnitPrice <= 0`
- Keep rows with valid `CustomerID`
- Create `Amount = Quantity * UnitPrice`

After cleaning:

- Valid transaction rows: `397,884`
- Customers: `4,338`
- Revenue: `£8.91M`

## Workflow

1. Build customer-level RFM features:
   - `Recency`: days since last purchase
   - `Frequency`: number of distinct invoices
   - `Monetary`: total spend

2. Score customers:
   - R score is reversed so recent buyers score higher
   - F and M score higher for more frequent and higher-value customers
   - Combine into readable labels such as `555`, `455`, and `111`

3. Cluster customers:
   - Apply `log1p` to RFM features to reduce long-tail skew
   - Standardize features
   - Compare K-means cluster counts using inertia, silhouette score, and business interpretability
   - Use four segments for the portfolio case study

4. Translate segments into CRM strategy:
   - High-value loyal customers: VIP / early access, avoid unnecessary discounts
   - Stable repeat customers: bundle recommendation and cross-sell
   - Recent low-spend customers: second-purchase nudges
   - Dormant customers: reactivation tests with strict holdout

5. Design A/B tests inside each segment:
   - Randomize within segment, not after pooling all customers
   - Define primary KPI, guardrails, and treatment cost
   - Compare lift by segment before scaling rollout

## Key Result

The top-value cluster represented `16.6%` of customers but contributed `64.8%` of revenue. This supports segment-specific campaign design because a global campaign would mix very different customer economics.

## Portfolio Positioning

This is not just a clustering project. It is a business analytics workflow:

RFM segmentation -> customer strategy -> experiment design -> rollout decision.

That framing is useful for Product Data Science, Growth Analytics, Marketing Analytics, and CRM roles.
