# Segment-Specific Experiment Design & ROI Simulation

This plan combines three signals computed in `src/`:

1. **RFM segment** (`rfm_segments.py`) — who the customer is today
2. **Predicted 6-month CLV** (`clv_model.py`, BG/NBD + Gamma-Gamma) — what they're worth going forward
3. **Churn risk score** (`churn_model.py`, logistic regression, AUC 0.81) — how likely they are to go quiet

Segment x CLV x churn risk together answer the targeting question: *who should get an intervention, how much should it cost, and is it worth it?*

## Segment Risk/Value Profile (calibration period -> 6m holdout)

| Segment | Customers | % of revenue | Avg 6m CLV | Actual churn rate | CLV at risk (6m) |
|---|---:|---:|---:|---:|---:|
| Champions | 477 | 55.8% | £4,201 | 4.4% | £80,340 |
| Loyal High-Value | 920 | 24.5% | £905 | 20.4% | £196,663 |
| At-Risk | 1,390 | 12.3% | £289 | 56.1% | £232,030 |
| Recent Low-Spend | 589 | 4.2% | £585 | 32.4% | £106,146 |
| Hibernating | 1,590 | 3.2% | £174 | 75.8% | £202,142 |

"CLV at risk" = predicted 6m CLV x churn probability, i.e. the expected revenue that disappears if nothing changes. Note that **At-Risk and Hibernating together account for £434K of CLV at risk (53% of the total £817K across all segments)** despite contributing only 15.5% of current revenue — these are the segments where an intervention has the most room to change the outcome, even though each individual customer is worth less than a Champion.

## Recommended Interventions

### Champions — VIP early-access + loyalty perks
- **Goal:** retain and deepen, not win back (churn risk already low at 4.4%).
- **Treatment:** early access to new collections, small loyalty perks (free shipping tier, birthday reward).
- **Control:** standard email cadence.
- **Primary KPI:** 6-month retention rate (target: +2pp on a 95.6% baseline).
- **Guardrails:** margin rate, perk redemption cost.
- **Cost:** £3/customer · **Sample size needed:** 1,262 per arm — segment (477) is too small to power this alone at +2pp; either run as a longer-duration test, accept a larger MDE, or treat as a "no-regret" rollout (low cost, low downside) without a formal test.

### Loyal High-Value — personalized cross-sell / bundle email
- **Goal:** convert repeat-but-not-yet-loyal customers into Champions; reduce 20.4% churn rate.
- **Treatment:** personalized product bundle recommendation based on purchase history.
- **Control:** generic best-seller email.
- **Primary KPI:** 6-month retention rate (target: +5pp on a 79.6% baseline).
- **Guardrails:** AOV, return rate.
- **Cost:** £2/customer · **Sample size needed:** 921 per arm — segment (920) is borderline; combine with a slightly smaller MDE (+3-4pp) or extend the test window to reach power.

### At-Risk — win-back campaign (15% discount + personalized email)
- **Goal:** the single largest CLV-at-risk pool (£232K). These customers had real repeat-purchase histories (median 3 orders) but have gone quiet (median 207 days).
- **Treatment:** 15% discount + "we miss you" email referencing past purchase categories.
- **Control:** no contact (true holdout).
- **Primary KPI:** 6-month retention rate (target: +8pp on a 43.9% baseline).
- **Guardrails:** discount cost vs. margin, repeat discount-seeking behavior in following period.
- **Cost:** £8/customer · **Sample size needed:** 611 per arm — **this segment is large enough (1,390 customers) to run a properly powered A/B test as designed.** Highest priority for a real experiment.

### Recent Low-Spend — onboarding / nurture series
- **Goal:** these are newer or low-engagement customers (median recency 18 days, median frequency 3) — build the repeat-purchase habit before they drift into At-Risk.
- **Treatment:** 3-email onboarding series (welcome, category guide, second-purchase incentive).
- **Control:** standard post-purchase email only.
- **Primary KPI:** 6-month retention rate (target: +6pp on a 67.6% baseline).
- **Guardrails:** unsubscribe rate, discount cost.
- **Cost:** £3/customer · **Sample size needed:** 904 per arm — segment (589) is undersized; run with a smaller MDE or pool with Loyal High-Value's onboarding cohort.

### Hibernating — low-cost automated reactivation
- **Goal:** largest segment by customer count (1,590, 32%) but lowest value per customer (median £233). Churn risk is highest (75.8%), driven mostly by one-time buyers who never returned.
- **Treatment:** single automated reactivation email, no discount (cost-sensitive — low CLV doesn't justify a discount).
- **Control:** no contact.
- **Primary KPI:** 6-month retention rate (target: +4pp on a 24.2% baseline).
- **Guardrails:** unsubscribe/spam complaint rate.
- **Cost:** £1/customer · **Sample size needed:** 1,896 per arm — segment (1,590) is undersized for +4pp; either lower the MDE to ~3pp or run over a longer window given the low per-send cost.

## Cost-Adjusted ROI Simulation (full-segment rollout, 6-month horizon)

| Segment | Campaign cost | Incremental revenue (if MDE hit) | Net value | ROI |
|---|---:|---:|---:|---:|
| Loyal High-Value | £1,840 | £41,645 | £39,805 | 21.6x |
| Champions | £1,431 | £40,074 | £38,643 | 27.0x |
| At-Risk | £11,120 | £32,108 | £20,988 | 1.9x |
| Recent Low-Spend | £1,767 | £20,678 | £18,911 | 10.7x |
| Hibernating | £1,590 | £11,063 | £9,473 | 6.0x |
| **Total** | **£17,748** | **£145,568** | **£127,820** | **7.2x** |

**Incremental revenue = (segment size) x (assumed retention-rate uplift) x (avg predicted 6m CLV).** All uplifts are *assumptions to be tested*, not guarantees — the sample-size column above tells you which of these assumptions can actually be validated with the available customer base, and which need a longer test window, a larger MDE, or segment pooling.

## Experimental Discipline

- Randomize within each RFM segment (treatment effects are expected to differ by segment — see CLV-at-risk table above).
- Keep a true holdout/control group per segment for incrementality, even on "no-regret" rollouts like Champions.
- Pre-register the primary KPI (6-month retention) and guardrails before launch.
- Re-run `src/rfm_segments.py` -> `clv_model.py` -> `churn_model.py` -> `segment_strategy.py` each cycle so segment membership, CLV, and churn risk reflect the latest behavior — segments are not static labels.
