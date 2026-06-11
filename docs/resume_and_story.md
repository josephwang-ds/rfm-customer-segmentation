# RFM 项目：简历 Bullets + 面试故事（中英文）

Live demo: https://josephwang-rfm-segmentation.streamlit.app/ | GitHub: https://github.com/josephwang-ds/rfm-customer-segmentation

---

## 1. 简历 Bullets / Resume Bullets

### 中文（完整版，3 条）

- 基于 80.5 万条英国电商交易数据（5,878 名客户，£1,774 万收入），用 K-means 构建 5 个 RFM 客户分群，识别出 9.6% 的"冠军客户"贡献 55.8% 的收入（25% 客户贡献 80% 收入）。
- 构建 BG/NBD + Gamma-Gamma 模型预测 6 个月客户终身价值（CLV），与留出期实际收入相关性达 0.84，十分位排序全部正确；流失模型（逻辑回归，AUC 0.81）显著优于教科书式 BTYD P(alive) 基线（AUC 0.44），并定位其失效原因为一次性购买客户（实际流失率 74% vs 复购客户 36%）。
- 将分群、CLV 与流失风险整合为可执行的 CRM 策略：£1.77 万模拟营销投入对应 £12.78 万净增量价值（综合 ROI 7.2 倍），并用功效分析给出每个分群的 A/B 测试样本量要求；以双语 Streamlit 应用交付。

### 中文（精简版，1 条）

- 端到端客户分析项目：80 万条电商交易 → RFM 分群（K-means）→ 6 个月 CLV 预测（BG/NBD + Gamma-Gamma，与实际相关性 0.84）→ 流失模型（AUC 0.81，超越 BTYD 基线 0.44）→ 成本调整 ROI 模拟（7.2x）与 A/B 测试设计；全部用 18 个月校准 / 6 个月留出期验证。

### English (full version, 3 bullets)

- Segmented 805K UK e-commerce transactions (5,878 customers, £17.7M revenue) into 5 RFM segments via K-means, identifying that 9.6% of customers (Champions) drive 55.8% of revenue (25% of customers → 80% of revenue).
- Built a BG/NBD + Gamma-Gamma model predicting 6-month CLV with 0.84 correlation to actual holdout revenue and correct rank-ordering across all 10 deciles; built a churn classifier (logistic regression, AUC 0.81) that beats the textbook BTYD P(alive) baseline (AUC 0.44) and diagnosed why it fails — one-time buyers churn at 74% vs. 36% for repeat buyers.
- Translated segments + CLV + churn risk into an actionable CRM strategy: £17.7K simulated campaign spend → £127.8K net incremental value (7.2x blended ROI), with power-analysis-based A/B test sample sizes per segment; delivered as a bilingual Streamlit app.

### English (compact, 1 bullet)

- End-to-end customer analytics: 805K e-commerce transactions → RFM segmentation (K-means) → 6-month CLV (BG/NBD + Gamma-Gamma, 0.84 corr. to actuals) → churn model (AUC 0.81 vs. 0.44 BTYD baseline) → cost-adjusted ROI simulation (7.2x) with A/B test design; all validated on an 18-month calibration / 6-month holdout split.

---

## 2. 为什么做这个项目 / Why This Project

### 中文（面试口径，约 1 分钟）

> 我做这个项目，是因为我发现两个普遍问题。第一，业务侧：很多 CRM 和增长团队的营销预算是"平均撒"的——所有客户收一样的邮件、一样的优惠券，因为团队回答不了三个问题：谁值得保护？谁值得挽回？这笔钱花得值不值？第二，求职市场侧：大多数作品集里的 RFM 项目止步于聚类——分群图很漂亮，但不产生任何决策。
>
> 所以我刻意把这个项目设计成一条完整的决策链：分群只是起点，每个分群接上 CLV 预测（这群人未来 6 个月值多少钱）、流失模型（谁即将沉默）、最后落到成本调整后的 ROI 模拟和 A/B 测试设计（这笔干预值不值、能不能被验证）。而且我只用交易流水数据，因为这是几乎所有零售商唯一真正拥有的数据——方法可以直接迁移到任何有订单表的业务。
>
> 验证方式我也刻意模拟真实工作场景：18 个月校准期训练、6 个月留出期检验，就像 CRM 团队今天给客户打分、两个季度后回头看预测准不准——而不是用样本内拟合自欺欺人。

### English (interview version, ~1 minute)

> I built this project because of two gaps I kept seeing. On the business side, most CRM/growth teams spread campaign budget evenly — every customer gets the same email and the same coupon — because the team can't answer three questions: who is worth protecting, who is worth winning back, and is the spend even worth it? On the portfolio side, most RFM projects stop at clustering — pretty segments, zero decisions.
>
> So I deliberately designed this as a full decision chain: segmentation is just the starting point. Each segment connects to a CLV prediction (what is this group worth over the next 6 months), a churn model (who is about to go quiet), and finally a cost-adjusted ROI simulation plus A/B test design (is the intervention worth it, and can it be validated). And I used only transaction history, because that's the one dataset virtually every retailer actually has — the method transfers to any business with an orders table.
>
> The validation deliberately mirrors how a real CRM team works: train on an 18-month calibration window, then check predictions against a 6-month holdout — score customers today, see two quarters later whether you were right — instead of fooling yourself with in-sample fit.

---

## 3. 面试故事（STAR）/ Interview Story (STAR)

### 中文

**情境（S）：** 零售商手里只有交易流水，想从"全员群发"转向精准 CRM，但不知道该把预算投给谁。

**任务（T）：** 用两年的英国电商数据（UCI Online Retail II，清洗后 80.5 万条交易、5,878 名客户）构建从分群到预算决策的完整链路，且每一步都要可验证。

**行动（A）：**
1. 清洗数据（剔除缺失 CustomerID、负数量/价格、取消订单），划分 18 个月校准期 + 6 个月留出期。
2. 用校准期计算 RFM 特征，K-means 分成 5 群；发现冠军客户（9.6%）贡献 55.8% 收入。
3. 用 BG/NBD + Gamma-Gamma 预测 6 个月 CLV，留出期验证：相关性 0.84，十分位排序全对。
4. 构建流失模型。这里有个关键发现：教科书推荐的 BTYD "1 − P(alive)" 启发式 AUC 只有 0.44——比随机还差。我深挖原因：BG/NBD 对校准期内零复购的客户一律给 P(alive)=1，但一次性购买客户的实际流失率高达 74%。换成逻辑回归 + 行为特征后 AUC 提升到 0.81。
5. 整合为分群策略：每群的"风险 CLV"= 平均 CLV × 流失概率，配上干预方案、单客成本、功效分析算出的 A/B 样本量，最后做成可调参数的 ROI 模拟器（双语 Streamlit 应用）。

**结果（R）：** £1.77 万模拟投入 → £12.78 万模拟净值（ROI 7.2x）；更重要的是产出一套可直接执行、可被 A/B 测试证伪的 CRM 方案。整套代码、数据管道、双语 demo 公开可复现。

### English

**Situation:** A retailer has only transaction logs and wants to move from one-size-fits-all email blasts to targeted CRM — but doesn't know where the budget should go.

**Task:** Build the full chain from segmentation to budget decision on two years of UK e-commerce data (UCI Online Retail II; 805K clean transactions, 5,878 customers), with every step validated.

**Action:**
1. Cleaned the data (dropped missing CustomerID, negative quantity/price, cancelled invoices); split into an 18-month calibration window and a 6-month holdout.
2. Computed RFM features on calibration data, K-means into 5 segments; found Champions (9.6% of customers) drive 55.8% of revenue.
3. Predicted 6-month CLV with BG/NBD + Gamma-Gamma; holdout validation: 0.84 correlation, correct rank-ordering in all 10 deciles.
4. Built a churn model — with a key finding: the textbook BTYD "1 − P(alive)" heuristic scored AUC 0.44, worse than random. I dug in: BG/NBD assigns P(alive)=1.0 to every customer with zero repeat purchases, yet one-time buyers actually churn at 74%. A logistic regression on behavioral features lifted AUC to 0.81.
5. Combined everything into a segment strategy: CLV-at-risk = avg CLV × churn probability, per-segment interventions, power-analysis A/B sample sizes, and an adjustable ROI simulator (bilingual Streamlit app).

**Result:** £17.7K simulated spend → £127.8K simulated net value (7.2x ROI) — and, more importantly, a CRM plan that is executable and falsifiable via A/B tests. Code, pipeline, and bilingual demo are fully public and reproducible.

---

## 4. 常见追问 Q&A / Likely Follow-up Questions

**Q: 为什么 BTYD 基线比随机还差？/ Why is the BTYD baseline worse than random?**
中: BG/NBD 只能从"复购间隔"学习流失信号，对零复购客户给 P(alive)=1。而一次性客户恰恰是流失率最高的群体（74%）。所以排序方向在最大的一个群体上是反的，AUC 被拉到 0.44。
EN: BG/NBD only learns dropout from repeat-purchase gaps, so it gives one-time buyers P(alive)=1.0 — but they're exactly the highest-churn group (74%). The ranking is inverted for the largest group, dragging AUC to 0.44.

**Q: ROI 是真实的吗？/ Is the ROI real?**
中: 不是实测，是模拟——客户数、CLV、流失率来自模型，留存提升假设是行业典型值。所以我同时给出了每个分群的 A/B 样本量：方案的下一步就是用实验把假设换成实测值。这正是我想展示的态度：不把模拟当结论。
EN: No — it's a simulation. Counts, CLV, and churn rates come from the models; uplift assumptions are typical industry values. That's why I included per-segment A/B sample sizes: the next step is replacing assumptions with measured effects. Knowing the difference is the point.

**Q: 为什么用 K-means 而不是规则打分？/ Why K-means over rule-based scoring?**
中: 规则打分（如 555 分箱）可解释但边界武断；K-means 让数据决定边界。我用轮廓系数选 k，并用业务可解释性检验分群（每群都能命名、策略不同）。两者本质都是工具——重点是分群之后接什么。
EN: Rule-based quintile scoring is interpretable but the boundaries are arbitrary; K-means lets the data set them. I chose k via silhouette and sanity-checked segments for business interpretability (each is nameable with a distinct strategy). Either works — what matters is what you attach downstream.

**Q: 留出期验证发现了什么问题？/ What did holdout validation reveal?**
中: 组合层面 CLV 低估约 16.6%——BG/NBD 在季节性数据上的典型偏差（留出期含圣诞季）。我在报告里明确标注了这个局限，而不是只展示好看的指标。
EN: Portfolio-level CLV under-predicts by ~16.6% — a typical BG/NBD bias on seasonal data (the holdout includes Christmas). I documented this limitation explicitly rather than showing only flattering metrics.

---

## 5. 30 秒电梯版 / 30-Second Pitch

中: 「我用 80 万条电商交易做了一条完整的 CRM 决策链：RFM 分群 → 6 个月 CLV 预测（与实际相关 0.84）→ 流失模型（AUC 0.81，并诊断出教科书 BTYD 方法为何失效）→ ROI 模拟与 A/B 测试设计。所有结论都用 6 个月留出期实际数据验证。一句话：不止分群，而是回答'预算该花在谁身上、值不值'。」

EN: "I built a complete CRM decision chain on 805K e-commerce transactions: RFM segments → 6-month CLV (0.84 correlation to actuals) → churn model (AUC 0.81, including a diagnosis of why the textbook BTYD approach fails) → ROI simulation with A/B test design. Everything validated on a 6-month holdout. In one line: not just segmentation, but 'whose budget, and is it worth it.'"
