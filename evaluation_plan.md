Here is a final, comprehensive evaluation plan that integrates all data, methods, benchmarks, and metrics into a single, cohesive research design.

---

### **Comprehensive Evaluation Plan: Factor-Neutral Alpha with Conditional Knockoffs**

### 1. Research Objective

To conduct a rigorous, point-in-time, out-of-sample simulation to determine if the proposed strategy, **`S6: Conditional Knockoff + Factor Neutralization (CK+FN)`**, provides a statistically significant and superior risk-adjusted alpha.

The performance of `S6` will be evaluated against:
1.  Simple beta benchmarks (market index, equal weight).
2.  Ablated (simplified) versions of itself to justify each component's complexity (e.g., vanilla knockoff, no neutralization).
3.  Alternative "common-sense" quant strategies (e.g., Lasso, "kitchen sink" composite).

### 2. The "Horse Race": Competing Strategies (S1-S8)

Eight strategies will be run in parallel. All strategies are subject to the same data, execution, and cost assumptions.



[Image of a financial data graph]


| Group | ID | Strategy Name | Signal Selection Method | Portfolio Construction Method |
| :--- | :--- | :--- | :--- | :--- |
| **A. Simple Baselines** | S1 | **Market Index** | None | Buy & Hold Index (e.g., SPY) |
| | S2 | **Equal Weight (EW-LO)** | None | Long-Only, Equal Weight, Monthly Rebalance |
| **B. Ablation: Vanilla Knockoff** | S3 | **VK + Simple L/S** | **Vanilla Knockoff** (VK) (Predicts $Y \sim A$) | Simple Long/Short (Top/Bottom 20%) |
| | S4 | **VK + Factor Neutral (FN)**| **Vanilla Knockoff** (VK) (Predicts $Y \sim A$) | **Factor-Neutral QP Optimizer** |
| **C. Ablation: The Proposal** | S5 | **CK + Simple L/S** | **Conditional Knockoff** (CK) (Predicts $Y \sim A \mid F$) | Simple Long/Short (Top/Bottom 20%) |
| | **S6** | **CK + FN (Hero Strategy)** | **Conditional Knockoff** (CK) (Predicts $Y \sim A \mid F$) | **Factor-Neutral QP Optimizer** |
| **D. Smart Baselines** | S7 | **Lasso + FN** | **Lasso Regression** (Predicts $Y \sim A + F$) | **Factor-Neutral QP Optimizer** |
| | S8 | **Composite + FN (Kitchen Sink)** | **None** (Uses all $A$) | **Factor-Neutral QP Optimizer** |

### 3. Data and Simulation Engine

This section defines the "fuel" (data) and the "rules of the race" (engine).

#### 3.1. Data & Universe
* **Universe:** Components of the **Russell 1000 Index**.
* **Filters (Monthly):** Must be a current Russell 1000 member with:
    * Market Cap > $1 billion.
    * 3-Month Average Daily Volume (ADV) > $10 million.
* **Full Data Period:** 2008-01-01 to 2024-12-31.
* **Backtest Period (OOS):** **2013-01-01 to 2024-12-31**.
* **Data Sources (Professional/Academic Standard):**
    * **Prices/Volume (CRSP):** All security price, dividend, and volume history.
    * **Fundamentals (Compustat):** All point-in-time balance sheet/income statement data (for $A$ and $F$).
    * **Estimates (I/B/E/S):** All point-in-time analyst estimate data (for $A$).
    * **Risk Factors (K. French Library):** Fama-French factor returns (for $F$).

#### 3.2. Data Factor Definitions
* **$Y$ (Target):** 1-Month Forward Excess Return ($R_{i, t+1M} - R_{f, t+1M}$).
* **$F$ (Risk Factors):**
    * **GICS Sectors:** 11 binary dummy variables.
    * **Style Factors:** Market Beta, SMB (Size), HML (Value), RMW (Profitability), CMA (Investment), UMD (Momentum), STR (Short-Term Reversal), VOL (Residual Volatility).
* **$A$ (Alpha Zoo):** A large, diverse set ($p \approx 200+$) of *potential* alpha signals, including variations of:
    * **Value:** E/P, B/P, S/P, FCF/P (and their 1-yr changes).
    * **Momentum:** 3/6/9/12mo Returns, Analyst EPS Revisions, Price vs. SMA.
    * **Quality:** ROE, ROA, Gross Margin, Debt-to-Equity, Accruals.
    * **Sentiment:** Short Interest, ADV Change, RSI.

#### 3.3. Data Preprocessing Pipeline
This pipeline is run *at each rebalance date $t$* using only the cross-sectional data available at $t$:
1.  **Handle Outliers:** Winsorize all factors (in $A$ and $F$) at the 1% and 99% levels.
2.  **Impute Missing:** Fill any remaining NaNs with the cross-sectional median for that factor.
3.  **Normalize:** Convert all factor values to cross-sectional z-scores (mean 0, std. dev. 1).

#### 3.4. Backtest Engine & Execution
* **Simulation Method:** **Rolling Window**.
    * **Training Set:** At any date $t$, use the past 5 years of data ($t-5Y$ to $t$) to run signal selection (Knockoff, Lasso) and estimate parameters.
    * **Test Set:** Hold the resulting portfolio from $t$ to $t+1M$.
    * Slide the window forward one month and repeat.
* **Optimizer (for S4, S6, S7, S8):** All "FN" strategies use an identical Quadratic Program (QP) with the following constraints:
    * **Factor Neutrality:** Portfolio beta = 0 for *every* factor in $F$.
    * **Market Neutrality:** $\sum w_i = 0$ (Dollar-neutral).
    * **Leverage:** $\sum |w_i| \le 1.0$ (100% Long, 100% Short).
    * **Limits:** Max position size $|w_i| \le 2\%$, Max sector deviation $|w_{sector}| \le 5\%$.
* **Transaction Costs:** **10 bps (0.10%)** per-trade (one-way) applied to all strategies' PnL based on their monthly turnover.

---

### 4. Evaluation Metrics & Success Criteria

#### 4.1. Primary Evaluation Scorecard
The *net-of-cost* PnL curves of all 8 strategies will be compared on these metrics:

| Category | Metric | Definition | Purpose |
| :--- | :--- | :--- | :--- |
| **Alpha Quality** | **Sharpe Ratio (Net)** | `Mean(R) / StdDev(R)` | **Primary Metric.** Risk-adjusted return. |
| | **Annualized Alpha (Net)**| `Mean(R) * 12` | Absolute performance. |
| **Risk** | **Volatility** | `StdDev(R) * sqrt(12)` | Price-path risk. |
| | **Max Drawdown** | `Max(Peak - Trough)` | "Pain" index; tail risk. |
| **Stability / Cost** | **Annual Turnover** | `Sum(Trades) / Gross Exposure` | Measures cost and strategy stability. |
| **Diagnostic** | **Realized Betas** | `Regression(PnL ~ F)` | **Sanity Check.** Must be near-zero for all "FN" strategies (S4, S6, S7, S8). |

#### 4.2. Key Research Questions (Hypothesis Testing)
This scorecard will be used to answer the following questions:

1.  **Does Alpha Exist? (S6 vs. Baselines)**
    * **Test:** Does `S6` have a higher Sharpe Ratio than `S1` (Market) and `S2` (Equal Weight)?
    * **Hypothesis:** Yes, a true alpha strategy should deliver superior risk-adjusted returns vs. simple beta.

2.  **Is "Conditional" (CK) the Key? (S6 vs. S4)**
    * **Test:** Does `S6 (CK+FN)` have a higher Sharpe/lower turnover than `S4 (VK+FN)`?
    * **Hypothesis:** Yes. By pre-selecting signals *orthogonal* to $F$, CK provides a "cleaner" alpha vector, leading to a more efficient and stable portfolio.

3.  **Is the "Factor Neutralizer" (FN) Necessary? (S6 vs. S5)**
    * **Test:** Does `S6 (CK+FN)` have a lower volatility and better realized betas than `S5 (CK+Simple L/S)`?
    * **Hypothesis:** Yes. `S5` will fail because even CK-selected signals are not *perfectly* orthogonal. The `FN` optimizer is required to *enforce* zero risk, resulting in a true alpha PnL.

4.  **Is Knockoff (CK) Superior to Lasso? (S6 vs. S7)**
    * **Test:** Does `S6 (CK+FN)` have a better out-of-sample Sharpe Ratio than `S7 (Lasso+FN)`?
    * **Hypothesis:** Yes. Lasso is prone to selecting spurious, overfit factors. The FDR-control of Knockoffs should produce a more robust and stable signal set.

5.  **Is "Signal Selection" Itself Valuable? (S6 vs. S8)**
    * **Test:** Does `S6 (CK+FN)` perform better than `S8 (Kitchen Sink+FN)`?
    * **Hypothesis:** Yes. Feeding all 200+ factors (S8) into the optimizer creates an unstable, high-turnover, and noisy problem. `S6`'s ability to select a *small, proven* set of factors is its key advantage.