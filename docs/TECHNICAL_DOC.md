# Technical Documentation

## 1. Forecasting Methodology
We model campaign performance using **Gradient Boosting Regressors (GBR)** from Scikit-Learn. Instead of training one massive global model or individual models per campaign name (which would fail when campaign names change in unseen test data), we train a dedicated GBR pipeline per `(channel, campaign_type)` group. 

### Why GBR instead of Prophet or ARIMA?
1. **Multivariate Dependencies:** Marketing performance depends heavily on dynamic inputs (e.g., changes in daily budgets, clicks, impressions). Classical univariate methods (ARIMA, Prophet) cannot naturally ingest future spend changes for simulation scenarios.
2. **Schema Agnostic/Robustness:** GBR handles tabular interactions (e.g., how a 15% budget increase affects Search vs. Display) much more flexibly than additive time series models.
3. **Data Availability:** Prophet/ARIMA struggle with short historical sequences. By leveraging group-level fallback models (trained globally across all channels) for smaller groups with < 60 days of data, we achieve extreme robustness.

---

## 2. Feature Engineering
Our pipeline produces a dense parquet file of temporal, lagged, rolling, and categorical features:
- **Temporal Features:** Month, week of year, day of week, quarter, and an integer `days_since_start` representing the baseline trend.
- **Seasonality Dummies:** Specific flags for Q4, November (Black Friday), and December. (Google Ads December revenue is ~4.9x the average non-Q4 month in the dataset).
- **Lag Features (7d, 14d, 30d):** Captures short-term and medium-term memory for spend, revenue, and ROAS.
- **Rolling Features (7d, 30d Mean & 7d SD):** Smooths out daily fluctuations and provides volatility estimates.
- **Budget Interaction:** `spend_vs_budget_ratio = spend / daily_budget` (clipped at 0-3) to capture campaign saturation.

---

## 3. Probabilistic Output
Predictions must express uncertainty. We use a **Parametric Bootstrap of In-Sample Residuals**:
1. During training, we compute the model's residuals ($y_{actual} - y_{pred}$) on the training set.
2. At inference, we generate the point prediction (daily value) using the group's pipeline.
3. We draw $N=1000$ random samples (with replacement) from the saved residuals pool.
4. We add these residuals to our point prediction, clip the resulting distribution at 0 (as revenue cannot be negative), and compute the 10th (p10), 50th (p50), and 90th (p90) percentiles.
5. If a group has fewer than 30 residuals, we fall back to the global portfolio residuals pool to guarantee statistical validity.

---

## 4. Budget Simulation
To simulate changes in marketing budget:
- When a user inputs a budget multiplier (e.g., +15%), we scale the baseline average spend and `daily_budget`.
- We recompute the `spend_vs_budget_ratio` feature dynamically.
- The model runs prediction with the updated spend features.
- We recalculate the **Marginal ROAS** as:
  $$\text{Marginal ROAS} = \frac{\Delta \text{Revenue}}{\Delta \text{Spend}}$$
  This allows marketers to identify the point of diminishing returns where incremental revenue is lower than incremental spend (Marginal ROAS < 1.0).

---

## 5. Data Assumptions
- **Meta Ads 'conversion' Column:** This column represents monetary revenue in USD, not a raw conversion count. Values range up to $26,538.
- **Meta Campaign Type Inference:** Since Meta data lacks a campaign type column, it is inferred by parsing the `campaign_name` prefix before the second underscore (mapping to `Prospecting`, `Remarketing`, or `Generic`).
- **Google Cost micros:** `metrics_cost_micros` is divided by 1,000,000 to convert from micro-currency to dollars.

---

## 6. LLM Integration & Offline Caching
To adhere to the **No Network Call at Runtime** constraint:
- During training (`train.py`), the system detects if `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set.
- If present, it formats channel stats, trend directions, and anomaly dates into a marketing-focused prompt.
- It calls `gemini-2.5-flash` using a direct HTTP POST request (via Python's standard `urllib` to eliminate SDK bloat).
- The generated causal summaries are saved directly inside `model_bundle['llm_cache']`.
- At prediction runtime, `predict.py` reads directly from the cache with zero latency and zero network dependencies.
- If no key is set, the system uses high-quality rule-based templates incorporating Q4 seasonality.

---

## 7. Limitations & Constraints
- **Unseen Campaign Types:** If test data contains entirely new campaign types, they fall back to the global baseline model.
- **Unseen Channels:** The model is structurally built for Google, Bing, and Meta.
- **Distant Forecasts:** Models predicting 90 days out rely on synthetic feature rolls (using the average of the last 30 days of data). Accuracy degrades as the forecast horizon extends beyond 60 days.
