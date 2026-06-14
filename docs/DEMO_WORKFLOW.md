# Demo Workflow Guide

This document walks you through the step-by-step operations of the **NetElixir-RevForecaster** utility prototype. It illustrates how the system processes data, generates forecasts, performs spend simulations, and displays AI-driven causal narratives.

---

## 1. Step 1: Data Ingestion & Validation
The prototype is designed to load, clean, and validate marketing campaign datasets from Google Ads, Microsoft (Bing) Ads, and Meta Ads.

### How it works:
1. **File Detection & Upload:** 
   - Open the Streamlit App (`streamlit run app/streamlit_demo.py`).
   - Navigate to the **📂 Data Ingestion & Validation** page.
   - You can upload new `.csv` datasets using the file uploaders or use the default local datasets pre-loaded from the `data/` folder.
2. **Schema Normalization:**
   - The ingestion pipeline automatically matches and standardizes platform-specific columns (e.g., Google's `metrics_cost_micros` is converted to USD, and Meta's `conversion` column is mapped as campaign revenue).
3. **Data Quality Checks:**
   - The app runs live diagnostics, checking for **missing values**, **zero-revenue rows**, and **date gaps** (missing dates in historical sequences).
   - A unified preview table shows the normalized, sorted portfolio rows.

---

## 2. Step 2: Forecast Generation
The core forecasting pipeline estimates expected revenue and blended ROAS over 30, 60, and 90-day horizons.

### How to run it:
1. **Production Command-line Run:**
   Run the entry point script to generate the submission prediction file:
   ```bash
   ./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
   ```
   This generates `predictions.csv` containing all forecast periods, channels, campaign types, and budget scenarios.
2. **Interactive UI View:**
   - Navigate to the **📈 Revenue & ROAS Forecast** page in the Streamlit app.
   - Choose your forecast horizon (**30**, **60**, or **90 days**) using the radio buttons.
   - Select the channels you want to visualize (Google, Bing, Meta, or `all` for the combined portfolio).
3. **Uncertainty Interpretation:**
   - Interactive charts display the **Expected Forecast (p50)** as bars, with **Optimistic (p90)** and **Pessimistic (p10)** ranges represented as error bounds.
   - You can download the complete forecasts as a CSV directly from the button in the UI.

---

## 3. Step 3: Budget Simulation & Optimization
Marketers need to know where to spend their next dollar. The prototype includes a live simulation layer to project revenue across media budget scenarios.

### How to use it:
1. **Adjust Spend Sliders:**
   - Navigate to the **💰 Budget Simulation** page.
   - In the sidebar, adjust the sliders for **Google**, **Meta**, and **Bing** spend multipliers (from `0.5x` to `2.0x` of historical spend).
2. **View Simulated Yield Curves:**
   - The app recalculates the features on-the-fly and runs the trained Gradient Boosting pipelines.
   - It displays a **Yield Curve Plot** showing the expected revenue scaling for each channel.
3. **Analyze Marginal ROAS:**
   - The **Marginal ROAS Table** calculates:
     $$\text{Marginal ROAS} = \frac{\Delta \text{Revenue}}{\Delta \text{Spend}}$$
     This tells you the exact yield you get for increasing budget in a channel. If a channel's Marginal ROAS drops below `1.0x`, it indicates diminishing returns where additional budget costs more than the revenue it returns.

---

## 4. Step 4: AI Causal Insights & Anomaly Timelines
To provide reasoning behind the forecast, the prototype integrates Google Gemini to deliver marketing-focused causal narratives.

### How to view it:
1. **Narrative Generation:**
   - Navigate to the **🤖 AI Causal Insights** page.
   - The page displays a portfolio-wide narrative explaining overall performance trends, risks, and Q4 preparation strategies.
2. **Channel-Specific Deep Dives:**
   - View 2-3 sentence causal summaries tailored specifically to Google, Meta, and Bing Ads.
   - *Note: These summaries are fetched with zero-latency at runtime from the cached LLM insights generated during the offline training step.*
3. **Anomaly & Risk Alerts:**
   - The page lists dates where historical campaign volume anomalies (Revenue Z-Score > 2.5) occurred, signaling seasonality spikes or tracking issues.
   - Critical performance risks (e.g., declining revenue trends or ROAS falling below critical thresholds) are highlighted in alert boxes.
