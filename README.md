# NetElixir-RevForecaster

An AI-assisted, schema-agnostic marketing performance forecasting system built for NetElixir AIgnition 2026.

## Overview
NetElixir-RevForecaster ingests, normalizes, and trains multi-channel machine learning models (Gradient Boosting Regressors) to produce probabilistic (p10/p50/p90) revenue and ROAS forecasts with live budget simulation scenarios and Gemini-powered causal insights.

## Prerequisites
- **Python Version:** 3.11.x
- **API Keys:** `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is required *only* during the one-time offline training phase to generate cached causal insights. No API keys or internet connection are required at prediction runtime.

## Setup
```bash
# Initialize and activate your virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

## Running the Prediction Pipeline
Run the entry point shell script to generate predictions. It requires no network or interactive inputs.
```bash
# Set execute permission if not already done
chmod +x run.sh

# Run with defaults: ./data, ./pickle/model.pkl, and ./output/predictions.csv
./run.sh

# Or specify custom paths:
./run.sh <data_dir> <model_path> <output_path>
```

## Training (One-time Offline Process)
To retrain the models and regenerate the LLM causal insights cache:
```bash
# Export your Gemini API key (optional, template fallback used if missing)
export GEMINI_API_KEY="your-api-key"

# Generate training features
python src/generate_features.py --data-dir ./data --out features.parquet

# Train models and generate model.pkl
python src/train.py --data-dir ./data --model-out ./pickle/model.pkl
```

## Streamlit Demo UI
To view the interactive data dashboards, dynamic budget simulator, and AI insights:
```bash
streamlit run app/streamlit_demo.py
```

## Output Format (`predictions.csv`)
| Column | Type | Description |
|---|---|---|
| `forecast_period_days` | int | Forecast horizon (30, 60, or 90 days) |
| `channel` | str | Channel identifier (`google`, `bing`, `meta`, or `all` for aggregate) |
| `campaign_type` | str | Grouping campaign type (e.g., `SEARCH`, `Prospecting`, `all`) |
| `revenue_p10` | float | 10th percentile revenue prediction (pessimistic) |
| `revenue_p50` | float | 50th percentile revenue prediction (expected) |
| `revenue_p90` | float | 90th percentile revenue prediction (optimistic) |
| `roas_p10` | float | 10th percentile ROAS prediction |
| `roas_p50` | float | 50th percentile ROAS prediction |
| `roas_p90` | float | 90th percentile ROAS prediction |
| `spend_input` | float | Total marketing budget spend used for this scenario |
| `budget_scenario` | str | Budget level (`base`, `minus_15pct`, `plus_15pct`, `minus_30pct`, `plus_30pct`) |
| `causal_summary` | str | Gemini-generated (or fallback template) causal business insight |

## Team Details
- **Team Name:** pachaikesav
- **Members:** Kesav P & Jeffryn Adaikalaraj A
- **College:** Chennai institute of technology


