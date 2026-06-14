# System Architecture

The following diagram illustrates the flow of data during both the **Training Phase** (one-time, offline, network calls allowed) and the **Prediction Phase** (production pipeline, self-contained, no network calls).

```mermaid
flowchart TD
    subgraph Data Prep [1. Data Ingest & Unification]
        A[Google Ads CSV] -->|Normalize cost & columns| D[Unified DataFrame]
        B[Bing Ads CSV] -->|Normalize columns| D
        C[Meta Ads CSV] -->|Infer campaign type & revenue| D
        D -->|Feature Engineering| E[(features.parquet)]
    end

    subgraph Offline Train [2. Offline Training Phase]
        E -->|Read Group Features| F[Train Loop]
        F -->|Fit Dedicated Models| G[GBR Per Group]
        F -->|Fit Global Baseline| H[GBR Fallback Model]
        D -->|Extract Campaign Metrics| I[Calculate Stats]
        I -->|GEMINI_API_KEY set?| J{API Key Check}
        J -->|Yes| K[Call Gemini API REST]
        J -->|No| L[Template-based Fallback]
        K --> M[Causal Insight Cache]
        L --> M
        G & H & M --> N[Serialize to model.pkl]
    end

    subgraph Runtime Predict [3. Production Prediction Pipeline]
        O[(features.parquet)] -->|Reindex training columns| P[Predictor]
        N -->|Unpickle model_bundle| P
        P -->|Group-level Predict| Q[Point Predictions]
        N -->|Residuals Pool| R[Bootstrap Simulation]
        Q & R -->|Sample 1000 Times| S[Probabilistic p10/p50/p90]
        S -->|Cross-Channel Rollup| T[Aggregate Rows]
        T -->|Combine with Causal Cache| U[(predictions.csv)]
    end

    subgraph UI App [4. Interactive Presentation]
        U --> V[Streamlit Dashboard]
        N -->|Live Budget Multipliers| W[Interactive Simulator]
        O --> W
        W -->|Real-time GBR inference| V
    end
```

## Workflow Execution Summary
1. **Unification:** `load_and_unify_all` maps varying columns into a unified scheme (date, channel, campaign_type, spend, revenue, clicks, impressions, conversions, daily_budget).
2. **Features:** `generate_features.py` builds time variables, rolling window values, and holiday flags.
3. **Training:** `train.py` runs once, fits $N$ models, runs API calls to build caches, and saves everything to `model.pkl`.
4. **Prediction:** `predict.py` executes without internet. It uses bootstrap residuals to add probabilistic range bounds.
5. **App:** `streamlit_demo.py` loads the cached model and outputs, permitting users to toggle budget sliders to dynamically see forecasted shifts.
