import os
import sys
import argparse
import random
import pickle
import numpy as np
import pandas as pd

# Set seeds everywhere
random.seed(42)
np.random.seed(42)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed

from llm_insights import generate_causal_summary

# Base training feature columns (excluding campaign type dummies which are dynamic)
BASE_FEATURE_COLUMNS = [
    # Temporal
    'month', 'week_of_year', 'day_of_week', 'quarter', 'is_q4', 'is_november', 'is_december', 'days_since_start',
    # Lags
    'lag_7d_revenue', 'lag_14d_revenue', 'lag_30d_revenue', 'lag_7d_spend', 'lag_30d_spend', 'lag_7d_roas', 'lag_30d_roas',
    # Rolling
    'rolling_7d_revenue_mean', 'rolling_30d_revenue_mean', 'rolling_7d_spend_mean', 'rolling_30d_spend_mean', 'rolling_7d_roas_mean', 'rolling_30d_roas_mean', 'rolling_7d_revenue_std',
    # Budget features
    'spend', 'spend_vs_budget_ratio',
    # Channel dummies
    'channel_google', 'channel_bing', 'channel_meta'
]

def make_pipeline():
    """
    Constructs the GBR Scikit-Learn pipeline.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        ))
    ])

def calculate_stats(df, label="Group"):
    """
    Calculate historical stats for LLM insights and explanation.
    """
    # Sort chronologically
    df_sorted = df.sort_values('date').reset_index(drop=True)
    if len(df_sorted) == 0:
        return {
            "avg_roas_last_30d": 0.0,
            "avg_roas_last_90d": 0.0,
            "total_revenue_last_90d": 0.0,
            "total_spend_last_90d": 0.0,
            "trend_direction": "flat",
            "anomaly_flags": []
        }
        
    last_date = df_sorted['date'].max()
    
    # Filter periods
    df_30d = df_sorted[df_sorted['date'] > last_date - pd.Timedelta(days=30)]
    df_90d = df_sorted[df_sorted['date'] > last_date - pd.Timedelta(days=90)]
    
    # ROAS 30d
    spend_30d = df_30d['spend'].sum()
    rev_30d = df_30d['revenue'].sum()
    roas_30d = rev_30d / spend_30d if spend_30d > 0 else 0.0
    roas_30d = min(max(roas_30d, 0.0), 50.0)
    
    # ROAS 90d
    spend_90d = df_90d['spend'].sum()
    rev_90d = df_90d['revenue'].sum()
    roas_90d = rev_90d / spend_90d if spend_90d > 0 else 0.0
    roas_90d = min(max(roas_90d, 0.0), 50.0)
    
    # Trend direction: Compare last 30d revenue with prior 30d (days 31-60)
    df_prior_30d = df_sorted[
        (df_sorted['date'] <= last_date - pd.Timedelta(days=30)) &
        (df_sorted['date'] > last_date - pd.Timedelta(days=60))
    ]
    rev_prior_30d = df_prior_30d['revenue'].sum()
    
    if rev_30d > rev_prior_30d * 1.05:
        trend = "up"
    elif rev_30d < rev_prior_30d * 0.95:
        trend = "down"
    else:
        trend = "flat"
        
    # Anomaly flags: revenue z-score > 2.5
    anomaly_dates = []
    mean_rev = df_sorted['revenue'].mean()
    std_rev = df_sorted['revenue'].std()
    if std_rev > 0:
        z_scores = (df_sorted['revenue'] - mean_rev) / std_rev
        anom_idx = df_sorted[np.abs(z_scores) > 2.5].index
        anomaly_dates = df_sorted.loc[anom_idx, 'date'].dt.strftime('%Y-%m-%d').tolist()
        
    return {
        "avg_roas_last_30d": roas_30d,
        "avg_roas_last_90d": roas_90d,
        "total_revenue_last_90d": rev_90d,
        "total_spend_last_90d": spend_90d,
        "trend_direction": trend,
        "anomaly_flags": anomaly_dates
    }

def train_group(channel, campaign_type, group_df, feature_columns, fallback_rev_pipeline, fallback_roas_pipeline):
    group_key = f"{channel}__{campaign_type}"
    group_size = len(group_df)
    
    # Calculate group stats
    stats = calculate_stats(group_df, label=group_key)
    
    X_group = group_df[feature_columns]
    y_rev_group = group_df['revenue']
    y_roas_group = group_df['roas']
    
    if group_size >= 60:
        # Dedicated pipeline
        rev_pipe = make_pipeline()
        roas_pipe = make_pipeline()
        
        rev_pipe.fit(X_group, y_rev_group)
        roas_pipe.fit(X_group, y_roas_group)
        
        # Predict and compute in-sample residuals
        pred_rev = rev_pipe.predict(X_group)
        pred_roas = roas_pipe.predict(X_group)
        
        res_rev = (y_rev_group - pred_rev).tolist()
        res_roas = (y_roas_group - pred_roas).tolist()
        is_dedicated = True
    else:
        # Fallback pipeline
        rev_pipe = fallback_rev_pipeline
        roas_pipe = fallback_roas_pipeline
        
        # Compute residuals using fallback pipeline on group's data
        pred_rev = fallback_rev_pipeline.predict(X_group)
        pred_roas = fallback_roas_pipeline.predict(X_group)
        
        res_rev = (y_rev_group - pred_rev).tolist()
        res_roas = (y_roas_group - pred_roas).tolist()
        is_dedicated = False
        
    return group_key, rev_pipe, roas_pipe, res_rev, res_roas, stats, is_dedicated

def main():
    parser = argparse.ArgumentParser(description="Model Training for NetElixir-RevForecaster")
    parser.add_argument("--data-dir", default="./data", help="Directory containing raw campaign CSV files")
    parser.add_argument("--model-out", default="./pickle/model.pkl", help="Output path for pickled model bundle")
    
    args = parser.parse_args()
    
    # Step 1: Run feature engineering to get the feature DataFrame
    # Let's import features.parquet. We assume it is already generated by generate_features.py, or we can check if it exists.
    features_parquet = "features.parquet"
    if not os.path.exists(features_parquet):
        print(f"features.parquet not found. Please run generate_features.py first.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading features from {features_parquet}...")
    df = pd.read_parquet(features_parquet)
    
    # Date parsing check
    df['date'] = pd.to_datetime(df['date'])
    
    # Identify training feature columns (base columns + campaign type dummies)
    camp_type_cols = [col for col in df.columns if col.startswith('campaign_type_')]
    feature_columns = BASE_FEATURE_COLUMNS + camp_type_cols
    print(f"Using {len(feature_columns)} feature columns for training.")
    
    # Step 2 & 3: Train dedicated and fallback pipelines
    revenue_pipelines = {}
    roas_pipelines = {}
    residuals_revenue = {}
    residuals_roas = {}
    group_stats = {}
    
    # Train global fallback pipelines on all data
    print("Training global fallback pipelines...")
    fallback_rev_pipeline = make_pipeline()
    fallback_roas_pipeline = make_pipeline()
    
    X_all = df[feature_columns]
    y_rev_all = df['revenue']
    y_roas_all = df['roas']
    
    fallback_rev_pipeline.fit(X_all, y_rev_all)
    fallback_roas_pipeline.fit(X_all, y_roas_all)
    
    # Calculate residuals for fallback pipeline
    fallback_residuals_rev = (y_rev_all - fallback_rev_pipeline.predict(X_all)).tolist()
    fallback_residuals_roas = (y_roas_all - fallback_roas_pipeline.predict(X_all)).tolist()
    
    # Save fallback under a special key
    revenue_pipelines['__fallback__'] = fallback_rev_pipeline
    roas_pipelines['__fallback__'] = fallback_roas_pipeline
    
    # Identify groups
    groups = df.groupby(['channel', 'campaign_type'])
    print(f"Identified {len(groups)} distinct (channel, campaign_type) groups.")
    
    # Train groups in parallel using joblib
    print("Training group-specific models in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(train_group)(
            channel, campaign_type, group_df, feature_columns, fallback_rev_pipeline, fallback_roas_pipeline
        )
        for (channel, campaign_type), group_df in groups
    )
    
    # Collect results
    for group_key, rev_pipe, roas_pipe, res_rev, res_roas, stats, is_dedicated in results:
        revenue_pipelines[group_key] = rev_pipe
        roas_pipelines[group_key] = roas_pipe
        residuals_revenue[group_key] = res_rev
        residuals_roas[group_key] = res_roas
        group_stats[group_key] = stats
        if is_dedicated:
            print(f"  Trained dedicated pipeline for group '{group_key}'.")
        else:
            print(f"  Assigned fallback pipeline for group '{group_key}' (size < 60).")
            
    # Step 6: Generate LLM Causal Summaries (cached)
    llm_cache = {}
    forecast_windows = [30, 60, 90]
    
    # Gather channel-level datasets for channel summaries
    channels = df['channel'].unique()
    for ch in channels:
        ch_df = df[df['channel'] == ch]
        ch_stats = calculate_stats(ch_df, label=ch)
        campaign_types = ch_df['campaign_type'].unique().tolist()
        
        # Top campaign type by revenue
        top_camp = ch_df.groupby('campaign_type')['revenue'].sum().idxmax()
        
        for window in forecast_windows:
            summary = generate_causal_summary(
                channel=ch,
                campaign_types=campaign_types,
                roas_30d=ch_stats['avg_roas_last_30d'],
                roas_90d=ch_stats['avg_roas_last_90d'],
                trend_direction=ch_stats['trend_direction'],
                top_campaign_type=top_camp,
                total_revenue_90d=ch_stats['total_revenue_last_90d'],
                forecast_window=window,
                anomaly_dates=ch_stats['anomaly_flags'][:5] # Pass up to 5 anomaly dates
            )
            llm_cache[f"{ch}__{window}d"] = summary
            
    # Gather aggregate portfolio stats (all channels summed)
    # Sum daily values across all channels/campaigns to get daily portfolio dataframe
    portfolio_daily = df.groupby('date', as_index=False).agg({
        'revenue': 'sum',
        'spend': 'sum'
    })
    # Add dummy channel to run stats
    portfolio_stats = calculate_stats(portfolio_daily, label="Aggregate")
    all_campaign_types = df['campaign_type'].unique().tolist()
    top_camp_overall = df.groupby('campaign_type')['revenue'].sum().idxmax()
    
    for window in forecast_windows:
        summary_agg = generate_causal_summary(
            channel="all",
            campaign_types=all_campaign_types,
            roas_30d=portfolio_stats['avg_roas_last_30d'],
            roas_90d=portfolio_stats['avg_roas_last_90d'],
            trend_direction=portfolio_stats['trend_direction'],
            top_campaign_type=top_camp_overall,
            total_revenue_90d=portfolio_stats['total_revenue_last_90d'],
            forecast_window=window,
            anomaly_dates=portfolio_stats['anomaly_flags'][:5]
        )
        llm_cache[f"aggregate__{window}d"] = summary_agg
        
    # Step 7: Pickle the entire model bundle
    model_bundle = {
        'revenue_pipelines': revenue_pipelines,
        'roas_pipelines': roas_pipelines,
        'feature_columns': feature_columns,
        'residuals_revenue': residuals_revenue,
        'residuals_roas': residuals_roas,
        'global_residuals_revenue': fallback_residuals_rev,
        'global_residuals_roas': fallback_residuals_roas,
        'llm_cache': llm_cache,
        'group_stats': group_stats,
        'training_metadata': {
            'min_date': df['date'].min().strftime('%Y-%m-%d'),
            'max_date': df['date'].max().strftime('%Y-%m-%d'),
            'num_rows': len(df)
        }
    }
    
    print(f"Saving model bundle to {args.model_out}...")
    out_dir = os.path.dirname(args.model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.model_out, 'wb') as f:
        pickle.dump(model_bundle, f)
        
    print("Training pipeline run complete and pickled model saved!")

if __name__ == "__main__":
    main()
