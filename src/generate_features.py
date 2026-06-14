import os
import sys
import argparse
import random
import numpy as np
import pandas as pd

# Set seeds everywhere
random.seed(42)
np.random.seed(42)

from utils import load_and_unify_all

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering for NetElixir-RevForecaster")
    parser.add_argument("--data-dir", required=True, help="Directory containing raw campaign CSV files")
    parser.add_argument("--out", default="features.parquet", help="Path to output feature parquet file")
    
    args = parser.parse_args()
    
    # 1. Ingest unified data
    print(f"Loading and normalizing data from {args.data_dir}...")
    try:
        df = load_and_unify_all(args.data_dir)
    except Exception as e:
        print(f"CRITICAL: Failed to load and unify data: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    # Validation check: minimum rows
    if len(df) < 100:
        print(f"CRITICAL: Unified DataFrame has only {len(df)} rows, expected at least 100 rows", file=sys.stderr)
        sys.exit(1)
        
    # Validation check: date parsing
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        print("CRITICAL: Date column failed to parse as datetime", file=sys.stderr)
        sys.exit(1)
        
    # Validation check: channel presence
    channels_present = df['channel'].unique()
    expected_channels = {'google', 'bing', 'meta'}
    for ch in expected_channels:
        if ch not in channels_present:
            print(f"WARNING: Expected channel '{ch}' is missing from the input data", file=sys.stderr)
            
    # Validation check: revenue positive
    zero_rev_pct = (df['revenue'] <= 0).mean()
    if zero_rev_pct > 0.5:
        print(f"WARNING: More than 50% of revenue rows are <= 0 ({zero_rev_pct:.1%} of rows)", file=sys.stderr)
        
    print(f"Unified data loaded successfully. Shape: {df.shape}")
    
    # 2. Grouping Strategy
    # Group by (date, channel, campaign_type) to aggregate daily campaign metrics
    print("Grouping and aggregating metrics by channel and campaign_type...")
    agg_rules = {
        'spend': 'sum',
        'revenue': 'sum',
        'clicks': 'sum',
        'impressions': 'sum',
        'conversions': 'sum',
        'daily_budget': 'sum'
    }
    
    df_grouped = df.groupby(['date', 'channel', 'campaign_type'], as_index=False).agg(agg_rules)
    
    # Recalculate ROAS: revenue_sum / spend_sum, replace inf with 0, clip at 0-50
    with np.errstate(divide='ignore', invalid='ignore'):
        df_grouped['roas'] = np.where(df_grouped['spend'] > 0, df_grouped['revenue'] / df_grouped['spend'], 0.0)
    df_grouped['roas'] = np.nan_to_num(df_grouped['roas'], nan=0.0, posinf=0.0, neginf=0.0)
    df_grouped['roas'] = np.clip(df_grouped['roas'], 0.0, 50.0)
    
    # Sort chronologically within each group
    df_grouped = df_grouped.sort_values(by=['channel', 'campaign_type', 'date']).reset_index(drop=True)
    
    # 3. Feature Generation
    print("Generating features...")
    
    # Temporal features
    df_grouped['month'] = df_grouped['date'].dt.month
    df_grouped['week_of_year'] = df_grouped['date'].dt.isocalendar().week.astype(int)
    df_grouped['day_of_week'] = df_grouped['date'].dt.dayofweek
    df_grouped['quarter'] = df_grouped['date'].dt.quarter
    
    # Q4 dummies (observation: Q4 has 3-5x revenue spikes)
    df_grouped['is_q4'] = df_grouped['month'].isin([10, 11, 12]).astype(int)
    df_grouped['is_november'] = (df_grouped['month'] == 11).astype(int)
    df_grouped['is_december'] = (df_grouped['month'] == 12).astype(int)
    
    # days_since_start: integer days since the global minimum date
    global_min_date = df_grouped['date'].min()
    df_grouped['days_since_start'] = (df_grouped['date'] - global_min_date).dt.days
    
    # Helper to compute group-wise lags
    def add_lag(df, group_cols, col, lag, fill_value=0.0):
        shifted = df.groupby(group_cols)[col].shift(lag)
        return shifted.fillna(fill_value)
        
    # Lag features
    group_keys = ['channel', 'campaign_type']
    print("Generating lag features...")
    df_grouped['lag_7d_revenue'] = add_lag(df_grouped, group_keys, 'revenue', 7)
    df_grouped['lag_14d_revenue'] = add_lag(df_grouped, group_keys, 'revenue', 14)
    df_grouped['lag_30d_revenue'] = add_lag(df_grouped, group_keys, 'revenue', 30)
    
    df_grouped['lag_7d_spend'] = add_lag(df_grouped, group_keys, 'spend', 7)
    df_grouped['lag_30d_spend'] = add_lag(df_grouped, group_keys, 'spend', 30)
    
    df_grouped['lag_7d_roas'] = add_lag(df_grouped, group_keys, 'roas', 7)
    df_grouped['lag_30d_roas'] = add_lag(df_grouped, group_keys, 'roas', 30)
    
    # Rolling features
    print("Generating rolling window features...")
    
    # Helper for rolling mean
    def get_rolling_mean(df, group_cols, col, window, min_periods=1):
        # transform with lambda to apply rolling within groups
        return df.groupby(group_cols)[col].transform(lambda x: x.rolling(window, min_periods=min_periods).mean())
        
    df_grouped['rolling_7d_revenue_mean'] = get_rolling_mean(df_grouped, group_keys, 'revenue', 7)
    df_grouped['rolling_30d_revenue_mean'] = get_rolling_mean(df_grouped, group_keys, 'revenue', 30)
    
    df_grouped['rolling_7d_spend_mean'] = get_rolling_mean(df_grouped, group_keys, 'spend', 7)
    df_grouped['rolling_30d_spend_mean'] = get_rolling_mean(df_grouped, group_keys, 'spend', 30)
    
    df_grouped['rolling_7d_roas_mean'] = get_rolling_mean(df_grouped, group_keys, 'roas', 7)
    df_grouped['rolling_30d_roas_mean'] = get_rolling_mean(df_grouped, group_keys, 'roas', 30)
    
    # Rolling standard deviation for revenue
    df_grouped['rolling_7d_revenue_std'] = df_grouped.groupby(group_keys)['revenue'].transform(
        lambda x: x.rolling(7, min_periods=2).std()
    ).fillna(0.0)
    
    # Budget features
    # spend_vs_budget_ratio = spend / daily_budget, clip 0-3, fill inf/NaN with 1
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = df_grouped['spend'] / df_grouped['daily_budget']
    df_grouped['spend_vs_budget_ratio'] = np.clip(ratio, 0.0, 3.0)
    df_grouped['spend_vs_budget_ratio'] = df_grouped['spend_vs_budget_ratio'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    # Encodings
    print("Generating channel and campaign_type encodings...")
    df_grouped['channel_encoded'] = df_grouped['channel']
    df_grouped['campaign_type_encoded'] = df_grouped['campaign_type']
    
    # channel get_dummies
    df_grouped = pd.get_dummies(df_grouped, columns=['channel_encoded'], prefix='channel', drop_first=False, dtype=int)
    # Ensure all expected channels are present as columns (even if dummy is 0)
    for ch in ['google', 'bing', 'meta']:
        col_name = f"channel_{ch}"
        if col_name not in df_grouped.columns:
            df_grouped[col_name] = 0
            
    # campaign_type get_dummies (DO NOT drop first, handle unseen gracefully at prediction time)
    df_grouped = pd.get_dummies(df_grouped, columns=['campaign_type_encoded'], prefix='campaign_type', drop_first=False, dtype=int)
    
    # 4. Save Output
    print(f"Writing features to {args.out}...")
    df_grouped.to_parquet(args.out, index=False)
    print("Feature generation complete!")

if __name__ == "__main__":
    main()
