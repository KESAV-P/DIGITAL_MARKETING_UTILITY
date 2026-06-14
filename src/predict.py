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

def simulate_budget(pipeline, features, base_spend, new_spend, daily_budget, residuals, forecast_window):
    """
    Simulate revenue for a new spend level by scaling the spend feature.
    Returns bootstrapped p10, p50, p90 predictions.
    """
    # Create copy of features to avoid modifying original
    X = features.copy()
    
    # Overwrite spend and update spend_vs_budget_ratio
    X['spend'] = new_spend
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = new_spend / daily_budget if daily_budget > 0 else 1.0
    X['spend_vs_budget_ratio'] = np.clip(ratio, 0.0, 3.0)
    if pd.isna(X['spend_vs_budget_ratio']) or np.isinf(X['spend_vs_budget_ratio']):
        X['spend_vs_budget_ratio'] = 1.0
        
    # Run point prediction (daily value)
    pred_row = pd.DataFrame([X])
    point_pred_daily = pipeline.predict(pred_row)[0]
    
    # Parametric bootstrap: sample N=1000 residuals
    np.random.seed(42)
    residual_samples = np.random.choice(residuals, size=1000, replace=True)
    
    # Bootstrapped daily samples
    pred_samples_daily = point_pred_daily + residual_samples
    
    # Scale from daily to the forecast window (extrapolation)
    pred_samples_window = pred_samples_daily * forecast_window
    
    # Clip at 0
    pred_samples_window = np.clip(pred_samples_window, 0.0, None)
    
    p10 = np.percentile(pred_samples_window, 10)
    p50 = np.percentile(pred_samples_window, 50)
    p90 = np.percentile(pred_samples_window, 90)
    
    return p10, p50, p90

def predict_roas(pipeline, features, base_spend, new_spend, daily_budget, residuals):
    """
    Predict ROAS using the ROAS pipeline.
    """
    X = features.copy()
    X['spend'] = new_spend
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = new_spend / daily_budget if daily_budget > 0 else 1.0
    X['spend_vs_budget_ratio'] = np.clip(ratio, 0.0, 3.0)
    if pd.isna(X['spend_vs_budget_ratio']) or np.isinf(X['spend_vs_budget_ratio']):
        X['spend_vs_budget_ratio'] = 1.0
        
    pred_row = pd.DataFrame([X])
    point_pred_roas = pipeline.predict(pred_row)[0]
    
    np.random.seed(42)
    residual_samples = np.random.choice(residuals, size=1000, replace=True)
    
    pred_samples_roas = point_pred_roas + residual_samples
    pred_samples_roas = np.clip(pred_samples_roas, 0.0, 50.0)
    
    p10 = np.percentile(pred_samples_roas, 10)
    p50 = np.percentile(pred_samples_roas, 50)
    p90 = np.percentile(pred_samples_roas, 90)
    
    return p10, p50, p90

def main():
    parser = argparse.ArgumentParser(description="Prediction Pipeline for NetElixir-RevForecaster")
    parser.add_argument("--features", default="features.parquet", help="Path to input features parquet")
    parser.add_argument("--model", default="./pickle/model.pkl", help="Path to pickled model bundle")
    parser.add_argument("--output", default="./output/predictions.csv", help="Path to write predictions CSV")
    
    args = parser.parse_args()
    
    # 1. Load model bundle
    if not os.path.exists(args.model):
        print(f"CRITICAL: Pickled model not found at '{args.model}'. Please run src/train.py first.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading model bundle from {args.model}...")
    with open(args.model, 'rb') as f:
        model_bundle = pickle.load(f)
        
    revenue_pipelines = model_bundle['revenue_pipelines']
    roas_pipelines = model_bundle['roas_pipelines']
    feature_columns = model_bundle['feature_columns']
    residuals_revenue = model_bundle['residuals_revenue']
    residuals_roas = model_bundle['residuals_roas']
    global_residuals_rev = model_bundle['global_residuals_revenue']
    global_residuals_roas = model_bundle['global_residuals_roas']
    llm_cache = model_bundle['llm_cache']
    
    # 2. Load features parquet
    if not os.path.exists(args.features):
        print(f"CRITICAL: Features parquet not found at '{args.features}'. Please run src/generate_features.py first.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading features from {args.features}...")
    df = pd.read_parquet(args.features)
    df['date'] = pd.to_datetime(df['date'])
    
    # Identify forecast base date (max date in historical data)
    forecast_base_date = df['date'].max()
    print(f"Forecast base date: {forecast_base_date.strftime('%Y-%m-%d')}")
    
    # Identify global min date from metadata
    global_min_date = pd.to_datetime(model_bundle['training_metadata']['min_date'])
    
    # Reindex features to training columns, fill missing with 0
    # Note: df has metadata columns like date, channel, campaign_type, revenue, roas
    # We will keep these separate but reindex the training columns
    
    # 3. Predict for each group and forecast window
    forecast_windows = [30, 60, 90]
    budget_multipliers = {
        'minus_30pct': 0.7,
        'minus_15pct': 0.85,
        'base': 1.0,
        'plus_15pct': 1.15,
        'plus_30pct': 1.3
    }
    
    predictions_rows = []
    
    # Group the historical features
    groups = df.groupby(['channel', 'campaign_type'])
    
    for (channel, campaign_type), group_df in groups:
        group_key = f"{channel}__{campaign_type}"
        
        # Get historical average spend and daily budget from last 30 days
        group_df_sorted = group_df.sort_values('date')
        last_30d_df = group_df_sorted[group_df_sorted['date'] > forecast_base_date - pd.Timedelta(days=30)]
        if len(last_30d_df) == 0:
            last_30d_df = group_df_sorted.tail(30)
            
        base_spend = last_30d_df['spend'].mean()
        base_daily_budget = last_30d_df['daily_budget'].mean()
        
        # Base feature row: rolling means of the last 30 days of features
        base_feature_row = last_30d_df[feature_columns].mean()
        # Handle case where some dummies might be float means - keep as is, or round if needed
        # We reindex to match training feature_columns (filling missing with 0)
        base_feature_row = base_feature_row.reindex(feature_columns, fill_value=0.0)
        
        # Determine residuals for bootstrap
        grp_res_rev = residuals_revenue.get(group_key, global_residuals_rev)
        grp_res_roas = residuals_roas.get(group_key, global_residuals_roas)
        
        # If group has < 30 residuals, fallback to global residuals pool
        if len(grp_res_rev) < 30:
            grp_res_rev = global_residuals_rev
        if len(grp_res_roas) < 30:
            grp_res_roas = global_residuals_roas
            
        rev_pipeline = revenue_pipelines.get(group_key, revenue_pipelines['__fallback__'])
        roas_pipeline = roas_pipelines.get(group_key, roas_pipelines['__fallback__'])
        
        for window in forecast_windows:
            # Construct midpoint date
            midpoint_date = forecast_base_date + pd.Timedelta(days=window / 2)
            
            # Build synthetic future feature row
            synth_row = base_feature_row.copy()
            synth_row['month'] = midpoint_date.month
            synth_row['week_of_year'] = midpoint_date.isocalendar().week
            synth_row['day_of_week'] = midpoint_date.dayofweek
            synth_row['quarter'] = midpoint_date.quarter
            synth_row['is_q4'] = 1 if midpoint_date.month in [10, 11, 12] else 0
            synth_row['is_november'] = 1 if midpoint_date.month == 11 else 0
            synth_row['is_december'] = 1 if midpoint_date.month == 12 else 0
            synth_row['days_since_start'] = (midpoint_date - global_min_date).days
            
            # Predict each budget scenario
            for scenario_name, multiplier in budget_multipliers.items():
                new_spend = base_spend * multiplier
                
                # Predict revenue p10/p50/p90
                rev_p10, rev_p50, rev_p90 = simulate_budget(
                    rev_pipeline, synth_row, base_spend, new_spend, base_daily_budget, grp_res_rev, window
                )
                
                # Predict ROAS p10/p50/p90
                roas_p10, roas_p50, roas_p90 = predict_roas(
                    roas_pipeline, synth_row, base_spend, new_spend, base_daily_budget, grp_res_roas
                )
                
                # Forecast window total spend input
                spend_input = new_spend * window
                
                # Get causal summary from cache
                causal_key = f"{channel}__{window}d"
                causal_summary = llm_cache.get(causal_key, "Causal summary not available.")
                
                predictions_rows.append({
                    "forecast_period_days": window,
                    "channel": channel,
                    "campaign_type": campaign_type,
                    "revenue_p10": float(rev_p10),
                    "revenue_p50": float(rev_p50),
                    "revenue_p90": float(rev_p90),
                    "roas_p10": float(roas_p10),
                    "roas_p50": float(roas_p50),
                    "roas_p90": float(roas_p90),
                    "spend_input": float(spend_input),
                    "budget_scenario": scenario_name,
                    "causal_summary": causal_summary
                })
                
    # 4. Generate Aggregate Rows
    # Sum revenue and spend across all channels for channel='all' and campaign_type='all'
    print("Generating aggregate rows...")
    predictions_df = pd.DataFrame(predictions_rows)
    
    agg_rows = []
    for window in forecast_windows:
        for scenario_name in budget_multipliers.keys():
            # Filter rows for this window and scenario
            sub_df = predictions_df[
                (predictions_df['forecast_period_days'] == window) &
                (predictions_df['budget_scenario'] == scenario_name)
            ]
            
            # Sum up spend and revenue bounds
            total_spend = sub_df['spend_input'].sum()
            total_rev_p10 = sub_df['revenue_p10'].sum()
            total_rev_p50 = sub_df['revenue_p50'].sum()
            total_rev_p90 = sub_df['revenue_p90'].sum()
            
            # ROAS aggregate: total_revenue / total_spend
            roas_agg_p10 = total_rev_p10 / total_spend if total_spend > 0 else 0.0
            roas_agg_p50 = total_rev_p50 / total_spend if total_spend > 0 else 0.0
            roas_agg_p90 = total_rev_p90 / total_spend if total_spend > 0 else 0.0
            
            # Ensure within logical bounds
            roas_agg_p10 = min(max(roas_agg_p10, 0.0), 50.0)
            roas_agg_p50 = min(max(roas_agg_p50, 0.0), 50.0)
            roas_agg_p90 = min(max(roas_agg_p90, 0.0), 50.0)
            
            # Get aggregate causal summary from cache
            causal_key = f"aggregate__{window}d"
            causal_summary = llm_cache.get(causal_key, "Aggregate causal summary not available.")
            
            agg_rows.append({
                "forecast_period_days": window,
                "channel": "all",
                "campaign_type": "all",
                "revenue_p10": float(total_rev_p10),
                "revenue_p50": float(total_rev_p50),
                "revenue_p90": float(total_rev_p90),
                "roas_p10": float(roas_agg_p10),
                "roas_p50": float(roas_agg_p50),
                "roas_p90": float(roas_agg_p90),
                "spend_input": float(total_spend),
                "budget_scenario": scenario_name,
                "causal_summary": causal_summary
            })
            
    # Combine individual group rows and aggregate rows
    final_predictions_df = pd.concat([predictions_df, pd.DataFrame(agg_rows)], ignore_index=True)
    
    # Sort order: forecast_period_days ASC, channel, campaign_type, budget_scenario
    # Custom channel sort: put 'all' at the end or sort alphabetically
    final_predictions_df = final_predictions_df.sort_values(
        by=['forecast_period_days', 'channel', 'campaign_type', 'budget_scenario']
    ).reset_index(drop=True)
    
    # 5. Write predictions.csv
    print(f"Writing final predictions to {args.output}...")
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    final_predictions_df.to_csv(args.output, index=False)
    print("Prediction pipeline complete!")

if __name__ == "__main__":
    main()
