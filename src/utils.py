import os
import glob
import sys
import pandas as pd
import numpy as np

def load_csv_by_pattern(directory, pattern):
    """
    Search for a CSV file in the directory matching pattern and load it.
    Drops the 'Unnamed: 0' index column if it exists.
    """
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    if not files:
        # Try a case-insensitive search if nothing found
        search_path_lower = os.path.join(directory, pattern.lower())
        files = glob.glob(search_path_lower)
        
    if not files:
        raise FileNotFoundError(f"No file matching pattern '{pattern}' found in '{directory}'")
    
    file_path = files[0]
    try:
        df = pd.read_csv(file_path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        return df, file_path
    except Exception as e:
        print(f"Error loading CSV {file_path}: {str(e)}", file=sys.stderr)
        sys.exit(1)

def detect_channel(df):
    """
    Detects which channel the DataFrame belongs to based on columns.
    Returns 'google', 'bing', or 'meta'.
    """
    cols = set(df.columns)
    if 'metrics_cost_micros' in cols and 'metrics_conversions_value' in cols:
        return 'google'
    elif 'TimePeriod' in cols and 'CampaignType' in cols:
        return 'bing'
    elif 'cpc' in cols and 'cpm' in cols and 'conversion' in cols:
        return 'meta'
    else:
        raise ValueError(f"Could not identify channel. Columns present: {df.columns.tolist()}")

def normalize_google(df):
    """
    Normalize Google Ads campaign data.
    """
    df = df.copy()
    
    # Rename columns
    rename_dict = {
        'segments_date': 'date',
        'metrics_clicks': 'clicks',
        'metrics_conversions': 'conversions',
        'metrics_impressions': 'impressions',
        'metrics_video_views': 'video_views',
        'metrics_conversions_value': 'revenue',
        'campaign_advertising_channel_type': 'campaign_type',
        'campaign_budget_amount': 'daily_budget'
    }
    df = df.rename(columns=rename_dict)
    
    # Drop Unnamed: 0 if it wasn't dropped
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    # Convert segments_date to date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Drop rows where date is NaT
    before_drop = len(df)
    df = df.dropna(subset=['date'])
    dropped = before_drop - len(df)
    if dropped > 0:
        print(f"Google: Dropped {dropped} rows with invalid dates", file=sys.stderr)
        
    # metrics_cost_micros divided by 1,000,000 to get USD spend
    if 'metrics_cost_micros' in df.columns:
        df['spend'] = df['metrics_cost_micros'].astype(float) / 1000000.0
    else:
        df['spend'] = 0.0
        
    df['clicks'] = df['clicks'].fillna(0).astype(int)
    df['impressions'] = df['impressions'].fillna(0).astype(int)
    df['conversions'] = df['conversions'].fillna(0.0).astype(float)
    df['revenue'] = df['revenue'].fillna(0.0).astype(float)
    df['video_views'] = df['video_views'].fillna(0).astype(int)
    df['campaign_id'] = df['campaign_id'].astype(str)
    df['campaign_name'] = df['campaign_name'].fillna("Unknown").astype(str)
    df['campaign_type'] = df['campaign_type'].fillna("SEARCH").astype(str)
    
    # Fill daily_budget NaN with median per campaign_type
    df['daily_budget'] = df['daily_budget'].astype(float)
    group_medians = df.groupby('campaign_type')['daily_budget'].transform('median')
    df['daily_budget'] = df['daily_budget'].fillna(group_medians)
    # Global fallback if still NaN
    df['daily_budget'] = df['daily_budget'].fillna(df['daily_budget'].median()).fillna(0.0)
    
    # Add channel column
    df['channel'] = 'google'
    
    # Compute ROAS: revenue / spend, replace inf with 0, clip at 0-50
    with np.errstate(divide='ignore', invalid='ignore'):
        df['roas'] = np.where(df['spend'] > 0, df['revenue'] / df['spend'], 0.0)
    df['roas'] = np.nan_to_num(df['roas'], nan=0.0, posinf=0.0, neginf=0.0)
    df['roas'] = np.clip(df['roas'], 0.0, 50.0)
    
    # Standard columns subset
    final_cols = [
        'date', 'channel', 'campaign_type', 'campaign_name', 'spend', 'revenue',
        'roas', 'clicks', 'impressions', 'conversions', 'daily_budget', 'campaign_id'
    ]
    return df[final_cols]

def normalize_bing(df):
    """
    Normalize Bing Ads campaign data.
    """
    df = df.copy()
    
    # Rename columns
    rename_dict = {
        'CampaignId': 'campaign_id',
        'TimePeriod': 'date',
        'Revenue': 'revenue',
        'Spend': 'spend',
        'Clicks': 'clicks',
        'Impressions': 'impressions',
        'Conversions': 'conversions',
        'CampaignType': 'campaign_type',
        'DailyBudget': 'daily_budget',
        'CampaignName': 'campaign_name'
    }
    df = df.rename(columns=rename_dict)
    
    # Drop Unnamed: 0 if it wasn't dropped
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Drop rows where date is NaT
    before_drop = len(df)
    df = df.dropna(subset=['date'])
    dropped = before_drop - len(df)
    if dropped > 0:
        print(f"Bing: Dropped {dropped} rows with invalid dates", file=sys.stderr)
        
    df['clicks'] = df['clicks'].fillna(0).astype(int)
    df['impressions'] = df['impressions'].fillna(0).astype(int)
    df['conversions'] = df['conversions'].fillna(0.0).astype(float)
    df['spend'] = df['spend'].fillna(0.0).astype(float)
    df['revenue'] = df['revenue'].fillna(0.0).astype(float)
    df['daily_budget'] = df['daily_budget'].fillna(0.0).astype(float)
    df['campaign_id'] = df['campaign_id'].astype(str)
    df['campaign_name'] = df['campaign_name'].fillna("Unknown").astype(str)
    df['campaign_type'] = df['campaign_type'].fillna("Search").astype(str)
    
    # Add columns
    df['channel'] = 'bing'
    df['video_views'] = 0
    
    # Compute ROAS: revenue / spend, replace inf with 0, clip at 0-50
    with np.errstate(divide='ignore', invalid='ignore'):
        df['roas'] = np.where(df['spend'] > 0, df['revenue'] / df['spend'], 0.0)
    df['roas'] = np.nan_to_num(df['roas'], nan=0.0, posinf=0.0, neginf=0.0)
    df['roas'] = np.clip(df['roas'], 0.0, 50.0)
    
    # Standard columns subset
    final_cols = [
        'date', 'channel', 'campaign_type', 'campaign_name', 'spend', 'revenue',
        'roas', 'clicks', 'impressions', 'conversions', 'daily_budget', 'campaign_id'
    ]
    return df[final_cols]

def normalize_meta(df):
    """
    Normalize Meta Ads campaign data.
    """
    df = df.copy()
    
    # Rename columns
    rename_dict = {
        'date_start': 'date',
        'conversion': 'revenue', # Meta conversion column IS revenue in dollars
        'clicks': 'clicks_raw',
        'impressions': 'impressions_raw'
    }
    df = df.rename(columns=rename_dict)
    
    # Drop Unnamed: 0 if it wasn't dropped
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Drop rows where date is NaT
    before_drop = len(df)
    df = df.dropna(subset=['date'])
    dropped = before_drop - len(df)
    if dropped > 0:
        print(f"Meta: Dropped {dropped} rows with invalid dates", file=sys.stderr)
        
    df['spend'] = df['spend'].fillna(0.0).astype(float)
    df['revenue'] = df['revenue'].fillna(0.0).astype(float)
    
    # conversions: revenue / 100 as proxy count, documented
    df['conversions'] = df['revenue'] / 100.0
    
    df['clicks'] = df['clicks_raw'].fillna(0).astype(float).astype(int)
    df['impressions'] = df['impressions_raw'].fillna(0).astype(float).astype(int)
    df['campaign_id'] = df['campaign_id'].astype(str)
    df['campaign_name'] = df['campaign_name'].fillna("Unknown").astype(str)
    
    # campaign_type inference: Split campaign_name by '_', take first token
    # Mapping: Prospecting -> Prospecting, Remarketing -> Remarketing, Generic -> Generic
    # Fallback to "Unknown"
    def infer_meta_campaign_type(name):
        if pd.isna(name) or name == "":
            return "Unknown"
        tokens = str(name).split('_')
        prefix = tokens[0]
        if prefix in ['Prospecting', 'Remarketing', 'Generic']:
            return prefix
        return "Unknown"
        
    df['campaign_type'] = df['campaign_name'].apply(infer_meta_campaign_type)
    
    # Fill daily_budget NaN with median per campaign_type
    df['daily_budget'] = df['daily_budget'].astype(float)
    group_medians = df.groupby('campaign_type')['daily_budget'].transform('median')
    df['daily_budget'] = df['daily_budget'].fillna(group_medians)
    # Global fallback if still NaN
    df['daily_budget'] = df['daily_budget'].fillna(df['daily_budget'].median()).fillna(0.0)
    
    # Add columns
    df['channel'] = 'meta'
    df['video_views'] = 0
    
    # Compute ROAS: revenue / spend, replace inf with 0, clip at 0-50
    with np.errstate(divide='ignore', invalid='ignore'):
        df['roas'] = np.where(df['spend'] > 0, df['revenue'] / df['spend'], 0.0)
    df['roas'] = np.nan_to_num(df['roas'], nan=0.0, posinf=0.0, neginf=0.0)
    df['roas'] = np.clip(df['roas'], 0.0, 50.0)
    
    # Standard columns subset
    final_cols = [
        'date', 'channel', 'campaign_type', 'campaign_name', 'spend', 'revenue',
        'roas', 'clicks', 'impressions', 'conversions', 'daily_budget', 'campaign_id'
    ]
    return df[final_cols]

def load_and_unify_all(data_dir):
    """
    Search for Google, Bing, and Meta files in data_dir using column structure or filenames,
    normalize each, and return a unified DataFrame.
    """
    search_path = os.path.join(data_dir, "*.csv")
    csv_files = glob.glob(search_path)
    if not csv_files:
        search_path_lower = os.path.join(data_dir, "*.CSV")
        csv_files = glob.glob(search_path_lower)
        
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")
        
    google_df, google_path = None, None
    bing_df, bing_path = None, None
    meta_df, meta_path = None, None
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            # Remove index column if present
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
                
            # Detect channel
            channel = None
            try:
                channel = detect_channel(df)
            except ValueError:
                # Fallback to filename checks
                filename = os.path.basename(file_path).lower()
                if 'google' in filename:
                    channel = 'google'
                elif 'bing' in filename or 'microsoft' in filename:
                    channel = 'bing'
                elif 'meta' in filename or 'facebook' in filename:
                    channel = 'meta'
                    
            if channel == 'google':
                google_df, google_path = df, file_path
            elif channel == 'bing':
                bing_df, bing_path = df, file_path
            elif channel == 'meta':
                meta_df, meta_path = df, file_path
        except Exception as e:
            print(f"Warning: Skipping file {file_path} due to error: {str(e)}", file=sys.stderr)
            
    if google_df is None:
        raise FileNotFoundError(f"Could not identify a Google Ads file in '{data_dir}' by column headers or filename.")
    if bing_df is None:
        raise FileNotFoundError(f"Could not identify a Microsoft/Bing Ads file in '{data_dir}' by column headers or filename.")
    if meta_df is None:
        raise FileNotFoundError(f"Could not identify a Meta Ads file in '{data_dir}' by column headers or filename.")
        
    print(f"Detected Google Ads file: {google_path}")
    print(f"Detected Bing Ads file: {bing_path}")
    print(f"Detected Meta Ads file: {meta_path}")
    
    # Normalize each DataFrame
    google_norm = normalize_google(google_df)
    bing_norm = normalize_bing(bing_df)
    meta_norm = normalize_meta(meta_df)
    
    # Concatenate and sort
    unified_df = pd.concat([google_norm, bing_norm, meta_norm], ignore_index=True)
    
    # Sort order: date ascending, channel, campaign_type, campaign_name
    unified_df = unified_df.sort_values(
        by=['date', 'channel', 'campaign_type', 'campaign_name']
    ).reset_index(drop=True)
    
    return unified_df

