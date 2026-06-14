import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pickle

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import simulate_budget, predict_roas
from src.utils import load_and_unify_all, detect_channel, normalize_google, normalize_bing, normalize_meta

# Set page config
st.set_page_config(
    page_title="NetElixir RevForecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Dark/Glassmorphism Feel
st.markdown("""
<style>
    /* Dark theme color overrides */
    .stApp {
        background-color: #0E1117;
        color: #ECEFF4;
    }
    .css-12w0qpk {
        background-color: #1A1C24;
    }
    /* Brand badges */
    .badge-google {
        background-color: #4285F4;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-bing {
        background-color: #00809D;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-meta {
        background-color: #1877F2;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    /* Cards styling */
    .metric-card {
        background-color: #1F232D;
        border: 1px solid #2D3139;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Helper Functions -----------------
@st.cache_resource
def load_model_bundle(model_path="./pickle/model.pkl"):
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_historical_features(features_path="features.parquet"):
    if not os.path.exists(features_path):
        return None
    df = pd.read_parquet(features_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Initialize session state for uploaded data
if 'google_df' not in st.session_state:
    st.session_state['google_df'] = None
if 'bing_df' not in st.session_state:
    st.session_state['bing_df'] = None
if 'meta_df' not in st.session_state:
    st.session_state['meta_df'] = None

# Logo and Sidebar Header
st.sidebar.markdown("# 📈 RevForecaster")
st.sidebar.markdown("*AI-Assisted Ecommerce Forecasting*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["📂 Data Ingestion & Validation", "📈 Revenue & ROAS Forecast", "💰 Budget Simulation", "🤖 AI Causal Insights"]
)

# ----------------- Page 1: Data Ingestion & Validation -----------------
if page == "📂 Data Ingestion & Validation":
    st.title("📂 Data Ingestion & Validation")
    st.write("Upload your channel data files or use the default local datasets for validation and model ingestion.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Google Ads")
        google_file = st.file_uploader("Upload Google Stats CSV", type="csv", key="google")
        if google_file:
            st.session_state['google_df'] = pd.read_csv(google_file)
            st.success("Google Ads file uploaded!")
        else:
            # Try loading default
            try:
                df, path = load_and_unify_all("./data"), "./data/google_ads_campaign_stats.csv"
                st.session_state['google_df'] = pd.read_csv(path)
                st.info("Loaded default google_ads_campaign_stats.csv")
            except Exception:
                st.warning("No Google Ads dataset loaded.")

    with col2:
        st.subheader("Bing Ads")
        bing_file = st.file_uploader("Upload Bing Stats CSV", type="csv", key="bing")
        if bing_file:
            st.session_state['bing_df'] = pd.read_csv(bing_file)
            st.success("Bing Ads file uploaded!")
        else:
            try:
                df, path = load_and_unify_all("./data"), "./data/bing_campaign_stats.csv"
                st.session_state['bing_df'] = pd.read_csv(path)
                st.info("Loaded default bing_campaign_stats.csv")
            except Exception:
                st.warning("No Bing Ads dataset loaded.")

    with col3:
        st.subheader("Meta Ads")
        meta_file = st.file_uploader("Upload Meta Stats CSV", type="csv", key="meta")
        if meta_file:
            st.session_state['meta_df'] = pd.read_csv(meta_file)
            st.success("Meta Ads file uploaded!")
        else:
            try:
                df, path = load_and_unify_all("./data"), "./data/meta_ads_campaign_stats.csv"
                st.session_state['meta_df'] = pd.read_csv(path)
                st.info("Loaded default meta_ads_campaign_stats.csv")
            except Exception:
                st.warning("No Meta Ads dataset loaded.")

    st.divider()

    # Process and Normalize
    g_raw = st.session_state['google_df']
    b_raw = st.session_state['bing_df']
    m_raw = st.session_state['meta_df']

    if g_raw is not None and b_raw is not None and m_raw is not None:
        try:
            # Perform normalization checks and display badges
            g_norm = normalize_google(g_raw)
            b_norm = normalize_bing(b_raw)
            m_norm = normalize_meta(m_raw)
            
            unified_df = pd.concat([g_norm, b_norm, m_norm], ignore_index=True)
            unified_df = unified_df.sort_values(by=['date', 'channel', 'campaign_type']).reset_index(drop=True)
            
            st.subheader("Unified Campaign Portfolio Status")
            
            # Status cards
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown("""
                <div class="metric-card">
                    <h4><span class="badge-google">Google</span> Status: <span style='color:#00C853'>🟢 Healthy</span></h4>
                    <p><b>Records:</b> {}</p>
                    <p><b>Spend (USD):</b> ${:,.2f}</p>
                    <p><b>Revenue:</b> ${:,.2f}</p>
                </div>
                """.format(len(g_norm), g_norm['spend'].sum(), g_norm['revenue'].sum()), unsafe_allow_html=True)
                
            with sc2:
                st.markdown("""
                <div class="metric-card">
                    <h4><span class="badge-bing">Bing</span> Status: <span style='color:#00C853'>🟢 Healthy</span></h4>
                    <p><b>Records:</b> {}</p>
                    <p><b>Spend (USD):</b> ${:,.2f}</p>
                    <p><b>Revenue:</b> ${:,.2f}</p>
                </div>
                """.format(len(b_norm), b_norm['spend'].sum(), b_norm['revenue'].sum()), unsafe_allow_html=True)
                
            with sc3:
                st.markdown("""
                <div class="metric-card">
                    <h4><span class="badge-meta">Meta</span> Status: <span style='color:#FFB300'>🟡 Warning</span></h4>
                    <p><b>Records:</b> {}</p>
                    <p><b>Spend (USD):</b> ${:,.2f}</p>
                    <p><b>Revenue:</b> ${:,.2f}</p>
                    <small><i>Note: 'conversion' column is treated as monetary revenue value. Proxy conversions set as revenue/100.</i></small>
                </div>
                """.format(len(m_norm), m_norm['spend'].sum(), m_norm['revenue'].sum()), unsafe_allow_html=True)

            st.write("### Portfolio Preview")
            st.dataframe(unified_df.head(20), use_container_width=True)

            # Data Quality Diagnostics
            st.write("### Data Quality Flags")
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                missing_vals = unified_df.isna().sum().sum()
                st.metric("Missing Values", missing_vals, help="Should be 0 after imputation")
                zero_rev_pct = (unified_df['revenue'] <= 0).mean()
                st.metric("Zero Revenue Row Ratio", f"{zero_rev_pct:.1%}", help="Warning if >50%")
            with q_col2:
                # Find date gaps
                min_date = unified_df['date'].min()
                max_date = unified_df['date'].max()
                expected_days = (max_date - min_date).days + 1
                actual_days = unified_df['date'].nunique()
                st.metric("Date Gaps Detected", expected_days - actual_days, help="Expected vs actual distinct dates")
                st.metric("Total Active Campaigns", unified_df['campaign_id'].nunique())
                
        except Exception as e:
            st.error(f"Failed to unify datasets: {str(e)}")
    else:
        st.warning("Please ensure all three channel datasets are loaded.")

# ----------------- Page 2: Revenue & ROAS Forecast -----------------
elif page == "📈 Revenue & ROAS Forecast":
    st.title("📈 Revenue & ROAS Forecast")
    
    # Load model bundle
    bundle = load_model_bundle()
    if bundle is None:
        st.error("Model bundle not found. Please run `train.py` or make sure `./pickle/model.pkl` is present.")
    else:
        st.sidebar.success("Model loaded successfully")
        
        # Load predictions.csv
        pred_path = "./output/predictions.csv"
        if not os.path.exists(pred_path):
            st.warning("No cached prediction file found at output/predictions.csv. Showing live predictions using the model...")
            # Run prediction logic to display
            # To show in UI, we can read the pickled cache or re-run predict logic on default features.parquet
        
        # We can dynamically construct predictions from the model & features
        feat_df = load_historical_features()
        if feat_df is None:
            st.error("Features file 'features.parquet' not found.")
        else:
            # Let's read the CSV file if it exists, otherwise generate on-the-fly
            if os.path.exists(pred_path):
                preds_df = pd.read_csv(pred_path)
            else:
                st.info("Generating predictions...")
                # Fallback to empty df or build it dynamically (could block UI if slow, but let's read the file)
                preds_df = pd.DataFrame()
            
            if not preds_df.empty:
                # Select window
                window = st.radio("Forecast Horizon", [30, 60, 90], horizontal=True)
                
                # Channels filter
                channels = st.multiselect("Channels", ["google", "bing", "meta", "all"], default=["google", "bing", "meta"])
                
                # Filter rows
                filtered_df = preds_df[
                    (preds_df['forecast_period_days'] == window) &
                    (preds_df['channel'].isin(channels)) &
                    (preds_df['budget_scenario'] == 'base')
                ]
                
                # Group by channel and campaign_type for display
                st.write(f"### Point Forecasts & Uncertainty Range ({window}-Day Horizon)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Revenue Forecast ($)")
                    fig_rev = go.Figure()
                    for idx, row in filtered_df.iterrows():
                        label = f"{row['channel'].upper()} - {row['campaign_type']}"
                        fig_rev.add_trace(go.Bar(
                            name=label,
                            x=[label],
                            y=[row['revenue_p50']],
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                array=[row['revenue_p90'] - row['revenue_p50']],
                                arrayminus=[row['revenue_p50'] - row['revenue_p10']]
                            ),
                            marker_color='#4285F4' if row['channel'] == 'google' else '#00809D' if row['channel'] == 'bing' else '#1877F2' if row['channel'] == 'meta' else '#00C853'
                        ))
                    fig_rev.update_layout(showlegend=False, template="plotly_dark", height=400)
                    st.plotly_chart(fig_rev, use_container_width=True)
                    
                with col2:
                    st.write("#### ROAS Forecast (x)")
                    fig_roas = go.Figure()
                    for idx, row in filtered_df.iterrows():
                        label = f"{row['channel'].upper()} - {row['campaign_type']}"
                        fig_roas.add_trace(go.Bar(
                            name=label,
                            x=[label],
                            y=[row['roas_p50']],
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                array=[row['roas_p90'] - row['roas_p50']],
                                arrayminus=[row['roas_p50'] - row['roas_p10']]
                            ),
                            marker_color='#FF6D00'
                        ))
                    fig_roas.update_layout(showlegend=False, template="plotly_dark", height=400)
                    st.plotly_chart(fig_roas, use_container_width=True)

                st.write("### Detailed Forecast Summary")
                st.dataframe(
                    filtered_df[['forecast_period_days', 'channel', 'campaign_type', 'spend_input', 'revenue_p10', 'revenue_p50', 'revenue_p90', 'roas_p10', 'roas_p50', 'roas_p90']],
                    use_container_width=True
                )
                
                # Download button
                csv = preds_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

# ----------------- Page 3: Budget Simulation -----------------
elif page == "💰 Budget Simulation":
    st.title("💰 Budget Simulation")
    
    bundle = load_model_bundle()
    feat_df = load_historical_features()
    
    if bundle is None or feat_df is None:
        st.error("Model bundle or Features parquet missing. Please check your training output.")
    else:
        st.write("Modify daily spend budgets using the sliders below to simulate the portfolio yield and calculate marginal ROAS.")
        
        # Get historical spend & daily budget for sliders
        forecast_base_date = feat_df['date'].max()
        groups = feat_df.groupby(['channel', 'campaign_type'])
        
        # User sliders
        st.sidebar.write("### Adjust Multipliers")
        multipliers = {}
        
        # Google
        multipliers['google'] = st.sidebar.slider("Google Spend Multiplier", 0.5, 2.0, 1.0, 0.05)
        # Meta
        multipliers['meta'] = st.sidebar.slider("Meta Spend Multiplier", 0.5, 2.0, 1.0, 0.05)
        # Bing
        multipliers['bing'] = st.sidebar.slider("Bing Spend Multiplier", 0.5, 2.0, 1.0, 0.05)
        
        st.sidebar.divider()
        window = st.sidebar.radio("Simulation Horizon", [30, 60, 90], index=1)
        
        # Dynamic predictions calculation
        simulated_rows = []
        for (channel, campaign_type), group_df in groups:
            group_key = f"{channel}__{campaign_type}"
            group_df_sorted = group_df.sort_values('date')
            last_30d_df = group_df_sorted[group_df_sorted['date'] > forecast_base_date - pd.Timedelta(days=30)]
            if len(last_30d_df) == 0:
                last_30d_df = group_df_sorted.tail(30)
                
            base_spend = last_30d_df['spend'].mean()
            base_daily_budget = last_30d_df['daily_budget'].mean()
            
            # Base feature row
            base_feature_row = last_30d_df[bundle['feature_columns']].mean()
            base_feature_row = base_feature_row.reindex(bundle['feature_columns'], fill_value=0.0)
            
            # Determine residuals
            grp_res_rev = bundle['residuals_revenue'].get(group_key, bundle['global_residuals_revenue'])
            grp_res_roas = bundle['residuals_roas'].get(group_key, bundle['global_residuals_roas'])
            
            if len(grp_res_rev) < 30:
                grp_res_rev = bundle['global_residuals_revenue']
            if len(grp_res_roas) < 30:
                grp_res_roas = bundle['global_residuals_roas']
                
            rev_pipeline = bundle['revenue_pipelines'].get(group_key, bundle['revenue_pipelines']['__fallback__'])
            roas_pipeline = bundle['roas_pipelines'].get(group_key, bundle['roas_pipelines']['__fallback__'])
            
            # Scale features forward
            midpoint_date = forecast_base_date + pd.Timedelta(days=window / 2)
            synth_row = base_feature_row.copy()
            synth_row['month'] = midpoint_date.month
            synth_row['week_of_year'] = midpoint_date.isocalendar().week
            synth_row['day_of_week'] = midpoint_date.dayofweek
            synth_row['quarter'] = midpoint_date.quarter
            synth_row['is_q4'] = 1 if midpoint_date.month in [10, 11, 12] else 0
            synth_row['is_november'] = 1 if midpoint_date.month == 11 else 0
            synth_row['is_december'] = 1 if midpoint_date.month == 12 else 0
            synth_row['days_since_start'] = (midpoint_date - pd.to_datetime(bundle['training_metadata']['min_date'])).days
            
            # Apply slider multiplier
            mult = multipliers[channel]
            new_spend = base_spend * mult
            
            # Predict revenue
            rev_p10, rev_p50, rev_p90 = simulate_budget(
                rev_pipeline, synth_row, base_spend, new_spend, base_daily_budget, grp_res_rev, window
            )
            
            # Predict ROAS
            roas_p10, roas_p50, roas_p90 = predict_roas(
                roas_pipeline, synth_row, base_spend, new_spend, base_daily_budget, grp_res_roas
            )
            
            simulated_rows.append({
                "channel": channel,
                "campaign_type": campaign_type,
                "spend_input": new_spend * window,
                "revenue_p10": rev_p10,
                "revenue_p50": rev_p50,
                "revenue_p90": rev_p90,
                "roas_p50": roas_p50
            })
            
        sim_df = pd.DataFrame(simulated_rows)
        
        # Display simulated metrics
        tot_spend = sim_df['spend_input'].sum()
        tot_rev_p50 = sim_df['revenue_p50'].sum()
        tot_rev_p10 = sim_df['revenue_p10'].sum()
        tot_rev_p90 = sim_df['revenue_p90'].sum()
        tot_roas = tot_rev_p50 / tot_spend if tot_spend > 0 else 0.0
        
        st.subheader("Simulated Portfolio Summary")
        sm_c1, sm_c2, sm_c3 = st.columns(3)
        with sm_c1:
            st.metric("Total Simulated Spend", f"${tot_spend:,.2f}")
        with sm_c2:
            st.metric("Simulated Revenue (p50)", f"${tot_rev_p50:,.2f}", 
                      delta=f"{tot_rev_p10:,.0f} (p10) to {tot_rev_p90:,.0f} (p90)", delta_color="off")
        with sm_c3:
            st.metric("Overall Simulated ROAS", f"{tot_roas:.2f}x")
            
        st.divider()
        
        # Build curve for visualization: Vary multiplier from 0.5 to 2.0
        st.write("### Spend Optimization & Yield Curves")
        curve_data = []
        mult_range = np.linspace(0.5, 2.0, 10)
        
        for ch_key in ['google', 'meta', 'bing']:
            for m in mult_range:
                temp_rev = 0
                temp_spend = 0
                # Predict for this channel under multiplier m
                for (channel, campaign_type), group_df in groups:
                    if channel != ch_key:
                        continue
                    group_key = f"{channel}__{campaign_type}"
                    group_df_sorted = group_df.sort_values('date')
                    last_30d_df = group_df_sorted[group_df_sorted['date'] > forecast_base_date - pd.Timedelta(days=30)]
                    if len(last_30d_df) == 0:
                        last_30d_df = group_df_sorted.tail(30)
                    base_spend = last_30d_df['spend'].mean()
                    base_daily_budget = last_30d_df['daily_budget'].mean()
                    
                    base_feature_row = last_30d_df[bundle['feature_columns']].mean()
                    base_feature_row = base_feature_row.reindex(bundle['feature_columns'], fill_value=0.0)
                    grp_res_rev = bundle['residuals_revenue'].get(group_key, bundle['global_residuals_revenue'])
                    if len(grp_res_rev) < 30:
                        grp_res_rev = bundle['global_residuals_revenue']
                    rev_pipeline = bundle['revenue_pipelines'].get(group_key, bundle['revenue_pipelines']['__fallback__'])
                    
                    midpoint_date = forecast_base_date + pd.Timedelta(days=window / 2)
                    synth_row = base_feature_row.copy()
                    synth_row['month'] = midpoint_date.month
                    synth_row['week_of_year'] = midpoint_date.isocalendar().week
                    synth_row['day_of_week'] = midpoint_date.dayofweek
                    synth_row['quarter'] = midpoint_date.quarter
                    synth_row['is_q4'] = 1 if midpoint_date.month in [10, 11, 12] else 0
                    synth_row['is_november'] = 1 if midpoint_date.month == 11 else 0
                    synth_row['is_december'] = 1 if midpoint_date.month == 12 else 0
                    synth_row['days_since_start'] = (midpoint_date - pd.to_datetime(bundle['training_metadata']['min_date'])).days
                    
                    r_p10, r_p50, r_p90 = simulate_budget(
                        rev_pipeline, synth_row, base_spend, base_spend * m, base_daily_budget, grp_res_rev, window
                    )
                    temp_rev += r_p50
                    temp_spend += base_spend * m * window
                    
                if temp_spend > 0:
                    curve_data.append({
                        "channel": ch_key,
                        "multiplier": m,
                        "spend": temp_spend,
                        "revenue_p50": temp_rev
                    })
                    
        curve_df = pd.DataFrame(curve_data)
        
        # Plot curves
        fig_curve = px.line(
            curve_df, x="spend", y="revenue_p50", color="channel",
            title="Yield Curve (Spend vs. Revenue)",
            color_discrete_map={"google": "#4285F4", "meta": "#1877F2", "bing": "#00809D"},
            labels={"spend": "Spend ($)", "revenue_p50": "Projected Revenue ($)"}
        )
        fig_curve.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_curve, use_container_width=True)

        # Calculate Marginal ROAS: (Rev_1.15 - Rev_1.0) / (Spend_1.15 - Spend_1.0)
        st.write("### Marginal ROAS Insights")
        marginal_insights = []
        for ch in ['google', 'meta', 'bing']:
            ch_curve = curve_df[curve_df['channel'] == ch]
            if len(ch_curve) >= 2:
                # Interpolate or extract near 1.0 and 1.15
                row_base = ch_curve.iloc[(ch_curve['multiplier'] - 1.0).abs().argsort()[:1]].iloc[0]
                row_high = ch_curve.iloc[(ch_curve['multiplier'] - 1.15).abs().argsort()[:1]].iloc[0]
                
                dS = row_high['spend'] - row_base['spend']
                dR = row_high['revenue_p50'] - row_base['revenue_p50']
                m_roas = dR / dS if dS > 0 else 0.0
                
                marginal_insights.append({
                    "channel": ch.upper(),
                    "base_spend": row_base['spend'],
                    "incremental_spend": dS,
                    "incremental_revenue": dR,
                    "marginal_roas": m_roas
                })
                
        m_df = pd.DataFrame(marginal_insights)
        st.table(m_df)
        
        st.markdown("""
        > [!TIP]
        > **Marginal ROAS explanation:** Channels with a marginal ROAS > 1.0 will yield incremental profit for every extra dollar spent. Shift budget to channels with the highest marginal ROAS first.
        """)

# ----------------- Page 4: AI Causal Insights -----------------
elif page == "🤖 AI Causal Insights":
    st.title("🤖 AI Causal Insights")
    
    bundle = load_model_bundle()
    if bundle is None:
        st.error("Model bundle not found. Please train model first.")
    else:
        llm_cache = bundle.get('llm_cache', {})
        group_stats = bundle.get('group_stats', {})
        
        st.write("### Portfolio Aggregate Causal Narrative")
        st.info(llm_cache.get("aggregate__60d", "Aggregate narrative not generated in model cache."))
        
        st.divider()
        
        st.write("### Channel Causal Deep Dives")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### <span class='badge-google'>Google Ads</span> Insights", unsafe_allow_html=True)
            st.write(llm_cache.get("google__60d", "Google summary not available in cache."))
            
        with col2:
            st.markdown("#### <span class='badge-meta'>Meta Ads</span> Insights", unsafe_allow_html=True)
            st.write(llm_cache.get("meta__60d", "Meta summary not available in cache."))
            
        with col3:
            st.markdown("#### <span class='badge-bing'>Bing Ads</span> Insights", unsafe_allow_html=True)
            st.write(llm_cache.get("bing__60d", "Bing summary not available in cache."))
            
        st.divider()
        
        # Anomaly Timeline and Risk Indicators
        st.write("### Anomalies & Performance Risk Indicators")
        
        # Gather all anomaly flags from group_stats
        anomaly_rows = []
        for g_key, stats in group_stats.items():
            for d in stats.get('anomaly_flags', []):
                anomaly_rows.append({
                    "group": g_key,
                    "date": d,
                    "issue": "Revenue Z-Score > 2.5 (Volume Anomaly)"
                })
                
        if anomaly_rows:
            st.dataframe(pd.DataFrame(anomaly_rows), use_container_width=True)
        else:
            st.success("No volume anomalies detected in historical campaign periods.")
            
        # Risk indicators
        st.write("#### Performance Risk Flags")
        has_risks = False
        for g_key, stats in group_stats.items():
            if stats.get('trend_direction') == 'down':
                st.warning(f"⚠️ **{g_key}** shows a declining trend in revenue over the last 30 days compared to the prior 30 days.")
                has_risks = True
            if stats.get('avg_roas_last_30d', 0.0) < 1.5:
                st.error(f"🚨 **{g_key}** average ROAS is extremely low ({stats.get('avg_roas_last_30d', 0.0):.2f}x). Immediate audit recommended.")
                has_risks = True
                
        if not has_risks:
            st.success("No campaign-level risk flags are active.")
