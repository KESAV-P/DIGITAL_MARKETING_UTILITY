import os
import sys
import logging
import json
import urllib.request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_insights")

# Fallback template for channel-specific summaries
FALLBACK_TEMPLATE = (
    "Based on historical data, {channel} shows a {trend_direction} revenue trend with "
    "an average ROAS of {roas_30d:.2f}x over the past 30 days. Monitor {top_campaign_type} "
    "campaigns closely as they represent the highest revenue contribution."
)

# Fallback template for aggregate summaries
FALLBACK_AGGREGATE_TEMPLATE = (
    "Across all marketing channels, performance shows an overall {trend_direction} trend "
    "with a combined average ROAS of {roas_30d:.2f}x over the last 30 days. Q4 historically "
    "drives 3-5x revenue — August spend decisions will affect Holiday performance. Strategic "
    "budget allocation across Google and Meta is recommended to maximize overall yield."
)

def generate_fallback_summary(channel, roas_30d, trend_direction, top_campaign_type):
    """
    Generate a template-based fallback summary when API key is missing or call fails.
    """
    if channel.lower() in ['all', 'aggregate']:
        return FALLBACK_AGGREGATE_TEMPLATE.format(
            trend_direction=trend_direction,
            roas_30d=roas_30d
        )
    else:
        return FALLBACK_TEMPLATE.format(
            channel=channel.capitalize(),
            trend_direction=trend_direction,
            roas_30d=roas_30d,
            top_campaign_type=top_campaign_type
        )

def generate_causal_summary(
    channel,
    campaign_types,
    roas_30d,
    roas_90d,
    trend_direction,
    top_campaign_type,
    total_revenue_90d,
    forecast_window,
    anomaly_dates=None
):
    """
    Generate causal summary using Google Gemini API.
    If the API key is not set or the call fails, it falls back to a template-based text.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning(f"Neither GEMINI_API_KEY nor GOOGLE_API_KEY found in environment. Generating fallback summary for {channel}.")
        return generate_fallback_summary(channel, roas_30d, trend_direction, top_campaign_type)
        
    campaign_types_str = ", ".join(campaign_types) if isinstance(campaign_types, list) else str(campaign_types)
    anomalies_str = ", ".join(anomaly_dates) if anomaly_dates else "None detected"
    
    system_prompt = (
        "You are a senior digital marketing analyst specializing in e-commerce performance forecasting. "
        "Be specific, concise, and actionable. Never use generic phrases."
    )
    
    user_prompt = (
        f"Channel: {channel}\n"
        f"Campaign types: {campaign_types_str}\n"
        f"Avg ROAS last 30 days: {roas_30d:.2f}x\n"
        f"Avg ROAS last 90 days: {roas_90d:.2f}x\n"
        f"Revenue trend: {trend_direction}\n"
        f"Top campaign type by revenue: {top_campaign_type}\n"
        f"Total revenue last 90 days: ${total_revenue_90d:,.0f}\n"
        f"Anomaly dates: {anomalies_str}\n"
        f"Forecast window: {forecast_window} days\n\n"
        f"Provide a 2-3 sentence business insight explaining the performance trend and key risks or opportunities "
        f"for the next {forecast_window} days. Be specific to this channel's data."
    )
    
    if channel.lower() in ['all', 'aggregate']:
        user_prompt = (
            f"Channel: Combined Portfolio (All Channels)\n"
            f"Campaign types represented: {campaign_types_str}\n"
            f"Combined Avg ROAS last 30 days: {roas_30d:.2f}x\n"
            f"Combined Avg ROAS last 90 days: {roas_90d:.2f}x\n"
            f"Portfolio Revenue trend: {trend_direction}\n"
            f"Top campaign type overall: {top_campaign_type}\n"
            f"Total portfolio revenue last 90 days: ${total_revenue_90d:,.0f}\n"
            f"Anomaly dates: {anomalies_str}\n"
            f"Forecast window: {forecast_window} days\n\n"
            f"Provide a 3-4 sentence overall business narrative explaining the portfolio-wide performance, "
            f"risks, and Q4 preparation opportunities for the next {forecast_window} days."
        )
        
    try:
        # Construct the REST API call to Gemini API using urllib
        # Model: gemini-2.5-flash
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        
        # Format payload
        payload = {
            "contents": [{
                "parts": [{"text": user_prompt}]
            }],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 500
            }
        }
        
        data = json.dumps(payload).encode('utf-8')
        
        logger.info(f"Calling Gemini API for channel {channel}, window {forecast_window}d...")
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            
        # Extract text from response
        # Structure: res_data['candidates'][0]['content']['parts'][0]['text']
        summary = res_data['candidates'][0]['content']['parts'][0]['text'].strip()
        logger.info(f"Successfully generated summary from Gemini for {channel} ({forecast_window}d)")
        return summary
        
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}. Falling back to template-based summary.")
        return generate_fallback_summary(channel, roas_30d, trend_direction, top_campaign_type)

