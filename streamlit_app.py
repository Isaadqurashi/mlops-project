import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
from dotenv import load_dotenv

# Load env vars (for local support)
load_dotenv()

# --- Config ---
st.set_page_config(page_title="Stock Prediction System", layout="wide", page_icon="📈")
# MODEL_DIR removed (Dynamic loading now used)

# --- Secrets ---
# Try to get from st.secrets (Cloud) or os.getenv (Local)
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# --- Helper Functions ---
@st.cache_resource
def load_models_local(symbol):
    """Loads models directly from disk for the specific symbol."""
    model_path = f"models/{symbol}"
    models = {}
    try:
        models['regression'] = joblib.load(f"{model_path}/regression_model.pkl")
        models['classification'] = joblib.load(f"{model_path}/classification_model.pkl")
        models['clustering'] = joblib.load(f"{model_path}/clustering_model.pkl")
        return models
    except Exception as e:
        # Fallback to AAPL if specific model missing (for robustness)
        if symbol != "AAPL":
             try:
                 # st.warning(f"Models for {symbol} not found. Using AAPL logic transfer.")
                 return load_models_local("AAPL")
             except:
                 pass
        st.error(f"Failed to load models for {symbol}: {e}")
        return None

from src.orchestration.notifications import notify_discord

def send_discord_notification(symbol, price, change_percent, prediction_dir):
    """Sends a formatted message to Discord using the centralized module."""
    
    emoji = "🚀" if change_percent > 0 else "🔻"
    pred_emoji = "🟢" if "UP" in prediction_dir else "🔴"
    
    # Format the message string
    message = (f"**Stock Update** 🕒\n"
               f"**{symbol}**: ${price:.2f} {emoji} ({change_percent:.2f}%)\n"
               f"**AI Prediction:** {prediction_dir} {pred_emoji}")
    
    # Use the robust notification function
    # It handles checking WEBHOOK_URL and printing errors
    # Use the robust notification function
    # It handles checking WEBHOOK_URL and printing errors
    return notify_discord(message)

@st.cache_data(ttl=3600) # CACHE FOR 1 HOUR
def fetch_live_data(symbol):
    """Fetches raw price data and calculates indicators locally (bypassing API limits)."""
    if not ALPHA_VANTAGE_KEY:
        st.warning("⚠️ ALPHA_VANTAGE_API_KEY not found. Using Mock Data.")
        return get_mock_data(symbol)

    try:
        # Fetch only Daily Price (Free Endpoint)
        ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact') # 100 data points is enough for indicators
        
        # Ensure sorted chronologically
        data = data.sort_index()
        
        # Rename columns standard for calculation
        data.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # --- Local Calculation (Free & Unlimited) ---
        # SMA
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility (20-day std dev of Returns)
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # MACD (12, 26, 9)
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        # signal = macd.ewm(span=9, adjust=False).mean() # We don't use signal for model input
        data['macd'] = macd

        # Get latest valid row
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        change_percent = ((latest['close'] - prev['close']) / prev['close']) * 100

        return {
            "price": float(latest['close']),
            "change": change_percent,
            "sma_20": float(latest['sma_20']),
            "sma_50": float(latest['sma_50']),
            "rsi": float(latest['rsi']),
            "macd": float(latest['macd']),
            "volatility": float(latest['volatility']) if not np.isnan(latest['volatility']) else 0.0,
            "is_mock": False
        }

    except Exception as e:
        # st.warning(f"API Error: {e}. Falling back to mock.")
        # Only show warning if it's not the common "Key Error" on first load
        print(f"Fetch failed: {e}")
        st.warning(f"Could not fetch data for {symbol} (API Limit?). Showing Mock Data.")
        return get_mock_data(symbol)

def get_mock_data(symbol):
    """Generates realistic mock data if API fails or key missing."""
    base_price = {"AAPL": 150, "GOOGL": 2800, "MSFT": 300, "AMZN": 3400, "TSLA": 900, "NVDA": 400}
    price = base_price.get(symbol, 100) + np.random.uniform(-5, 5)
    return {
        "price": price,
        "change": np.random.uniform(-2, 2),
        "sma_20": price * 0.95,
        "sma_50": price * 0.90,
        "rsi": np.random.uniform(30, 70),
        "macd": np.random.uniform(-1, 1),
        "is_mock": True
    }

# --- UI Layout ---
st.title("📈 AI Stock Prediction System")

# Sidebar
st.sidebar.header("Control Panel")
available_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"]
symbol = st.sidebar.selectbox("Select Stock", available_stocks)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear() # Clear cache to force update
    st.rerun()





# --- Main Logic ---

# 1. Fetch Data
with st.spinner(f"Fetching Live Data for {symbol}..."):
    data = fetch_live_data(symbol)

# --- Layout: Tabs ---
tab1, tab2, tab3 = st.tabs(["🚀 Dashboard", "🧠 Deep Dive", "📊 Raw Data"])

# === TAB 1: DASHBOARD ===
with tab1:
    # A. Header Metrics
    col_head1, col_head2, col_head3, col_head4 = st.columns(4)
    with col_head1:
        st.metric("Current Price", f"${data['price']:.2f}", f"{data['change']:.2f}%")
    with col_head2:
        st.metric("RSI (Momentum)", f"{data['rsi']:.1f}", "Overbought" if data['rsi']>70 else "Oversold" if data['rsi']<30 else "Neutral", delta_color="off")
    with col_head3:
        st.metric("Volatility", f"{data.get('volatility', 0):.4f}", help="20-Day Std Dev of Returns")
    with col_head4:
        source = "🔴 Mock" if data['is_mock'] else "🟢 Live"
        st.metric("Data Source", source)

    st.markdown("---")

    # B. AI Prediction Section
    st.subheader(f"🤖 AI Prediction for {symbol}")
    
    features = np.array([[data['sma_20'], data['sma_50'], data['rsi'], data['macd']]])
    models = load_models_local(symbol)
    
    if models:
        col_pred1, col_pred2 = st.columns(2)
        
        # Regression
        pred_price = models['regression'].predict(features)[0]
        
        # Classification
        pred_direction_prob = models['classification'].predict_proba(features)[0]
        direction = "UP 🚀" if pred_direction_prob[1] > 0.5 else "DOWN 🔻"
        confidence = max(pred_direction_prob)
        
        with col_pred1:
            st.info(f"**Predicted Direction:** {direction}")
            st.progress(float(confidence), text=f"Confidence: {confidence*100:.1f}%")
            
        with col_pred2:
            st.success(f"**Target Price (Next Close):** ${pred_price:.2f}")

    # C. Price Chart (Candlestick)
    st.subheader("📉 Price History")
    # Note: fetch_live_data only returns the LAST row's calculated metrics + latest meta, 
    # but for charts we need the full dataframe. 
    # To fix this without breaking the cache, we'll fetch full history purely for charting here.
    # Ideally, fetch_live_data should return the full DF, but let's do a quick fetch for charts:
    try:
        if not data['is_mock'] and ALPHA_VANTAGE_KEY:
            ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
            hist_data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
            hist_data = hist_data.sort_index()
            hist_data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            fig = go.Figure(data=[go.Candlestick(x=hist_data.index,
                            open=hist_data['open'],
                            high=hist_data['high'],
                            low=hist_data['low'],
                            close=hist_data['close'])])
            fig.update_layout(title=f"{symbol} Daily Price", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
             st.warning("Charts unavailable in Mock Data mode (Add API Key to see charts).")
    except Exception as e:
        st.error(f"Could not load chart: {e}")


# === TAB 2: DEEP DIVE (Unsupervised & Technicals) ===
with tab2:
    st.header("🧠 Advanced Analysis")
    
    # Clustering / Market Regime
    if models and 'clustering' in models:
        st.subheader("🧐 Market Regime (Clustering)")
        
        clus_features = np.array([[data.get('volatility', 0), data['rsi']]])
        cluster_id = models['clustering'].predict(clus_features)[0]
        
        regime_labels = {
            0: "Regime 0 (Watch) 👁️",
            1: "Regime 1 (Accumulate) 💰",
            2: "Regime 2 (Risk/Volatile) ⚠️"
        }
        regime_name = regime_labels.get(cluster_id, f"Cluster {cluster_id}")
        
        st.info(f"**Current State:** {regime_name}")
        st.caption("We use K-Means Clustering on Volatility & RSI to identify the market state.")

    st.markdown("---")
    
    # Technical Indicators Chart
    st.subheader("📊 Technical Indicators")
    if not data['is_mock'] and 'hist_data' in locals():
        # Calculate Indicators on history for plotting
        # Simple RSI calculation for plotting
        delta = hist_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist_data['rsi_plot'] = 100 - (100 / (1 + rs))

        fig_rsi = px.line(hist_data, x=hist_data.index, y='rsi_plot', title="Relative Strength Index (14)")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(template="plotly_dark")
        st.plotly_chart(fig_rsi, use_container_width=True)

# === TAB 3: RAW DATA ===
with tab3:
    st.subheader("Raw Data View")
    st.json(data)

# --- Sidebar Notification ---
st.sidebar.markdown("---")
if st.sidebar.button("🔔 Send Discord Update"):
    # Use current data if available, else defaults
    current_price = data.get('price', 0.0)
    current_change = data.get('change', 0.0)
    # If models failed, we won't have 'direction', so we use a placeholder checks
    test_direction = direction if 'direction' in locals() else "N/A"
    
    success, status_msg = send_discord_notification(symbol, current_price, current_change, test_direction)
    if success:
        st.sidebar.success("Sent!")
    else:
        st.sidebar.error(f"Failed: {status_msg}")

# 4. Footer
st.markdown("---")
st.caption("AI Stock Prediction System | Deployed on Hugging Face Spaces")
