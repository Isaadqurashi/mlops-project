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
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf
import hashlib

# Load env vars (for local support)
load_dotenv()

# --- Config ---
st.set_page_config(
    page_title="Nuqta | AI Market Insight", 
    layout="wide", 
    page_icon="🔷", 
    initial_sidebar_state="collapsed"
)

# --- Start Background Monitor for Discord Notifications ---
# This runs in a separate thread and checks all tickers every 3 minutes
if 'monitor_started' not in st.session_state:
    try:
        from src.orchestration.monitor import start_monitor
        start_monitor()
        st.session_state.monitor_started = True
    except Exception as e:
        print(f"⚠️  Could not start background monitor: {e}")
        st.session_state.monitor_started = False

# --- Custom CSS Styling with Modern Islamic FinTech Theme ---
def inject_custom_css():
    """Injects comprehensive custom CSS for Nuqta - Modern Islamic FinTech aesthetic."""
    css = """
    <style>
    /* Import Google Fonts - Outfit for headings, Inter for body */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Import RemixIcon */
    @import url('https://cdn.jsdelivr.net/npm/remixicon@3.5.0/fonts/remixicon.css');
    
    /* Global Theme - Deep Slate Background */
    .stApp {
        background: #1A202C;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Remove default padding and margins */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Title Styling - Nuqta Brand */
    h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(135deg, #D4AF37 0%, #10B981 50%, #D4AF37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
        text-align: center;
        letter-spacing: -0.02em;
        text-shadow: 0 0 40px rgba(212, 175, 55, 0.4);
    }
    
    /* Subheader Styling */
    h2, h3 {
        color: #E5E7EB;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Hide Sidebar Completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Top Navigation Bar - Horizontal Ticker Selection */
    .nuqta-navbar {
        background: linear-gradient(135deg, rgba(26, 32, 44, 0.98) 0%, rgba(30, 41, 59, 0.98) 100%);
        backdrop-filter: blur(20px);
        border-bottom: 2px solid rgba(212, 175, 55, 0.3);
        padding: 1.2rem 2.5rem;
        margin: -1rem -2rem 2rem -2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 2rem;
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(212, 175, 55, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .nuqta-navbar::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(212, 175, 55, 0.6) 25%, 
            rgba(16, 185, 129, 0.6) 50%, 
            rgba(212, 175, 55, 0.6) 75%, 
            transparent 100%);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .nuqta-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.5rem;
        color: #D4AF37;
        text-decoration: none;
    }
    
    .nuqta-nav-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .nuqta-nav-label {
        color: #D4AF37;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        white-space: nowrap;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom Selectbox in Navbar */
    .nuqta-navbar [data-testid="stSelectbox"] {
        min-width: 200px;
    }
    
    .nuqta-navbar [data-testid="stSelectbox"] label {
        display: none;
    }
    
    .nuqta-navbar [data-baseweb="select"] {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
        border-radius: 10px !important;
        color: #D4AF37 !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Custom Button Styling - Gold Accent */
    .stButton > button {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
        color: #D4AF37;
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        white-space: nowrap;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.25) 0%, rgba(16, 185, 129, 0.25) 100%);
        border-color: rgba(212, 175, 55, 0.5);
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.3);
        transform: translateY(-2px);
    }
    
    /* Tabs Styling - Gold & Emerald */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 10px 10px 0 0;
        padding: 0.75rem 1.5rem;
        color: #9CA3AF;
        font-weight: 500;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(16, 185, 129, 0.2) 100%);
        color: #D4AF37;
        border-color: rgba(212, 175, 55, 0.4);
        box-shadow: 0 0 15px rgba(212, 175, 55, 0.2);
    }
    
    /* Metric Card Container - Floating Tiles/Dots Design */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.95) 0%, rgba(26, 32, 44, 0.95) 100%);
        border: 2px solid;
        border-image: linear-gradient(135deg, rgba(212, 175, 55, 0.4), rgba(16, 185, 129, 0.4)) 1;
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(15px);
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(212, 175, 55, 0.1),
            0 0 0 1px rgba(212, 175, 55, 0.15);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.1), transparent);
        transition: left 0.6s ease;
    }
    
    .metric-card:hover {
        border-image: linear-gradient(135deg, rgba(212, 175, 55, 0.6), rgba(16, 185, 129, 0.6)) 1;
        box-shadow: 
            0 15px 50px rgba(212, 175, 55, 0.4),
            inset 0 1px 0 rgba(212, 175, 55, 0.2),
            0 0 0 1px rgba(212, 175, 55, 0.3);
        transform: translateY(-6px) scale(1.02);
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-label {
        color: #9CA3AF;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-value {
        color: #D4AF37;
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        margin: 0.5rem 0;
    }
    
    .metric-delta {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        font-weight: 600;
        padding: 0.5rem 1rem;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 8px;
        border-left: 3px solid;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-delta.positive {
        border-left-color: #10B981;
        color: #10B981;
    }
    
    .metric-delta.negative {
        border-left-color: #EF4444;
        color: #EF4444;
    }
    
    .metric-delta.neutral {
        border-left-color: #9CA3AF;
        color: #9CA3AF;
    }
    
    /* AI Prediction Hero Section - Gold & Emerald */
    .prediction-capsule {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, rgba(16, 185, 129, 0.2) 100%);
        border: 2px solid rgba(212, 175, 55, 0.4);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        backdrop-filter: blur(15px);
        box-shadow: 0 10px 50px rgba(212, 175, 55, 0.3), inset 0 0 80px rgba(16, 185, 129, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-capsule::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(212, 175, 55, 0.15) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.1); }
    }
    
    .prediction-direction {
        font-size: 2.8rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #D4AF37 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1rem 0;
        text-align: center;
    }
    
    .confidence-bar-container {
        background: rgba(26, 32, 44, 0.6);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    .confidence-label {
        color: #9CA3AF;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'Outfit', sans-serif;
    }
    
    .confidence-bar {
        height: 28px;
        background: linear-gradient(90deg, rgba(212, 175, 55, 0.2) 0%, rgba(16, 185, 129, 0.2) 100%);
        border-radius: 14px;
        overflow: hidden;
        position: relative;
        border: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #D4AF37 0%, #10B981 100%);
        border-radius: 14px;
        transition: width 1s ease-out;
        box-shadow: 0 0 25px rgba(212, 175, 55, 0.5);
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 0.75rem;
        color: #1A202C;
        font-weight: 700;
        font-size: 0.9rem;
        font-family: 'Inter', sans-serif;
    }
    
    .target-price {
        font-size: 2.5rem;
        font-weight: 800;
        color: #10B981;
        font-family: 'Outfit', sans-serif;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Chart Container */
    .plotly-container {
        background: rgba(26, 32, 44, 0.4);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(212, 175, 55, 0.15);
        margin: 1.5rem 0;
    }
    
    /* Info/Success/Warning Boxes */
    .stAlert {
        background-color: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 10px;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(212, 175, 55, 0.3) 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Icon Styling */
    .nuqta-icon {
        font-size: 1.2rem;
        color: #D4AF37;
        margin-right: 0.5rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #D4AF37;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Custom UI Components ---
def render_metric_card(label, value, delta=None, delta_label=None, color="#D4AF37"):
    """Renders a custom metric card with glassmorphism effect - Floating Tile Design."""
    delta_class = "neutral"
    arrow_symbol = ""
    delta_text = ""
    
    if delta is not None:
        if delta > 0:
            delta_class = "positive"
            arrow_symbol = "↑"
        elif delta < 0:
            delta_class = "negative"
            arrow_symbol = "↓"
    
    if delta is not None and delta_label:
        delta_text = f'<div class="metric-delta {delta_class}"><span style="font-size: 1.2rem; margin-right: 0.3rem;">{arrow_symbol}</span><span>{delta_label}</span></div>'
    elif delta is not None:
        delta_text = f'<div class="metric-delta {delta_class}"><span style="font-size: 1.2rem; margin-right: 0.3rem;">{arrow_symbol}</span><span>{delta:+.2f}%</span></div>'
    
    html = f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="color: {color};">{value}</div>{delta_text}</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_prediction_hero(direction, confidence, target_price, symbol):
    """Renders the hero AI Prediction section with Gold & Emerald gradient."""
    direction_color = "#10B981" if "UP" in direction else "#EF4444"
    direction_emoji = "📈" if "UP" in direction else "📉"
    
    html = f'''
    <div class="prediction-capsule">
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h3 style="color: #9CA3AF; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; font-family: 'Outfit', sans-serif;">
                🧠 AI Prediction for {symbol}
            </h3>
            <div class="prediction-direction">{direction} {direction_emoji}</div>
        </div>
        <div class="confidence-bar-container">
            <div class="confidence-label">📊 Model Confidence</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence * 100}%;">
                    {confidence * 100:.1f}%
                </div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 1.5rem;">
            <div style="color: #9CA3AF; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.1em; font-family: 'Outfit', sans-serif;">
                🎯 Target Price (Next Close)
            </div>
            <div class="target-price">${target_price:.2f}</div>
        </div>
    </div>
    '''
    return st.markdown(html, unsafe_allow_html=True)

# --- Secrets ---
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

# --- Ticker Name Mapping ---
TICKER_NAMES = {
    # USA
    "AAPL": "Apple (AAPL)",
    "GOOGL": "Alphabet (GOOGL)",
    "MSFT": "Microsoft (MSFT)",
    "NVDA": "NVIDIA (NVDA)",
    "TSLA": "Tesla (TSLA)",
    "AMZN": "Amazon (AMZN)",
    # Pakistan
    "OGDC.KA": "OGDC (OGDC.KA)",
    "LUCK.KA": "Lucky Cement (LUCK.KA)",
    "TRG.KA": "TRG Pakistan (TRG.KA)",
    "ENGRO.KA": "Engro (ENGRO.KA)",
    "SYS.KA": "Systems Limited (SYS.KA)",
    # India
    "RELIANCE.NS": "Reliance Industries (RELIANCE.NS)",
    "TCS.NS": "Tata Consultancy Services (TCS.NS)",
    "HDFCBANK.NS": "HDFC Bank (HDFCBANK.NS)",
    "INFY.NS": "Infosys (INFY.NS)",
    # UK
    "RR.L": "Rolls-Royce (RR.L)",
    "AZN.L": "AstraZeneca (AZN.L)",
    "HSBA.L": "HSBC (HSBA.L)",
    "BP.L": "BP (BP.L)",
    # Japan
    "7203.T": "Toyota (7203.T)",
    "6758.T": "Sony (6758.T)",
    "9984.T": "SoftBank Group (9984.T)",
    # Hong Kong
    "0700.HK": "Tencent (0700.HK)",
    "9988.HK": "Alibaba (9988.HK)",
    "1810.HK": "Xiaomi (1810.HK)",
    # Germany
    "SAP.DE": "SAP (SAP.DE)",
    "SIE.DE": "Siemens (SIE.DE)",
    "VOW3.DE": "Volkswagen (VOW3.DE)",
}

def get_available_stocks():
    """Dynamically scans models directory for available trained models."""
    available = []
    models_dir = "models"
    
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            symbol_dir = os.path.join(models_dir, item)
            if os.path.isdir(symbol_dir):
                reg_path = os.path.join(symbol_dir, "regression_model.pkl")
                if os.path.exists(reg_path):
                    available.append(item)
    
    available.sort()
    return available if available else ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"]

def get_friendly_name(ticker: str) -> str:
    """Returns friendly display name for a ticker."""
    return TICKER_NAMES.get(ticker, ticker)

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
        if symbol != "AAPL":
             try:
                 return load_models_local("AAPL")
             except:
                 pass
        st.error(f"Failed to load models for {symbol}: {e}")
        return None

from src.orchestration.notifications import notify_discord

def send_discord_alert(symbol, price, change_percent, prediction_dir, target_price, prediction_change_pct):
    """Sends a professionally formatted message to Discord using the centralized module."""
    # Calculate price change direction and magnitude
    is_bullish = prediction_change_pct >= 0.5
    is_bearish = prediction_change_pct <= -0.5
    
    # Professional emojis and formatting
    direction_emoji = "📈" if is_bullish else "📉"
    trend_emoji = "🟢" if is_bullish else "🔴"
    arrow_emoji = "⬆️" if is_bullish else "⬇️"
    
    # Format friendly name
    friendly_name = get_friendly_name(symbol)
    
    # Professional message format
    message = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**🔷 NUQTA | AI MARKET INSIGHT**\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"**📊 Ticker:** `{symbol}`\n"
        f"**🏢 Company:** {friendly_name}\n\n"
        f"**💰 Current Price:** `${price:.2f}`\n"
        f"**{direction_emoji} Prediction:** {prediction_dir} {trend_emoji}\n"
        f"**🎯 Target Price:** `${target_price:.2f}`\n"
        f"**{arrow_emoji} Expected Change:** `{prediction_change_pct:+.2f}%`\n\n"
        f"**📈 Current Movement:** {change_percent:+.2f}%\n"
        f"**⏰ Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    
    return notify_discord(message)

def get_prediction_id(symbol, date_str, predicted_price):
    """Creates a unique hash ID for a prediction to prevent duplicate alerts."""
    unique_string = f"{symbol}_{date_str}_{predicted_price:.2f}"
    return hashlib.md5(unique_string.encode()).hexdigest()

@st.cache_data(ttl=3600)
def fetch_live_data(symbol):
    """Fetches raw price data and calculates indicators locally.
    Uses Yahoo Finance for international tickers, Alpha Vantage for US stocks.
    """
    has_international_suffix = any(symbol.endswith(suffix) for suffix in ['.KA', '.NS', '.L', '.T', '.HK', '.DE'])
    
    if has_international_suffix or not ALPHA_VANTAGE_KEY:
        return fetch_live_data_yahoo(symbol)
    else:
        try:
            return fetch_live_data_alphavantage(symbol)
        except Exception as e:
            print(f"Alpha Vantage failed for {symbol}: {e}. Trying Yahoo Finance...")
            return fetch_live_data_yahoo(symbol)

def fetch_live_data_yahoo(symbol):
    """Fetches data using Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        data.columns = [col.lower() for col in data.columns]
        data = data[['open', 'high', 'low', 'close', 'volume']]
        data = data.sort_index()
        
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        change_percent = ((latest['close'] - prev['close']) / prev['close']) * 100
        
        return {
            "price": float(latest['close']),
            "change": change_percent,
            "sma_20": float(latest['sma_20']) if not np.isnan(latest['sma_20']) else float(latest['close']),
            "sma_50": float(latest['sma_50']) if not np.isnan(latest['sma_50']) else float(latest['close']),
            "rsi": float(latest['rsi']) if not np.isnan(latest['rsi']) else 50.0,
            "macd": float(latest['macd']) if not np.isnan(latest['macd']) else 0.0,
            "volatility": float(latest['volatility']) if not np.isnan(latest['volatility']) else 0.0,
            "is_mock": False
        }
    except Exception as e:
        print(f"Yahoo Finance fetch failed for {symbol}: {e}")
        st.warning(f"Could not fetch data for {symbol}. Showing Mock Data.")
        return get_mock_data(symbol)

def fetch_live_data_alphavantage(symbol):
    """Fetches data using Alpha Vantage API."""
    if not ALPHA_VANTAGE_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY not found")
    
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        
        data = data.sort_index()
        data.columns = ['open', 'high', 'low', 'close', 'volume']
        
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        
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
        raise Exception(f"Alpha Vantage fetch failed: {e}")

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
        "volatility": np.random.uniform(0.01, 0.03),
        "is_mock": True
    }

# --- Initialize Session State for Discord Alerts ---
if 'last_sent_alert' not in st.session_state:
    st.session_state.last_sent_alert = None

# --- Inject Custom CSS ---
inject_custom_css()

# --- Top Navigation Bar ---
st.markdown("""
<div class="nuqta-navbar">
    <div class="nuqta-brand">
        <span style="font-size: 1.8rem;">🔷</span>
        <span>Nuqta</span>
    </div>
    <div class="nuqta-nav-item">
        <div class="nuqta-nav-label">📊 Select Ticker</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Controls
col_nav1, col_nav2, col_nav3 = st.columns([3, 1, 1])
available_stocks = get_available_stocks()
stock_options = {get_friendly_name(ticker): ticker for ticker in available_stocks}
display_names = list(stock_options.keys())

with col_nav1:
    selected_display = st.selectbox("", display_names, label_visibility="collapsed", key="stock_selector")
    symbol = stock_options[selected_display]

with col_nav2:
    refresh_clicked = st.button("🔄 Refresh", key="refresh_btn", use_container_width=True)
    if refresh_clicked:
        st.cache_data.clear()
        st.rerun()

# Main Title
st.markdown("<br>", unsafe_allow_html=True)
st.title("Nuqta | AI Market Insight")

# --- Main Logic ---

# 1. Fetch Data
with st.spinner(f"Fetching Live Data for {symbol}..."):
    data = fetch_live_data(symbol)

# 2. Load Models
features = np.array([[data['sma_20'], data['sma_50'], data['rsi'], data['macd']]])
models = load_models_local(symbol)

# 3. Calculate Predictions
direction = "N/A"
confidence = 0.0
pred_price = 0.0
prediction_change_pct = 0.0
if models:
    current_price = data.get('price', 0.0)
    pred_price = models['regression'].predict(features)[0]
    pred_direction_prob = models['classification'].predict_proba(features)[0]
    direction = "UP" if pred_direction_prob[1] > 0.5 else "DOWN"
    confidence = max(pred_direction_prob)
    
    # Calculate prediction change percentage
    if current_price > 0:
        prediction_change_pct = ((pred_price - current_price) / current_price) * 100
    
    # Auto-send Discord Alert only if significant move (>= 0.5% or <= -0.5%)
    if abs(prediction_change_pct) >= 0.5:
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_prediction_id = get_prediction_id(symbol, current_date, pred_price)
        
        if st.session_state.last_sent_alert != current_prediction_id:
            # Significant prediction - send alert
            success, status_msg = send_discord_alert(
                symbol, 
                current_price, 
                data.get('change', 0.0), 
                direction,
                pred_price,
                prediction_change_pct
            )
            if success:
                st.session_state.last_sent_alert = current_prediction_id
                st.toast("🔔 Discord Alert Sent!", icon="✅")

# --- Layout: Tabs with Icons ---
tab1, tab2 = st.tabs([
    '📊 Dashboard', 
    '🧠 Deep Dive'
])

# === TAB 1: DASHBOARD ===
with tab1:
    # Header Metrics - Custom Floating Tile Cards
    col_head1, col_head2, col_head3, col_head4 = st.columns(4)
    
    with col_head1:
        render_metric_card(
            '💰 Current Price',
            f"${data['price']:.2f}",
            delta=data['change'],
            color="#D4AF37"
        )
    
    with col_head2:
        rsi_status = "Overbought" if data['rsi'] > 70 else "Oversold" if data['rsi'] < 30 else "Neutral"
        rsi_color = "#EF4444" if data['rsi'] > 70 else "#10B981" if data['rsi'] < 30 else "#9CA3AF"
        render_metric_card(
            '📊 RSI (Momentum)',
            f"{data['rsi']:.1f}",
            delta_label=rsi_status,
            color=rsi_color
        )
    
    with col_head3:
        vol_color = "#D4AF37" if data.get('volatility', 0) > 0.02 else "#10B981"
        render_metric_card(
            '📉 Volatility',
            f"{data.get('volatility', 0):.4f}",
            delta_label="20-Day Std Dev",
            color=vol_color
        )
    
    with col_head4:
        source = "Mock" if data['is_mock'] else "Live"
        source_color = "#EF4444" if data['is_mock'] else "#10B981"
        render_metric_card(
            '🌐 Data Source',
            source,
            color=source_color
        )

    st.markdown("---")

    # AI Prediction Hero Section
    if models:
        render_prediction_hero(direction, confidence, pred_price, symbol)
    else:
        st.error("⚠️ Models could not be loaded. Please check model files.")

    # Price Chart (Candlestick)
    st.subheader('📈 Price History')
    try:
        if not data['is_mock']:
            has_international_suffix = any(symbol.endswith(suffix) for suffix in ['.KA', '.NS', '.L', '.T', '.HK', '.DE'])
            
            if has_international_suffix or not ALPHA_VANTAGE_KEY:
                ticker = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=100)
                hist_data = ticker.history(start=start_date, end=end_date)
                hist_data = hist_data.sort_index()
                hist_data.columns = [col.lower() for col in hist_data.columns]
                hist_data = hist_data[['open', 'high', 'low', 'close', 'volume']]
            else:
                ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
                hist_data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
                hist_data = hist_data.sort_index()
                hist_data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            fig = go.Figure(data=[go.Candlestick(
                x=hist_data.index,
                open=hist_data['open'],
                high=hist_data['high'],
                low=hist_data['low'],
                close=hist_data['close'],
                increasing_line_color='#10B981',
                decreasing_line_color='#EF4444',
                increasing_fillcolor='#10B981',
                decreasing_fillcolor='#EF4444',
            )])
            
            fig.update_layout(
                title={
                    'text': f"{get_friendly_name(symbol)} Daily Price",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#E5E7EB', 'family': 'Outfit'}
                },
                xaxis_title="Date",
                yaxis_title="Price",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#9CA3AF', 'family': 'Inter'},
                xaxis=dict(
                    gridcolor='rgba(212, 175, 55, 0.1)',
                    gridwidth=1,
                    showgrid=True,
                    zeroline=False,
                    linecolor='rgba(212, 175, 55, 0.3)',
                ),
                yaxis=dict(
                    gridcolor='rgba(212, 175, 55, 0.1)',
                    gridwidth=1,
                    showgrid=True,
                    zeroline=False,
                    linecolor='rgba(212, 175, 55, 0.3)',
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font={'color': '#9CA3AF'},
                    bgcolor='rgba(0,0,0,0)',
                ),
                margin=dict(l=0, r=0, t=50, b=0),
                height=500,
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
        else:
            st.warning("Charts unavailable in Mock Data mode.")
    except Exception as e:
        st.error(f"Could not load chart: {e}")

# === TAB 2: DEEP DIVE ===
with tab2:
    st.header('🧠 Advanced Analysis')
    
    # Clustering / Market Regime
    if models and 'clustering' in models:
        st.subheader('🔍 Market Regime (Clustering)')
        
        clus_features = np.array([[data.get('volatility', 0), data['rsi']]])
        cluster_id = models['clustering'].predict(clus_features)[0]
        
        regime_labels = {
            0: "Regime 0 (Watch) 👁️",
            1: "Regime 1 (Accumulate) 💰",
            2: "Regime 2 (Risk/Volatile) ⚠️"
        }
        regime_name = regime_labels.get(cluster_id, f"Cluster {cluster_id}")
        
        regime_html = f'''
        <div class="metric-card" style="max-width: 600px;">
            <div class="metric-label">📊 Current Market Regime</div>
            <div class="metric-value" style="color: #D4AF37; font-size: 1.5rem;">{regime_name}</div>
            <div style="color: #9CA3AF; font-size: 0.85rem; margin-top: 0.5rem;">
                K-Means Clustering on Volatility & RSI to identify market state.
            </div>
        </div>
        '''
        st.markdown(regime_html, unsafe_allow_html=True)

    st.markdown("---")
    
    # Technical Indicators Chart
    st.subheader('📊 Technical Indicators')
    try:
        if not data['is_mock']:
            has_international_suffix = any(symbol.endswith(suffix) for suffix in ['.KA', '.NS', '.L', '.T', '.HK', '.DE'])
            
            if has_international_suffix or not ALPHA_VANTAGE_KEY:
                ticker = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=100)
                hist_data = ticker.history(start=start_date, end=end_date)
                hist_data = hist_data.sort_index()
                hist_data.columns = [col.lower() for col in hist_data.columns]
                hist_data = hist_data[['open', 'high', 'low', 'close', 'volume']]
            else:
                ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
                hist_data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
                hist_data = hist_data.sort_index()
                hist_data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            delta = hist_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist_data['rsi_plot'] = 100 - (100 / (1 + rs))

            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['rsi_plot'],
                mode='lines',
                name='RSI (14)',
                line=dict(color='#D4AF37', width=2.5),
                hovertemplate='<b>RSI</b>: %{y:.2f}<br>Date: %{x}<extra></extra>'
            ))
            
            fig_rsi.add_hline(
                y=70,
                line_dash="dash",
                line_color="#EF4444",
                annotation_text="Overbought (70)",
                annotation_position="right",
                annotation_font_color="#EF4444"
            )
            fig_rsi.add_hline(
                y=30,
                line_dash="dash",
                line_color="#10B981",
                annotation_text="Oversold (30)",
                annotation_position="right",
                annotation_font_color="#10B981"
            )
            
            fig_rsi.update_layout(
                title={
                    'text': "Relative Strength Index (14)",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#E5E7EB', 'family': 'Outfit'}
                },
                xaxis_title="Date",
                yaxis_title="RSI",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#9CA3AF', 'family': 'Inter'},
                xaxis=dict(
                    gridcolor='rgba(212, 175, 55, 0.1)',
                    gridwidth=1,
                    showgrid=True,
                    zeroline=False,
                    linecolor='rgba(212, 175, 55, 0.3)',
                ),
                yaxis=dict(
                    gridcolor='rgba(212, 175, 55, 0.1)',
                    gridwidth=1,
                    showgrid=True,
                    zeroline=False,
                    linecolor='rgba(212, 175, 55, 0.3)',
                    range=[0, 100]
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font={'color': '#9CA3AF'},
                    bgcolor='rgba(0,0,0,0)',
                ),
                margin=dict(l=0, r=0, t=50, b=0),
                height=500,
            )
            
            st.plotly_chart(fig_rsi, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
        else:
            st.warning("Technical indicators unavailable in Mock Data mode.")
    except Exception as e:
        st.error(f"Could not load technical indicators: {e}")

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #6B7280; font-size: 0.75rem; padding: 1rem; font-family: \'Inter\', sans-serif;">Nuqta | AI Market Insight | Professional Financial Terminal</div>', unsafe_allow_html=True)
