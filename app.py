import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf
import hashlib
import time
from functools import wraps

# Load env vars (for local support)
load_dotenv()

# Configure Plotly renderer for Hugging Face Spaces
pio.renderers.default = "browser"

# --- Config ---
st.set_page_config(
    page_title="Nuqta | AI Market Insight", 
    layout="wide", 
    page_icon="🔷", 
    initial_sidebar_state="collapsed"
)

# --- Model Verification on Startup ---
def verify_models_on_startup():
    """Verifies that models are accessible at startup."""
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    models_dir = BASE_DIR / "models"
    
    if not models_dir.exists():
        print(f"❌ CRITICAL: Models directory not found at {models_dir}")
        return False
    
    pkl_files = list(models_dir.rglob("*.pkl"))
    if len(pkl_files) == 0:
        print(f"❌ CRITICAL: No model files (.pkl) found in {models_dir}")
        return False
    
    print(f"✅ Model verification: Found {len(pkl_files)} model files in {models_dir}")
    print(f"   Example models: {[f.parent.name for f in pkl_files[:5]]}")
    return True

# Run verification on startup
if 'models_verified' not in st.session_state:
    verify_models_on_startup()
    st.session_state.models_verified = True

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

# --- Retry Decorator for API Calls ---
def retry_api_call(max_retries=3, delay=2, backoff=2):
    """Decorator to retry API calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        print(f"✅ API call succeeded on attempt {attempt + 1}")
                    return result
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        print(f"⚠️  API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        print(f"   Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        print(f"❌ API call failed after {max_retries} attempts: {str(e)}")
            
            raise last_exception
        return wrapper
    return decorator

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
    from pathlib import Path
    
    available = []
    BASE_DIR = Path(__file__).resolve().parent
    models_dir = BASE_DIR / "models"
    
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                reg_path = item / "regression_model.pkl"
                if reg_path.exists():
                    available.append(item.name)
                    print(f"✅ Found models for {item.name}")
    
    available.sort()
    if not available:
        print("⚠️  No models found, using default tickers")
        return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"]
    
    print(f"📊 Found {len(available)} stocks with trained models")
    return available

def get_friendly_name(ticker: str) -> str:
    """Returns friendly display name for a ticker."""
    return TICKER_NAMES.get(ticker, ticker)

# --- Helper Functions ---
@st.cache_resource
def load_models_local(symbol):
    """Loads models directly from disk for the specific symbol."""
    from pathlib import Path
    
    # Get absolute path - works in both local and Hugging Face environments
    BASE_DIR = Path(__file__).resolve().parent
    model_path = BASE_DIR / "models" / symbol
    
    models = {}
    errors = []
    
    # Check if directory exists
    if not model_path.exists():
        error_msg = f"Model directory not found: {model_path}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)
    else:
        print(f"✅ Found model directory: {model_path}")
        
        # Try to load each model with detailed error reporting
        model_files = {
            'regression': model_path / "regression_model.pkl",
            'classification': model_path / "classification_model.pkl",
            'clustering': model_path / "clustering_model.pkl"
        }
        
        for model_type, model_file in model_files.items():
            if not model_file.exists():
                error_msg = f"Model file not found: {model_file}"
                print(f"❌ {error_msg}")
                errors.append(error_msg)
            else:
                try:
                    print(f"📦 Loading {model_type} model from {model_file}...")
                    models[model_type] = joblib.load(model_file)
                    print(f"✅ Successfully loaded {model_type} model")
                except Exception as e:
                    error_msg = f"Failed to load {model_type} model from {model_file}: {str(e)}"
                    print(f"❌ {error_msg}")
                    errors.append(error_msg)
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
    
    # If we have some models but not all, still return what we have
    if models:
        if errors:
            print(f"⚠️  Warning: Some models failed to load. Loaded: {list(models.keys())}")
        return models
    
    # If no models loaded, try fallback to AAPL
    if symbol != "AAPL" and not models:
        print(f"⚠️  Attempting fallback to AAPL models...")
        try:
            return load_models_local("AAPL")
        except Exception as fallback_error:
            print(f"❌ Fallback to AAPL also failed: {fallback_error}")
    
    # If we get here, nothing worked
    error_summary = "\n".join(errors) if errors else "Unknown error"
    st.error(f"⚠️ Failed to load models for {symbol}:\n{error_summary}")
    print(f"❌ All model loading attempts failed for {symbol}")
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

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_data(symbol):
    """Fetches raw price data and calculates indicators locally.
    Uses Yahoo Finance for international tickers, Alpha Vantage for US stocks.
    """
    print(f"🔍 Fetching live data for {symbol}...")
    has_international_suffix = any(symbol.endswith(suffix) for suffix in ['.KA', '.NS', '.L', '.T', '.HK', '.DE'])
    
    if has_international_suffix or not ALPHA_VANTAGE_KEY:
        print(f"📊 Using Yahoo Finance for {symbol} (international ticker or no Alpha Vantage key)")
        return fetch_live_data_yahoo(symbol)
    else:
        try:
            print(f"📊 Trying Alpha Vantage first for {symbol}...")
            return fetch_live_data_alphavantage(symbol)
        except Exception as e:
            print(f"⚠️  Alpha Vantage failed for {symbol}: {e}")
            print(f"📊 Falling back to Yahoo Finance for {symbol}...")
            return fetch_live_data_yahoo(symbol)

@retry_api_call(max_retries=3, delay=2)
def fetch_live_data_yahoo(symbol):
    """Fetches data using Yahoo Finance with retry logic."""
    try:
        print(f"📡 Connecting to Yahoo Finance for {symbol}...")
        ticker = yf.Ticker(symbol)
        
        # Set timeout and retry parameters
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        print(f"   Fetching data from {start_date.date()} to {end_date.date()}...")
        data = ticker.history(start=start_date, end=end_date, timeout=10)
        
        if data is None:
            raise ValueError(f"Yahoo Finance returned None for {symbol}")
        
        if data.empty:
            raise ValueError(f"Yahoo Finance returned empty DataFrame for {symbol}")
        
        print(f"   ✅ Received {len(data)} rows of data")
        
        # Validate data has required columns
        data.columns = [col.lower() for col in data.columns]
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available: {list(data.columns)}")
        
        data = data[required_cols]
        data = data.sort_index()
        
        # Validate data has valid values
        if data[['open', 'high', 'low', 'close']].isna().all().all():
            raise ValueError(f"All price data is NaN for {symbol}")
        
        # Calculate indicators
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
        prev = data.iloc[-2] if len(data) > 1 else latest
        
        change_percent = ((latest['close'] - prev['close']) / prev['close']) * 100 if prev['close'] > 0 else 0.0
        
        result = {
            "price": float(latest['close']),
            "change": change_percent,
            "sma_20": float(latest['sma_20']) if not np.isnan(latest['sma_20']) else float(latest['close']),
            "sma_50": float(latest['sma_50']) if not np.isnan(latest['sma_50']) else float(latest['close']),
            "rsi": float(latest['rsi']) if not np.isnan(latest['rsi']) else 50.0,
            "macd": float(latest['macd']) if not np.isnan(latest['macd']) else 0.0,
            "volatility": float(latest['volatility']) if not np.isnan(latest['volatility']) else 0.0,
            "is_mock": False
        }
        
        print(f"✅ Successfully fetched live data for {symbol}: Price=${result['price']:.2f}")
        return result
        
    except Exception as e:
        error_msg = f"Yahoo Finance fetch failed for {symbol}: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise Exception(error_msg)

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
            "sma_20": float(latest['sma_20']) if not np.isnan(latest['sma_20']) else float(latest['close']),
            "sma_50": float(latest['sma_50']) if not np.isnan(latest['sma_50']) else float(latest['close']),
            "rsi": float(latest['rsi']) if not np.isnan(latest['rsi']) else 50.0,
            "macd": float(latest['macd']) if not np.isnan(latest['macd']) else 0.0,
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
col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([3, 1, 1, 1])
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

with col_nav3:
    if st.button("🔍 Debug", key="debug_btn", use_container_width=True, help="Show diagnostic information"):
        st.session_state.show_debug = not st.session_state.get('show_debug', False)

# Main Title
st.markdown("<br>", unsafe_allow_html=True)
st.title("Nuqta | AI Market Insight")

# Debug Panel (if enabled)
if st.session_state.get('show_debug', False):
    with st.expander("🔍 Diagnostic Information", expanded=True):
        st.subheader("Environment Check")
        
        # Check API keys
        col_env1, col_env2 = st.columns(2)
        with col_env1:
            alpha_status = "✅ Set" if ALPHA_VANTAGE_KEY else "❌ Not Set"
            st.write(f"**Alpha Vantage API Key:** {alpha_status}")
        
        with col_env2:
            webhook_status = "✅ Set" if WEBHOOK_URL else "❌ Not Set"
            st.write(f"**Discord Webhook:** {webhook_status}")
        
        # Test Yahoo Finance connection
        st.subheader("API Connection Test")
        test_symbol = st.text_input("Test Ticker", value="AAPL")
        if st.button("🧪 Test Connection"):
            with st.spinner("Testing Yahoo Finance connection..."):
                try:
                    test_ticker = yf.Ticker(test_symbol)
                    test_data = test_ticker.history(period="5d", timeout=10)
                    if not test_data.empty:
                        st.success(f"✅ Successfully connected! Got {len(test_data)} rows for {test_symbol}")
                        st.dataframe(test_data.tail())
                    else:
                        st.error(f"❌ Connection succeeded but returned empty data for {test_symbol}")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language='python')
        
        # System info
        st.subheader("System Information")
        import sys
        import platform
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Platform:** {platform.platform()}")
        st.write(f"**yfinance Version:** {yf.__version__ if hasattr(yf, '__version__') else 'Unknown'}")
        
        # Cache info
        st.subheader("Cache Status")
        if st.button("🗑️ Clear All Caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ All caches cleared!")

# --- Main Logic ---

    # 1. Fetch Data
with st.spinner(f"Fetching Live Data for {symbol}..."):
    try:
        data = fetch_live_data(symbol)
    except Exception as e:
        error_msg = f"Failed to fetch data for {symbol}: {str(e)}"
        print(f"❌ {error_msg}")
        st.error(f"⚠️ {error_msg}")
        st.warning("🔄 Falling back to mock data. This may indicate network issues or API problems.")
        data = get_mock_data(symbol)

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
            hist_data = None
            
            try:
                if has_international_suffix or not ALPHA_VANTAGE_KEY:
                    print(f"📊 Fetching historical chart data for {symbol} using Yahoo Finance...")
                    
                    # Use retry logic for chart data
                    max_chart_retries = 3
                    chart_data_success = False
                    hist_data = None
                    
                    for chart_attempt in range(max_chart_retries):
                        try:
                            ticker = yf.Ticker(symbol)
                            end_date = datetime.now()
                            start_date = end_date - timedelta(days=100)
                            
                            print(f"   Attempt {chart_attempt + 1}/{max_chart_retries}: Fetching from {start_date.date()} to {end_date.date()}...")
                            hist_data = ticker.history(start=start_date, end=end_date, timeout=15)
                            
                            if hist_data is None:
                                raise ValueError(f"Yahoo Finance returned None for {symbol}")
                            
                            if hist_data.empty:
                                raise ValueError(f"Yahoo Finance returned empty DataFrame for {symbol}")
                            
                            print(f"   ✅ Received {len(hist_data)} rows of historical data")
                            chart_data_success = True
                            break
                            
                        except Exception as chart_error:
                            if chart_attempt < max_chart_retries - 1:
                                wait_time = 2 * (chart_attempt + 1)
                                print(f"   ⚠️  Attempt {chart_attempt + 1} failed: {str(chart_error)}")
                                print(f"   Retrying in {wait_time} seconds...")
                                time.sleep(wait_time)
                            else:
                                raise chart_error
                    
                    if not chart_data_success or hist_data is None or hist_data.empty:
                        raise ValueError(f"Failed to fetch chart data after {max_chart_retries} attempts")
                    
                    print(f"✅ Fetched {len(hist_data)} rows of historical data")
                    hist_data = hist_data.sort_index()
                    hist_data.columns = [col.lower() for col in hist_data.columns]
                    
                    # Validate required columns exist
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in hist_data.columns]
                    if missing_cols:
                        raise ValueError(f"Missing required columns: {missing_cols}. Available: {list(hist_data.columns)}")
                    
                    hist_data = hist_data[required_cols]
                    
                    # Validate data has valid values
                    if hist_data[['open', 'high', 'low', 'close']].isna().all().all():
                        raise ValueError(f"All price data is NaN for {symbol}")
                else:
                    print(f"📊 Fetching historical data for {symbol} using Alpha Vantage...")
                    ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
                    hist_data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
                    
                    if hist_data is None or hist_data.empty:
                        raise ValueError(f"Alpha Vantage returned empty data for {symbol}")
                    
                    print(f"✅ Fetched {len(hist_data)} rows of historical data")
                    hist_data = hist_data.sort_index()
                    hist_data.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # CRITICAL: Validate data is not empty before creating chart
                if hist_data.empty:
                    raise ValueError(f"Historical data is empty for {symbol}")
                
                if len(hist_data) == 0:
                    raise ValueError(f"No data points found for {symbol}")
                
                # Validate data has valid values (not all NaN)
                if hist_data[['open', 'high', 'low', 'close']].isna().all().all():
                    raise ValueError(f"All price data is NaN for {symbol}")
                
                print(f"📈 Creating chart with {len(hist_data)} data points...")
                
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
                    paper_bgcolor='#1A202C',
                    plot_bgcolor='#2D3748',
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
                    autosize=True,
                )
                
                st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    config={'displayModeBar': True, 'displaylogo': False},
                    key=f"price_chart_{symbol}"
                )
                print(f"✅ Chart displayed successfully for {symbol}")
                
            except Exception as fetch_error:
                error_msg = str(fetch_error)
                print(f"❌ Error fetching chart data for {symbol}: {error_msg}")
                import traceback
                full_traceback = traceback.format_exc()
                print(f"Full traceback:\n{full_traceback}")
                
                # Show detailed error in UI
                st.error(f"❌ **Chart Data Error for {symbol}**")
                st.warning(f"**Error Details:** {error_msg}")
                
                # Provide helpful troubleshooting info
                troubleshooting = f"""
                **Possible Causes:**
                1. 🌐 Network connectivity issues on Hugging Face
                2. ⏱️ Yahoo Finance API timeout or rate limiting
                3. 🔑 API key issues (if using Alpha Vantage)
                4. 📊 Ticker symbol not available or invalid
                
                **Troubleshooting Steps:**
                - 🔄 Try refreshing the page
                - ⏳ Wait a few minutes and try again (rate limits)
                - 🔍 Check if the ticker symbol is correct
                - 📝 Check the logs below for more details
                """
                st.info(troubleshooting)
                
                # Show error details in expander
                with st.expander("🔍 View Technical Details"):
                    st.code(full_traceback, language='python')
        else:
            st.warning("Charts unavailable in Mock Data mode.")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Chart error for {symbol}: {error_msg}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        st.error(f"Could not load chart: {error_msg}")

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
            hist_data = None
            
            try:
                if has_international_suffix or not ALPHA_VANTAGE_KEY:
                    print(f"📊 Fetching RSI data for {symbol} using Yahoo Finance...")
                    
                    # Use retry logic for RSI data
                    max_rsi_retries = 3
                    rsi_data_success = False
                    hist_data = None
                    
                    for rsi_attempt in range(max_rsi_retries):
                        try:
                            ticker = yf.Ticker(symbol)
                            end_date = datetime.now()
                            start_date = end_date - timedelta(days=100)
                            
                            print(f"   Attempt {rsi_attempt + 1}/{max_rsi_retries}: Fetching RSI data...")
                            hist_data = ticker.history(start=start_date, end=end_date, timeout=15)
                            
                            if hist_data is None or hist_data.empty:
                                raise ValueError(f"Yahoo Finance returned empty data for {symbol}")
                            
                            rsi_data_success = True
                            break
                            
                        except Exception as rsi_error:
                            if rsi_attempt < max_rsi_retries - 1:
                                wait_time = 2 * (rsi_attempt + 1)
                                print(f"   ⚠️  Attempt {rsi_attempt + 1} failed: {str(rsi_error)}")
                                print(f"   Retrying in {wait_time} seconds...")
                                time.sleep(wait_time)
                            else:
                                raise rsi_error
                    
                    if not rsi_data_success or hist_data is None or hist_data.empty:
                        raise ValueError(f"Failed to fetch RSI data after {max_rsi_retries} attempts")
                    
                    hist_data = hist_data.sort_index()
                    hist_data.columns = [col.lower() for col in hist_data.columns]
                    hist_data = hist_data[['open', 'high', 'low', 'close', 'volume']]
                else:
                    print(f"📊 Fetching RSI data for {symbol} using Alpha Vantage...")
                    ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
                    hist_data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
                    
                    if hist_data is None or hist_data.empty:
                        raise ValueError(f"Alpha Vantage returned empty data for {symbol}")
                    
                    hist_data = hist_data.sort_index()
                    hist_data.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # Validate data is not empty
                if hist_data.empty or len(hist_data) == 0:
                    raise ValueError(f"Historical data is empty for {symbol}")
                
                if 'close' not in hist_data.columns:
                    raise ValueError(f"Missing 'close' column in data for {symbol}")
                
                print(f"📊 Calculating RSI with {len(hist_data)} data points...")
                
                delta = hist_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                hist_data['rsi_plot'] = 100 - (100 / (1 + rs))
                
                # Validate RSI data is valid
                if hist_data['rsi_plot'].isna().all():
                    raise ValueError(f"RSI calculation resulted in all NaN values for {symbol}")
                
                print(f"📈 Creating RSI chart with {len(hist_data)} data points...")
                
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
                    paper_bgcolor='#1A202C',
                    plot_bgcolor='#2D3748',
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
                    autosize=True,
                )
                
                st.plotly_chart(
                    fig_rsi, 
                    use_container_width=True, 
                    config={'displayModeBar': True, 'displaylogo': False},
                    key=f"rsi_chart_{symbol}"
                )
                print(f"✅ RSI chart displayed successfully for {symbol}")
                
            except Exception as fetch_error:
                error_msg = str(fetch_error)
                print(f"❌ Error fetching RSI data for {symbol}: {error_msg}")
                import traceback
                full_traceback = traceback.format_exc()
                print(f"Full traceback:\n{full_traceback}")
                
                st.error(f"❌ **Technical Indicators Error for {symbol}**")
                st.warning(f"**Error Details:** {error_msg}")
                st.info("💡 This may be due to network issues, API rate limits, or the ticker not being available. Try refreshing the page.")
        else:
            st.warning("Technical indicators unavailable in Mock Data mode.")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Technical indicators error for {symbol}: {error_msg}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        st.error(f"Could not load technical indicators: {error_msg}")

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #6B7280; font-size: 0.75rem; padding: 1rem; font-family: \'Inter\', sans-serif;">Nuqta | AI Market Insight | Professional Financial Terminal</div>', unsafe_allow_html=True)
