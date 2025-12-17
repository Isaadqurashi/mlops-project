"""
Background monitoring service for Discord notifications.
Checks all tickers every 3 minutes and sends notifications for significant predictions.
"""
import threading
import time
import os
import joblib
import numpy as np
import hashlib
from datetime import datetime, timedelta
from typing import Dict
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv

from src.orchestration.notifications import notify_discord

load_dotenv()

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
CHECK_INTERVAL = 180  # 3 minutes in seconds
PREDICTION_THRESHOLD = 0.5  # 0.5% threshold

# Track sent notifications to avoid duplicates
# Format: {symbol: (last_prediction_id, last_check_time)}
sent_notifications: Dict[str, tuple] = {}
notification_lock = threading.Lock()

# Ticker name mapping (same as in app.py)
TICKER_NAMES = {
    "AAPL": "Apple (AAPL)",
    "GOOGL": "Alphabet (GOOGL)",
    "MSFT": "Microsoft (MSFT)",
    "NVDA": "NVIDIA (NVDA)",
    "TSLA": "Tesla (TSLA)",
    "AMZN": "Amazon (AMZN)",
    "OGDC.KA": "OGDC (OGDC.KA)",
    "LUCK.KA": "Lucky Cement (LUCK.KA)",
    "TRG.KA": "TRG Pakistan (TRG.KA)",
    "ENGRO.KA": "Engro (ENGRO.KA)",
    "SYS.KA": "Systems Limited (SYS.KA)",
    "RELIANCE.NS": "Reliance Industries (RELIANCE.NS)",
    "TCS.NS": "Tata Consultancy Services (TCS.NS)",
    "HDFCBANK.NS": "HDFC Bank (HDFCBANK.NS)",
    "INFY.NS": "Infosys (INFY.NS)",
    "RR.L": "Rolls-Royce (RR.L)",
    "AZN.L": "AstraZeneca (AZN.L)",
    "HSBA.L": "HSBC (HSBA.L)",
    "BP.L": "BP (BP.L)",
    "7203.T": "Toyota (7203.T)",
    "6758.T": "Sony (6758.T)",
    "9984.T": "SoftBank Group (9984.T)",
    "0700.HK": "Tencent (0700.HK)",
    "9988.HK": "Alibaba (9988.HK)",
    "1810.HK": "Xiaomi (1810.HK)",
    "SAP.DE": "SAP (SAP.DE)",
    "SIE.DE": "Siemens (SIE.DE)",
    "VOW3.DE": "Volkswagen (VOW3.DE)",
}

def get_friendly_name(ticker: str) -> str:
    """Returns friendly display name for a ticker."""
    return TICKER_NAMES.get(ticker, ticker)

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
    return available if available else []

def load_models(symbol):
    """Loads models for a specific symbol."""
    model_path = f"models/{symbol}"
    models = {}
    try:
        models['regression'] = joblib.load(f"{model_path}/regression_model.pkl")
        models['classification'] = joblib.load(f"{model_path}/classification_model.pkl")
        return models
    except Exception as e:
        print(f"Failed to load models for {symbol}: {e}")
        return None

def fetch_live_data(symbol):
    """Fetches live data for a ticker."""
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
        }
    except Exception as e:
        print(f"Yahoo Finance fetch failed for {symbol}: {e}")
        return None

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
        }
    except Exception as e:
        raise Exception(f"Alpha Vantage fetch failed: {e}")

def get_prediction_id(symbol, date_str, predicted_price):
    """Creates a unique hash ID for a prediction to prevent duplicate alerts."""
    unique_string = f"{symbol}_{date_str}_{predicted_price:.2f}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def send_discord_alert(symbol, price, change_percent, prediction_dir, target_price, prediction_change_pct):
    """Sends a professionally formatted message to Discord."""
    is_bullish = prediction_change_pct >= 0.5
    is_bearish = prediction_change_pct <= -0.5
    
    direction_emoji = "📈" if is_bullish else "📉"
    trend_emoji = "🟢" if is_bullish else "🔴"
    arrow_emoji = "⬆️" if is_bullish else "⬇️"
    
    friendly_name = get_friendly_name(symbol)
    
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

def check_ticker(symbol):
    """Checks a single ticker and sends notification if threshold is met."""
    try:
        # Load models
        models = load_models(symbol)
        if not models:
            return False
        
        # Fetch live data
        data = fetch_live_data(symbol)
        if not data:
            return False
        
        # Calculate predictions
        features = np.array([[data['sma_20'], data['sma_50'], data['rsi'], data['macd']]])
        current_price = data.get('price', 0.0)
        pred_price = models['regression'].predict(features)[0]
        pred_direction_prob = models['classification'].predict_proba(features)[0]
        direction = "UP" if pred_direction_prob[1] > 0.5 else "DOWN"
        
        # Calculate prediction change percentage
        if current_price > 0:
            prediction_change_pct = ((pred_price - current_price) / current_price) * 100
        else:
            return False
        
        # Check if threshold is met
        if abs(prediction_change_pct) >= PREDICTION_THRESHOLD:
            # Check if we've already sent this notification
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_prediction_id = get_prediction_id(symbol, current_date, pred_price)
            
            with notification_lock:
                last_check_time = sent_notifications.get(symbol, (None, None))[1]
                
                # Send notification if:
                # 1. We haven't checked this ticker yet, OR
                # 2. It's been more than 3 minutes since last check (send every 3 minutes if threshold still met)
                should_send = (
                    last_check_time is None or
                    (time.time() - last_check_time) >= CHECK_INTERVAL
                )
                
                if should_send:
                    success, status_msg = send_discord_alert(
                        symbol,
                        current_price,
                        data.get('change', 0.0),
                        direction,
                        pred_price,
                        prediction_change_pct
                    )
                    
                    if success:
                        sent_notifications[symbol] = (current_prediction_id, time.time())
                        print(f"✅ Sent Discord notification for {symbol}: {prediction_change_pct:+.2f}%")
                        return True
                    else:
                        print(f"❌ Failed to send notification for {symbol}: {status_msg}")
                        return False
                else:
                    time_remaining = CHECK_INTERVAL - (time.time() - last_check_time)
                    print(f"⏭️  Skipping {symbol} (next check in {time_remaining/60:.1f} min)")
                    return False
        
        return False
    except Exception as e:
        print(f"❌ Error checking {symbol}: {e}")
        return False

def monitor_loop():
    """Main monitoring loop that runs every 3 minutes."""
    print(f"🔔 Starting Discord notification monitor (checking every {CHECK_INTERVAL/60:.1f} minutes)...")
    
    while True:
        try:
            # Get all available tickers
            tickers = get_available_stocks()
            
            if not tickers:
                print("⚠️  No tickers available. Waiting for models to be trained...")
                time.sleep(CHECK_INTERVAL)
                continue
            
            print(f"\n{'='*60}")
            print(f"🔍 Checking {len(tickers)} tickers at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            notifications_sent = 0
            for symbol in tickers:
                if check_ticker(symbol):
                    notifications_sent += 1
                time.sleep(1)  # Small delay between tickers to avoid rate limiting
            
            print(f"📊 Check complete: {notifications_sent} notification(s) sent")
            print(f"⏰ Next check in {CHECK_INTERVAL/60:.1f} minutes\n")
            
        except Exception as e:
            print(f"❌ Error in monitoring loop: {e}")
        
        # Wait for next check
        time.sleep(CHECK_INTERVAL)

def start_monitor():
    """Starts the background monitoring thread."""
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    print("✅ Background monitor started")
    return monitor_thread
