import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def fetch_daily_data(symbol: str, output_dir: str = "data/raw"):
    """
    Fetches daily time series data for a given symbol.
    Uses Yahoo Finance for international tickers (with suffixes like .NS, .L, .T, etc.)
    Falls back to Alpha Vantage for US stocks if API key is available.
    """
    # Check if symbol has international suffix (not a pure US ticker)
    has_international_suffix = any(symbol.endswith(suffix) for suffix in ['.KA', '.NS', '.L', '.T', '.HK', '.DE'])
    
    # Use Yahoo Finance for international tickers or if Alpha Vantage key is missing
    if has_international_suffix or not API_KEY:
        return fetch_daily_data_yahoo(symbol, output_dir)
    else:
        # Try Alpha Vantage first for US stocks
        try:
            return fetch_daily_data_alphavantage(symbol, output_dir)
        except Exception as e:
            print(f"Alpha Vantage failed for {symbol}: {e}. Falling back to Yahoo Finance...")
            return fetch_daily_data_yahoo(symbol, output_dir)

def fetch_daily_data_yahoo(symbol: str, output_dir: str = "data/raw"):
    """
    Fetches daily time series data using Yahoo Finance.
    Works for both US and international stocks.
    """
    print(f"Fetching data for {symbol} from Yahoo Finance...")
    
    try:
        # Download data for the past 2 years (enough for indicators)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Rename columns to match expected format
        df.reset_index(inplace=True)
        df.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        # Select only required columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by date
        df = df.sort_values('timestamp')
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{symbol}_daily.csv")
        df.to_csv(file_path, index=False)
        
        print(f"Data saved to {file_path} ({len(df)} rows)")
        return file_path
        
    except Exception as e:
        raise Exception(f"Yahoo Finance fetch failed for {symbol}: {str(e)}")

def fetch_daily_data_alphavantage(symbol: str, output_dir: str = "data/raw"):
    """
    Fetches daily time series data from Alpha Vantage (for US stocks).
    """
    if not API_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables.")

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY,
        "datatype": "csv",
        "outputsize": "compact" # Get compact history (last 100 data points)
    }

    print(f"Fetching data for {symbol} from Alpha Vantage...")
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.text}")

    # Check if response contains error message
    if "Error Message" in response.text:
         raise Exception(f"API Error: {response.text}")

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{symbol}_daily.csv")
    
    with open(file_path, "w") as f:
        f.write(response.text)
    
    print(f"Data saved to {file_path}")
    return file_path

if __name__ == "__main__":
    # Example usage for manual execution
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"]
    print(f"Manually fetching data for: {symbols}")
    
    for symbol in symbols:
        try:
            fetch_daily_data(symbol)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
