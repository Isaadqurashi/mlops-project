import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def fetch_daily_data(symbol: str, output_dir: str = "data/raw"):
    """
    Fetches daily time series data for a given symbol from Alpha Vantage
    and saves it as a CSV file.
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

    print(f"Fetching data for {symbol}...")
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
