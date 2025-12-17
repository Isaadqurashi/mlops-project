import pandas as pd
import numpy as np

def calculate_sma(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """Calculates Simple Moving Average (SMA)."""
    return data['close'].rolling(window=window).mean()

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.DataFrame, slow: int = 26, fast: int = 12, signal: int = 9):
    """Calculates MACD, Signal Line, and Histogram."""
    exp1 = data['close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def process_data(file_path: str, output_path: str = None):
    """
    Loads data, adds technical indicators, and saves processed data.
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Ensure column names are lower case
    df.columns = [c.lower() for c in df.columns]

    # Add indicators
    df['sma_20'] = calculate_sma(df, 20)
    df['sma_50'] = calculate_sma(df, 50)
    df['rsi'] = calculate_rsi(df)
    df['macd'], df['macd_signal'] = calculate_macd(df)
    
    # Target for Classification (Next Day Direction: 1 for Up, 0 for Down)
    df['target_direction'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Target for Regression (Next Day Close)
    df['target_price'] = df['close'].shift(-1)
    
    # Drop NaNs created by rolling windows
    df = df.dropna()
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    # Example usage
    # process_data("data/raw/AAPL_daily.csv", "data/processed/AAPL_processed.csv")
    pass
