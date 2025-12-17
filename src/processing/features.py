import pandas as pd
import numpy as np

# --- Configuration ---
ROLLING_WINDOW_DAYS = 700  # ~2-3 years of trading days for rolling window training

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

def apply_rolling_window(df: pd.DataFrame, window_days: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """
    Applies rolling window to keep only recent data for training.
    This minimizes data drift by focusing on current market dynamics.
    
    Args:
        df: DataFrame with historical data
        window_days: Number of trading days to keep (default: 700 ~2-3 years)
    
    Returns:
        DataFrame with only the most recent window_days of data
    """
    if len(df) <= window_days:
        print(f"   ⚠️ Data has {len(df)} rows, less than {window_days}. Using all available data.")
        return df
    
    # Keep only the last N trading days
    df_windowed = df.tail(window_days).copy()
    print(f"   ✅ Applied rolling window: {len(df)} → {len(df_windowed)} rows (last {window_days} days)")
    return df_windowed

def add_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds stationary features to reduce data drift.
    Log returns are stationary (mean-reverting) unlike raw prices.
    
    Args:
        df: DataFrame with 'close' column
    
    Returns:
        DataFrame with added stationary features
    """
    # Log Returns - more stationary than raw prices
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Percentage Returns (alternative stationary feature)
    df['pct_return'] = df['close'].pct_change()
    
    # Rolling volatility (stationary measure of risk)
    df['volatility_20'] = df['log_return'].rolling(window=20).std()
    
    # Normalized price (z-score within rolling window)
    df['price_zscore'] = (df['close'] - df['close'].rolling(window=50).mean()) / df['close'].rolling(window=50).std()
    
    return df

def process_data(file_path: str, output_path: str = None, apply_window: bool = True):
    """
    Loads data, adds technical indicators with drift-resistant features, and saves processed data.
    
    Args:
        file_path: Path to raw data CSV
        output_path: Path to save processed data (optional)
        apply_window: If True, applies rolling window to focus on recent data
    
    Returns:
        Processed DataFrame
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Ensure column names are lower case
    df.columns = [c.lower() for c in df.columns]
    
    # --- STEP 1: Apply Rolling Window (minimize drift from old data) ---
    if apply_window:
        df = apply_rolling_window(df, ROLLING_WINDOW_DAYS)

    # --- STEP 2: Add Stationary Features (reduce drift from non-stationary data) ---
    df = add_stationary_features(df)

    # --- STEP 3: Add Technical Indicators ---
    df['sma_20'] = calculate_sma(df, 20)
    df['sma_50'] = calculate_sma(df, 50)
    df['rsi'] = calculate_rsi(df)
    df['macd'], df['macd_signal'] = calculate_macd(df)
    
    # --- STEP 4: Create Targets ---
    # Target for Classification (Next Day Direction: 1 for Up, 0 for Down)
    df['target_direction'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Target for Regression (Next Day Close)
    df['target_price'] = df['close'].shift(-1)
    
    # --- STEP 5: Drop NaNs created by rolling windows and calculations ---
    initial_len = len(df)
    df = df.dropna()
    print(f"   Dropped {initial_len - len(df)} rows with NaN values. Final: {len(df)} rows.")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"   ✅ Processed data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    # Example usage
    # process_data("data/raw/AAPL_daily.csv", "data/processed/AAPL_processed.csv")
    pass
