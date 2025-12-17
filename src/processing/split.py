import pandas as pd
from typing import Tuple

# Configuration for rolling window
ROLLING_WINDOW_DAYS = 700  # ~2-3 years of trading days

def apply_rolling_window(df: pd.DataFrame, window_days: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """
    Applies rolling window to keep only recent data for training.
    This minimizes data drift by focusing on current market dynamics.
    """
    if len(df) <= window_days:
        print(f"   ⚠️ Data has {len(df)} rows, less than {window_days}. Using all available data.")
        return df
    
    df_windowed = df.tail(window_days).copy()
    print(f"   ✅ Applied rolling window: {len(df)} → {len(df_windowed)} rows (last {window_days} days)")
    return df_windowed

def split_data(df: pd.DataFrame, test_size: float = 0.2, apply_window: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into training and testing sets using time-series split (no shuffling).
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing (default: 0.2)
        apply_window: If True, applies rolling window to focus on recent data
    
    Returns:
        Tuple of (train_df, test_df)
    """
    # Apply rolling window first to minimize drift from old data
    if apply_window:
        df = apply_rolling_window(df)
    
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"   ✅ Data split: Train ({len(train_df)}), Test ({len(test_df)})")
    return train_df, test_df

if __name__ == "__main__":
    # Example usage
    pass
