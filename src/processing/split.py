import pandas as pd
from typing import Tuple

def split_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into training and testing sets using time-series split (no shuffling).
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Data split: Train ({len(train_df)}), Test ({len(test_df)})")
    return train_df, test_df

if __name__ == "__main__":
    # Example usage
    pass
