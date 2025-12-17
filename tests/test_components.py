import pytest
import pandas as pd
import numpy as np
from src.processing.features import calculate_sma, calculate_rsi, calculate_macd, process_data
from src.processing.split import split_data

# Sample Data Fixture
@pytest.fixture
def sample_data():
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=100),
        'close': np.random.rand(100) * 100
    }
    return pd.DataFrame(data)

def test_calculate_sma(sample_data):
    """Test Simple Moving Average calculation."""
    window = 20
    sma = calculate_sma(sample_data, window)
    assert len(sma) == 100
    assert sma.iloc[0:window-1].isna().all() # First window-1 should be NaN
    assert not sma.iloc[window:].isna().any()

def test_calculate_rsi(sample_data):
    """Test RSI calculation."""
    rsi = calculate_rsi(sample_data)
    assert len(rsi) == 100
    assert rsi.min() >= 0
    assert rsi.max() <= 100

def test_calculate_macd(sample_data):
    """Test MACD calculation."""
    macd, signal = calculate_macd(sample_data)
    assert len(macd) == 100
    assert len(signal) == 100
    assert not macd.isna().all()

def test_split_data(sample_data):
    """Test data splitting."""
    train, test = split_data(sample_data, test_size=0.2)
    assert len(train) == 80
    assert len(test) == 20
    # Ensure no overlap and correct order
    assert train['timestamp'].max() < test['timestamp'].min()

def test_process_data_structure(tmp_path):
    """Test process_data function output structure."""
    # Create a dummy CSV
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=60),
        'close': [100 + i for i in range(60)] # Linear uptrend
    })
    input_file = tmp_path / "test_input.csv"
    df.to_csv(input_file, index=False)
    
    processed_df = process_data(str(input_file))
    
    expected_columns = ['sma_20', 'sma_50', 'rsi', 'macd', 'target_direction', 'target_price']
    for col in expected_columns:
        assert col in processed_df.columns
    
    # Check if NaNs from rolling windows are dropped
    # SMA_50 needs 50 points, so we expect some data loss
    assert len(processed_df) < 60
