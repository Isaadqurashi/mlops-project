import os
import pandas as pd
from prefect import flow, task
from src.ingestion.ingest import fetch_daily_data
from src.processing.features import process_data
from src.processing.split import split_data
from src.models.train import ModelTrainer
from dotenv import load_dotenv

load_dotenv()


from src.orchestration.notifications import notify_discord  # noqa: E402

# Optional import for data validation (may fail due to deepchecks compatibility)
try:
    from tests.data_validation import validate_data
    VALIDATION_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Data validation not available: {e}")
    print("⚠️  Pipeline will continue without validation...")
    VALIDATION_AVAILABLE = False
    def validate_data(*args, **kwargs):
        """Dummy function if validation is not available."""
        return None, None

@task(retries=3, retry_delay_seconds=60)
def fetch_stock_data(symbol: str):
    """Task to fetch stock data with retries."""
    try:
        file_path = fetch_daily_data(symbol)
        return file_path
    except Exception as e:
        raise e

@task
def process_stock_data(file_path: str, symbol: str):
    """Task to process stock data."""
    output_path = f"data/processed/{symbol}_processed.csv"
    os.makedirs("data/processed", exist_ok=True)
    df = process_data(file_path, output_path)
    return df

@task
def train_and_evaluate(df: pd.DataFrame, symbol: str):
    """Task to train models and evaluate."""
    train_df, test_df = split_data(df)
    
    # Validation (optional - continues even if validation fails)
    if VALIDATION_AVAILABLE:
        try:
            validate_data(train_df, test_df, output_dir=f"reports/{symbol}")
        except Exception as e:
            print(f"⚠️  Validation skipped for {symbol}: {e}")
            print("⚠️  Continuing with model training...")
    else:
        print(f"⚠️  Skipping validation for {symbol} (validation not available)")
    
    # Training
    trainer = ModelTrainer(output_dir=f"models/{symbol}", metrics_dir=f"reports/{symbol}")
    trainer.train_regression(train_df, test_df)
    trainer.train_classification(train_df, test_df)
    trainer.train_clustering(df)
    trainer.train_pca(df)
    trainer.save_metrics()
    
    return True

# Global ticker list organized by region
GLOBAL_TICKERS = {
    "USA": ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA", "AMZN"],
    "Pakistan": ["OGDC.KA", "LUCK.KA", "TRG.KA", "ENGRO.KA", "SYS.KA"],
    "India": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"],
    "UK": ["RR.L", "AZN.L", "HSBA.L", "BP.L"],
    "Japan": ["7203.T", "6758.T", "9984.T"],
    "Hong Kong": ["0700.HK", "9988.HK", "1810.HK"],
    "Germany": ["SAP.DE", "SIE.DE", "VOW3.DE"]
}

def get_all_tickers() -> list[str]:
    """Returns a flat list of all global tickers."""
    return [ticker for region_tickers in GLOBAL_TICKERS.values() for ticker in region_tickers]

@flow(name="End-to-End Stock Prediction Pipeline")
def main_pipeline(symbols: list[str] = []):
    """
    Main flow to run the entire pipeline.
    
    Args:
        symbols: List of stock symbols to process. If None or empty, uses all global tickers.
    """
    # Prefect validation requires a list type (not None)
    # Empty list is used as sentinel to trigger default behavior (use all tickers)
    if len(symbols) == 0:
        symbols = get_all_tickers()
    
    notify_discord(f"🚀 Starting End-to-End Pipeline for {len(symbols)} symbols...")
    
    successful = []
    failed = []
    
    for symbol in symbols:
        try:
            print(f"\n{'='*60}")
            print(f"Processing {symbol}...")
            print(f"{'='*60}")
            
            raw_path = fetch_stock_data(symbol)
            df = process_stock_data(raw_path, symbol)
            train_and_evaluate(df, symbol)
            
            successful.append(symbol)
            notify_discord(f"✅ Pipeline completed for {symbol}")
            print(f"✅ Successfully processed {symbol}")
            
        except Exception as e:
            error_msg = str(e)
            failed.append((symbol, error_msg))
            notify_discord(f"❌ Pipeline failed for {symbol}: {error_msg}")
            print(f"❌ Error processing {symbol}: {error_msg}")
            print("⚠️  Continuing with next ticker...")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("Pipeline Summary:")
    print(f"  ✅ Successful: {len(successful)}/{len(symbols)}")
    print(f"  ❌ Failed: {len(failed)}/{len(symbols)}")
    if failed:
        print("\nFailed symbols:")
        for symbol, error in failed:
            print(f"  - {symbol}: {error}")
    print(f"{'='*60}")
    
    notify_discord(f"📊 Pipeline completed: {len(successful)}/{len(symbols)} successful")

if __name__ == "__main__":
    # Call without arguments - empty list will trigger default behavior (use all tickers)
    main_pipeline()
