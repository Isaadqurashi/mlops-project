import os
import requests
import pandas as pd
from prefect import flow, task
from src.ingestion.ingest import fetch_daily_data
from src.processing.features import process_data
from src.processing.split import split_data
from src.models.train import ModelTrainer
from tests.data_validation import validate_data
from dotenv import load_dotenv

load_dotenv()

from src.orchestration.notifications import notify_discord

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
    
    # Validation
    validate_data(train_df, test_df, output_dir=f"reports/{symbol}")
    
    # Training
    trainer = ModelTrainer(output_dir=f"models/{symbol}", metrics_dir=f"reports/{symbol}")
    trainer.train_regression(train_df, test_df)
    trainer.train_classification(train_df, test_df)
    trainer.train_clustering(df)
    trainer.train_pca(df)
    trainer.save_metrics()
    
    return True

@flow(name="End-to-End Stock Prediction Pipeline")
def main_pipeline(symbols: list[str] = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"]):
    """Main flow to run the entire pipeline."""
    notify_discord("🚀 Starting End-to-End Pipeline...")
    
    for symbol in symbols:
        try:
            print(f"Processing {symbol}...")
            raw_path = fetch_stock_data(symbol)
            df = process_stock_data(raw_path, symbol)
            train_and_evaluate(df, symbol)
            notify_discord(f"✅ Pipeline completed for {symbol}")
        except Exception as e:
            notify_discord(f"❌ Pipeline failed for {symbol}: {e}")
            print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    main_pipeline()
