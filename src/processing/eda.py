import pandas as pd
import matplotlib.pyplot as plt
import os

def perform_eda(file_path: str, output_dir: str = "reports/eda"):
    """
    Generates EDA plots for the given stock data.
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Close Price with SMA
    plt.figure(figsize=(14, 7))
    plt.plot(df['timestamp'], df['close'], label='Close Price')
    if 'sma_20' in df.columns:
        plt.plot(df['timestamp'], df['sma_20'], label='SMA 20')
    if 'sma_50' in df.columns:
        plt.plot(df['timestamp'], df['sma_50'], label='SMA 50')
    plt.title('Stock Price & Moving Averages')
    plt.legend()
    plt.savefig(f"{output_dir}/price_sma.png")
    plt.close()
    
    # Plot 2: RSI
    if 'rsi' in df.columns:
        plt.figure(figsize=(14, 5))
        plt.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
        plt.axhline(70, linestyle='--', color='red')
        plt.axhline(30, linestyle='--', color='green')
        plt.title('Relative Strength Index (RSI)')
        plt.legend()
        plt.savefig(f"{output_dir}/rsi.png")
        plt.close()

    print(f"EDA plots saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    # perform_eda("data/processed/AAPL_processed.csv")
    pass
