import argparse
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Add project directory to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
import config

DATA_DIR = config.DATA_DIR
os.makedirs(DATA_DIR, exist_ok=True)

# Default tickers - can be overridden by command line
TICKERS = ["SPY"]
PERIOD = "10y"  # Increased from 5y to 10y
LOOK_BACK = 60


def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def compute_bollinger_bands(prices, period=20):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return sma, upper, lower


def compute_atr(high, low, close, period=14):
    tr = np.maximum(
        high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
    )
    atr = tr.rolling(window=period).mean()
    return atr


def add_technical_indicators(df):
    print("\nAdding technical indicators...")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # RSI
    df["RSI"] = compute_rsi(close)

    # MACD
    macd, signal, hist = compute_macd(close)
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Hist"] = hist

    # Moving Averages
    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()
    df["SMA_200"] = close.rolling(window=200).mean()

    # Price relative to MAs
    df["Price_SMA20_Ratio"] = close / df["SMA_20"]
    df["Price_SMA50_Ratio"] = close / df["SMA_50"]

    # Bollinger Bands
    sma, upper, lower = compute_bollinger_bands(close)
    df["BB_Upper"] = upper
    df["BB_Lower"] = lower
    df["BB_Width"] = (upper - lower) / sma

    # ATR
    df["ATR"] = compute_atr(high, low, close)

    # Momentum
    df["Momentum_5"] = close / close.shift(5) - 1
    df["Momentum_10"] = close / close.shift(10) - 1
    df["Momentum_20"] = close / close.shift(20) - 1

    # Volume indicators
    df["Volume_SMA_20"] = volume.rolling(window=20).mean()
    df["Volume_Ratio"] = volume / df["Volume_SMA_20"]

    # Price changes
    df["Daily_Return"] = close.pct_change()
    df["Volatility_10"] = df["Daily_Return"].rolling(window=10).std()
    df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

    # Fill NaN with 0 for indicators that can't be computed at start
    df = df.fillna(0)

    print(f"  Added features: RSI, MACD, SMAs, Bollinger, ATR, Momentum, Volume")
    return df


def download_data():
    print("Downloading historical data...")
    data_frames = []

    for ticker in TICKERS:
        print(f"  Fetching {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(period=PERIOD)
        df["Ticker"] = ticker
        data_frames.append(df)

    combined_df = pd.concat(data_frames)
    combined_df = combined_df.dropna()
    combined_df.to_csv(f"{DATA_DIR}/raw_data.csv")
    print(f"Saved {len(combined_df)} rows to {DATA_DIR}/raw_data.csv")
    return combined_df


def feature_selection(df):
    print("\nFeature Selection:")
    print("  Features: OHLCV + Technical Indicators")

    # Select features
    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "SMA_20",
        "SMA_50",
        "SMA_200",
        "Price_SMA20_Ratio",
        "Price_SMA50_Ratio",
        "BB_Upper",
        "BB_Lower",
        "BB_Width",
        "ATR",
        "Momentum_5",
        "Momentum_10",
        "Momentum_20",
        "Volume_Ratio",
        "Daily_Return",
        "Volatility_10",
        "Volatility_20",
    ]

    features = df[feature_cols]

    # Create directional labels (1 = next day close > today close, 0 = otherwise)
    labels = (df["Close"].shift(-1) > df["Close"]).astype(int)

    print(f"  Total features: {len(feature_cols)}")
    print(f"  Target: Next day direction (up=1, down=0)")

    return features, labels


def normalize_data(data):
    print("\nApplying Min-Max Scaling...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    scaler_filename = f"{DATA_DIR}/scaler.pkl"
    import pickle

    with open(scaler_filename, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_filename}")

    return scaled_data, scaler


def create_sequences(data, labels, look_back=LOOK_BACK):
    print(f"\nCreating sequences with look-back window of {look_back}...")
    X, y = [], []
    y_direction = []

    for i in range(look_back, len(data) - 1):
        window = data[i - look_back : i]
        target = data[i, 3]  # Close price
        direction = labels.iloc[i] if hasattr(labels, "iloc") else labels[i]

        if np.isnan(window).any() or np.isnan(target):
            continue

        X.append(window)
        y.append(target)
        y_direction.append(direction)

    X = np.array(X)
    y = np.array(y)
    y_direction = np.array(y_direction)

    print(f"  X shape: {X.shape} (samples, time_steps, features)")
    print(f"  y shape: {y.shape}")
    print(
        f"  Direction labels: {len(y_direction)}, up={sum(y_direction)}, down={len(y_direction) - sum(y_direction)}"
    )

    return X, y, y_direction


def create_sequences_flat(data, labels, look_back=LOOK_BACK):
    """Flat sequences for XGBoost (not 3D)"""
    print(f"\nCreating flat sequences for XGBoost...")
    X, y, y_direction = create_sequences(data, labels, look_back)

    # Flatten: (samples, timesteps, features) -> (samples, timesteps*features)
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)

    print(f"  X_flat shape: {X_flat.shape}")

    return X_flat, y, y_direction


def train_test_split(X, y, y_direction, train_ratio=0.8):
    print(f"\nSplitting data with train ratio {train_ratio}...")
    train_size = int(len(X) * train_ratio)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    y_dir_train, y_dir_test = y_direction[:train_size], y_direction[train_size:]

    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, y_dir_train, y_dir_test


def save_data(X, y, y_direction, X_flat, X_train_size):
    """Save data files with ticker-specific naming"""
    ticker_dir = config.get_data_dir(TICKERS[0])
    os.makedirs(ticker_dir, exist_ok=True)

    # Save price-based labels
    np.save(f"{ticker_dir}/X_train.npy", X[:X_train_size])
    np.save(f"{ticker_dir}/X_test.npy", X[X_train_size:])
    np.save(f"{ticker_dir}/y_train.npy", y[:X_train_size])
    np.save(f"{ticker_dir}/y_test.npy", y[X_train_size:])

    # Save directional labels for classifier
    np.save(f"{ticker_dir}/y_dir_train.npy", y_direction[:X_train_size])
    np.save(f"{ticker_dir}/y_dir_test.npy", y_direction[X_train_size:])

    # Also save flat versions for XGBoost
    X_flat_train = X_flat[:X_train_size]
    X_flat_test = X_flat[X_train_size:]
    np.save(f"{ticker_dir}/X_train_flat.npy", X_flat_train)
    np.save(f"{ticker_dir}/X_test_flat.npy", X_flat_test)

    print(f"\n  Data saved to {ticker_dir}/")
    print(f"  Look-back window: {LOOK_BACK}")
    print(f"  Files: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
    print(f"         y_dir_train.npy, y_dir_test.npy (directional)")
    print(f"         X_train_flat.npy, X_test_flat.npy (for XGBoost)")


def download_and_process_ticker(ticker, period=PERIOD, look_back=LOOK_BACK):
    """Download and process data for a single ticker"""
    print(f"\n{'=' * 60}")
    print(f"Processing ticker: {ticker}")
    print(f"{'=' * 60}")

    # Create ticker-specific data directory
    ticker_dir = config.get_data_dir(ticker)
    os.makedirs(ticker_dir, exist_ok=True)

    # Download data
    print(f"Downloading historical data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if len(df) == 0:
        print(f"  ERROR: No data returned for {ticker}")
        return None

    df = df.dropna()

    # Add technical indicators
    df = add_technical_indicators(df)

    # Save raw data
    raw_data_path = f"{ticker_dir}/raw_data.csv"
    df.to_csv(raw_data_path)
    print(f"  Saved {len(df)} rows to {raw_data_path}")

    # Feature selection
    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "SMA_20",
        "SMA_50",
        "SMA_200",
        "Price_SMA20_Ratio",
        "Price_SMA50_Ratio",
        "BB_Upper",
        "BB_Lower",
        "BB_Width",
        "ATR",
        "Momentum_5",
        "Momentum_10",
        "Momentum_20",
        "Volume_Ratio",
        "Daily_Return",
        "Volatility_10",
        "Volatility_20",
    ]
    features_df = df[feature_cols]

    # Create directional labels
    labels = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Normalize data
    print("  Applying Min-Max Scaling...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features_df)

    # Save scaler
    scaler_path = f"{ticker_dir}/scaler.pkl"
    import pickle

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler to {scaler_path}")

    # Create sequences
    print(f"  Creating sequences with look-back window of {look_back}...")
    X, y, y_direction = [], [], []

    for i in range(look_back, len(scaled_data) - 1):
        window = scaled_data[i - look_back : i]
        target = scaled_data[i, 3]  # Close price
        direction = labels.iloc[i] if hasattr(labels, "iloc") else labels[i]

        if np.isnan(window).any() or np.isnan(target):
            continue

        X.append(window)
        y.append(target)
        y_direction.append(direction)

    X = np.array(X)
    y = np.array(y)
    y_direction = np.array(y_direction)

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(
        f"  Direction labels: up={sum(y_direction)}, down={len(y_direction) - sum(y_direction)}"
    )

    # Flat sequences for XGBoost
    X_flat = X.reshape(X.shape[0], -1)
    print(f"  X_flat shape: {X_flat.shape}")

    # Train/test split
    train_ratio = 0.8
    train_size = int(len(X) * train_ratio)

    return X, y, y_direction, X_flat, train_size, ticker


def main():
    parser = argparse.ArgumentParser(description="Prepare data for trading bot")
    parser.add_argument(
        "--ticker", type=str, default=None, help="Specific ticker to process"
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all tickers from config"
    )
    args = parser.parse_args()

    # Determine which tickers to process
    if args.all:
        tickers = config.TICKERS
    elif args.ticker:
        tickers = [args.ticker]
    else:
        tickers = [
            config.DEFAULT_TICKER if hasattr(config, "DEFAULT_TICKER") else "SPY"
        ]

    # Save tickers globally for use in save_data
    global TICKERS
    TICKERS = tickers

    print(f"\n{'#' * 60}")
    print(f"# MrKrabs Data Preparation")
    print(f"# Tickers: {tickers}")
    print(f"{'#' * 60}")

    # Process each ticker
    for ticker in tickers:
        TICKERS = [ticker]
        result = download_and_process_ticker(ticker)

        if result is None:
            print(f"  ERROR: Failed to process {ticker}")
            continue

        X, y, y_direction, X_flat, train_size, _ = result

        # Save data
        save_data(X, y, y_direction, X_flat, train_size)

    print(f"\n\n{'#' * 60}")
    print(f"# Data Preparation Complete")
    print(f"# Processed tickers: {tickers}")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
