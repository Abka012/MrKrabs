import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

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
    tr = np.maximum(high - low, 
                    np.maximum(abs(high - close.shift(1)), 
                               abs(low - close.shift(1))))
    atr = tr.rolling(window=period).mean()
    return atr


def add_technical_indicators(df):
    print("\nAdding technical indicators...")
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # RSI
    df['RSI'] = compute_rsi(close)
    
    # MACD
    macd, signal, hist = compute_macd(close)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    
    # Moving Averages
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['SMA_200'] = close.rolling(window=200).mean()
    
    # Price relative to MAs
    df['Price_SMA20_Ratio'] = close / df['SMA_20']
    df['Price_SMA50_Ratio'] = close / df['SMA_50']
    
    # Bollinger Bands
    sma, upper, lower = compute_bollinger_bands(close)
    df['BB_Upper'] = upper
    df['BB_Lower'] = lower
    df['BB_Width'] = (upper - lower) / sma
    
    # ATR
    df['ATR'] = compute_atr(high, low, close)
    
    # Momentum
    df['Momentum_5'] = close / close.shift(5) - 1
    df['Momentum_10'] = close / close.shift(10) - 1
    df['Momentum_20'] = close / close.shift(20) - 1
    
    # Volume indicators
    df['Volume_SMA_20'] = volume.rolling(window=20).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA_20']
    
    # Price changes
    df['Daily_Return'] = close.pct_change()
    df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
    
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
        df['Ticker'] = ticker
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
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'SMA_20', 'SMA_50', 'SMA_200',
        'Price_SMA20_Ratio', 'Price_SMA50_Ratio',
        'BB_Upper', 'BB_Lower', 'BB_Width',
        'ATR', 'Momentum_5', 'Momentum_10', 'Momentum_20',
        'Volume_Ratio', 'Daily_Return', 'Volatility_10', 'Volatility_20'
    ]
    
    features = df[feature_cols]
    
    # Create directional labels (1 = next day close > today close, 0 = otherwise)
    labels = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Target: Next day direction (up=1, down=0)")
    
    return features, labels


def normalize_data(data):
    print("\nApplying Min-Max Scaling...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    scaler_filename = f"{DATA_DIR}/scaler.pkl"
    import pickle
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_filename}")
    
    return scaled_data, scaler


def create_sequences(data, labels, look_back=LOOK_BACK):
    print(f"\nCreating sequences with look-back window of {look_back}...")
    X, y = [], []
    y_direction = []
    
    for i in range(look_back, len(data) - 1):
        window = data[i - look_back:i]
        target = data[i, 3]  # Close price
        direction = labels.iloc[i] if hasattr(labels, 'iloc') else labels[i]
        
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
    print(f"  Direction labels: {len(y_direction)}, up={sum(y_direction)}, down={len(y_direction)-sum(y_direction)}")
    
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


def main():
    raw_df = download_data()
    
    raw_df = add_technical_indicators(raw_df)
    
    features_df, labels = feature_selection(raw_df)
    
    scaled_data, scaler = normalize_data(features_df)
    
    X, y, y_direction = create_sequences(scaled_data, labels)
    
    X_train, X_test, y_train, y_test, y_dir_train, y_dir_test = train_test_split(X, y, y_direction)
    
    # Save price-based labels
    np.save(f"{DATA_DIR}/X_train.npy", X_train)
    np.save(f"{DATA_DIR}/X_test.npy", X_test)
    np.save(f"{DATA_DIR}/y_train.npy", y_train)
    np.save(f"{DATA_DIR}/y_test.npy", y_test)
    
    # Save directional labels for classifier
    np.save(f"{DATA_DIR}/y_dir_train.npy", y_dir_train)
    np.save(f"{DATA_DIR}/y_dir_test.npy", y_dir_test)
    
    # Also save flat versions for XGBoost
    X_flat, _, _ = create_sequences_flat(scaled_data, labels)
    X_flat_train = X_flat[:len(X_train)]
    X_flat_test = X_flat[len(X_train):]
    
    np.save(f"{DATA_DIR}/X_train_flat.npy", X_flat_train)
    np.save(f"{DATA_DIR}/X_test_flat.npy", X_flat_test)
    
    print(f"\n  Data saved to {DATA_DIR}/")
    print(f"  Look-back window: {LOOK_BACK}")
    print(f"  Files: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
    print(f"         y_dir_train.npy, y_dir_test.npy (directional)")
    print(f"         X_train_flat.npy, X_test_flat.npy (for XGBoost)")


if __name__ == "__main__":
    main()