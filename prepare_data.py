import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS = ["^GSPC"]
PERIOD = "5y"
LOOK_BACK = 60

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
    combined_df.to_csv(f"{DATA_DIR}/raw_data.csv")
    print(f"Saved {len(combined_df)} rows to {DATA_DIR}/raw_data.csv")
    return combined_df

def feature_selection(df):
    print("\nFeature Selection:")
    print("  Target: Next day's Close price")
    print("  Input features: Open, High, Low, Close, Volume")
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

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

def create_sequences(data, look_back=LOOK_BACK):
    print(f"\nCreating sequences with look-back window of {look_back}...")
    X, y = [], []
    
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i, 3])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"  X shape: {X.shape} (samples, time_steps, features)")
    print(f"  y shape: {y.shape}")
    
    return X, y

def train_test_split(X, y, train_ratio=0.8):
    print(f"\nSplitting data with train ratio {train_ratio}...")
    train_size = int(len(X) * train_ratio)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def main():
    raw_df = download_data()
    
    features_df = feature_selection(raw_df)
    
    scaled_data, scaler = normalize_data(features_df)
    
    X, y = create_sequences(scaled_data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    np.save(f"{DATA_DIR}/X_train.npy", X_train)
    np.save(f"{DATA_DIR}/X_test.npy", X_test)
    np.save(f"{DATA_DIR}/y_train.npy", y_train)
    np.save(f"{DATA_DIR}/y_test.npy", y_test)
    
    print(f"  Data saved to {DATA_DIR}/")
    print(f"  Look-back window: {LOOK_BACK}")

if __name__ == "__main__":
    main()