import os
import pickle
import warnings
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import yfinance as yf
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

env_file = os.path.join(PROJECT_DIR, ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

LOOK_BACK = 60
TICKER = "SPY"
THRESHOLD = 0.45  # Classifier threshold (45% - trade on any upward signal)

ALPACA_URL = os.getenv("ALPACA_URL")
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")

HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}


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
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    df['RSI'] = compute_rsi(close)
    
    macd, signal, hist = compute_macd(close)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['SMA_200'] = close.rolling(window=200).mean()
    
    df['Price_SMA20_Ratio'] = close / df['SMA_20']
    df['Price_SMA50_Ratio'] = close / df['SMA_50']
    
    sma, upper, lower = compute_bollinger_bands(close)
    df['BB_Upper'] = upper
    df['BB_Lower'] = lower
    df['BB_Width'] = (upper - lower) / sma
    
    df['ATR'] = compute_atr(high, low, close)
    
    df['Momentum_5'] = close / close.shift(5) - 1
    df['Momentum_10'] = close / close.shift(10) - 1
    df['Momentum_20'] = close / close.shift(20) - 1
    
    df['Volume_SMA_20'] = volume.rolling(window=20).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA_20']
    
    df['Daily_Return'] = close.pct_change()
    df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
    
    df = df.fillna(0)
    return df


def load_models():
    scaler = pickle.load(open(f"{DATA_DIR}/scaler.pkl", "rb"))
    classifier = load_model(f"{MODEL_DIR}/classifier_model.keras")
    return classifier, scaler


def get_yfinance_data(ticker=TICKER, days=120):
    stock = yf.Ticker(ticker)
    end = datetime.now()
    start = end - timedelta(days=days)
    df = stock.history(start=start, end=end)
    # Remove rows with NaN in Close price
    df = df.dropna(subset=['Close'])
    return df


def prepare_features(df):
    df = add_technical_indicators(df)
    
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
    return features


def predict_direction(classifier, scaler, data):
    scaled = scaler.transform(data.values)
    seq = scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 25)
    prob = classifier.predict(seq, verbose=0)[0, 0]
    return prob


def check_alpaca_keys():
    if not ALPACA_KEY or not ALPACA_SECRET:
        print("ERROR: Alpaca keys not set in .env")
        print("\nSet these in .env:")
        print("  ALPACA_URL=your_alpaca_url")
        print("  ALPACA_KEY=your_api_key")
        print("  ALPACA_SECRET=your_secret_key")
        print("\nGet free keys at: https://app.alpaca.markets/paper")
        return False
    return True


def get_account():
    resp = requests.get(f"{ALPACA_URL}/v2/account", headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    return None


def get_position(symbol="SPY"):
    resp = requests.get(f"{ALPACA_URL}/v2/positions/{symbol}", headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    return None


def get_market_status():
    resp = requests.get(f"{ALPACA_URL}/v2/clock", headers=HEADERS)
    if resp.status_code == 200:
        return resp.json().get("is_open", False)
    return False


def get_open_orders():
    resp = requests.get(f"{ALPACA_URL}/v2/orders?status=open", headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    return []


def place_order(symbol, qty, side):
    order = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    resp = requests.post(f"{ALPACA_URL}/v2/orders", json=order, headers=HEADERS)
    return resp.json() if resp.status_code in [200, 201] else {"error": resp.text}


def close_position(symbol="SPY"):
    resp = requests.delete(f"{ALPACA_URL}/v2/positions/{symbol}", headers=HEADERS)
    return resp.json() if resp.status_code in [200, 204] else {"error": resp.text}


def log_signal(action, price, probability, direction):
    log_file = os.path.join(PROJECT_DIR, "logs", "signals.log")
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(
            f"{timestamp} | {action} | Price: ${price:.2f} | Prob(UP): {probability:.2%} | Signal: {direction}\n"
        )


def main():
    print("=" * 50)
    print("Enhanced Alpaca Paper Trading Bot")
    print("=" * 50)

    if not check_alpaca_keys():
        return

    account = get_account()
    if account:
        cash = float(account.get("cash", 0))
        equity = float(account.get("portfolio_value", cash))
        print(f"Connected - Cash: ${cash:.2f} | Equity: ${equity:.2f}")
    else:
        print("ERROR: Could not connect to Alpaca")
        print("Check your API keys in .env")
        return

    classifier, scaler = load_models()

    print("\nFetching market data with technical indicators...")
    raw_data = get_yfinance_data()
    features = prepare_features(raw_data)
    
    current_price = raw_data['Close'].iloc[-1]
    prob_up = predict_direction(classifier, scaler, features)
    
    print(f"Current price: ${current_price:.2f}")
    print(f"Probability of UP: {prob_up:.2%}")
    print(f"Probability of DOWN: {1-prob_up:.2%}")

    position = get_position("SPY")
    shares = int(float(position["qty"])) if position else 0

    open_orders = get_open_orders()
    pending = [o for o in open_orders if o.get("symbol") == "SPY"]
    has_pending = len(pending) > 0

    print(f"Current position: {shares} shares")
    print(f"Pending orders: {len(pending)}")
    print(f"Threshold: {THRESHOLD:.0%}")

    # Check if market is open
    market_open = get_market_status()
    if not market_open:
        print("\n>>> Market closed, skipping trade")
        return

    # Trading logic using classifier
    # prob_up > 0.5 means model predicts price will go UP -> LONG
    # prob_up < 0.5 means model predicts price will go DOWN -> SHORT
    
    prob_down = 1 - prob_up
    
    # Check if we have a position and determine type (long or short)
    position_type = position.get("side", "flat") if position else "flat"
    
    if prob_up > THRESHOLD and shares == 0 and not has_pending:
        # Predicting UP with high confidence -> BUY/LONG
        action = "BUY (LONG)"
        available = cash * 0.15
        qty = int(available / current_price)
        print(f"\n>>> {action} {qty} shares of SPY")
        result = place_order("SPY", qty, "buy")
        if "error" in result:
            print(f"    FAILED: {result.get('error')}")
            action = "FAILED"
        else:
            print(f"    Result: {result.get('status', result)}")
        log_signal(action, current_price, prob_up, "UP")

    elif prob_down > THRESHOLD and shares == 0 and not has_pending:
        # Predicting DOWN with high confidence -> SHORT
        action = "SHORT"
        available = cash * 0.15
        qty = int(available / current_price)
        print(f"\n>>> {action} {qty} shares of SPY")
        result = place_order("SPY", qty, "buy")
        if "error" in result:
            print(f"    FAILED: {result.get('error')}")
            action = "FAILED"
        else:
            print(f"    Result: {result.get('status', result)}")
        log_signal(action, current_price, prob_up, "DOWN")

    elif prob_up > THRESHOLD and shares > 0 and position_type == "short":
        # Have a short position and predicting UP -> COVER
        action = "COVER SHORT"
        print(f"\n>>> {action} {shares} shares of SPY")
        result = close_position("SPY")
        print(f"    Result: {result.get('status', result)}")
        log_signal(action, current_price, prob_up, "UP")

    elif prob_down > THRESHOLD and shares > 0 and position_type == "long":
        # Have a long position and predicting DOWN -> SELL
        action = "SELL (STOP LOSS)"
        print(f"\n>>> {action} {shares} shares of SPY")
        result = close_position("SPY")
        print(f"    Result: {result.get('status', result)}")
        log_signal(action, current_price, prob_up, "DOWN")

    else:
        direction = "UP" if prob_up > 0.5 else "DOWN"
        action = "HOLD"
        print(f"\n>>> {action} - No clear signal (predicted: {direction})")
        log_signal(action, current_price, prob_up, direction)

    print(f"Position: {shares} shares")


if __name__ == "__main__":
    main()