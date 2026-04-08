import os
import pickle
import warnings

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
TICKER = "^GSPC"
THRESHOLD = 0.005

ALPACA_URL = os.getenv("ALPACA_URL")
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")

HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}


def load_model_and_scaler():
    scaler = pickle.load(open(f"{DATA_DIR}/scaler.pkl", "rb"))
    model = load_model(f"{MODEL_DIR}/best_model.keras")
    return model, scaler


def get_yfinance_data(ticker=TICKER, days=90):
    stock = yf.Ticker(ticker)
    end = datetime.now()
    start = end - timedelta(days=days)
    df = stock.history(start=start, end=end)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def predict_next(model, scaler, data):
    scaled = scaler.transform(data.values)
    seq = scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 5)
    pred = model.predict(seq, verbose=0)[0, 0]

    dummy = np.zeros((1, 5))
    dummy[0, 3] = pred
    return scaler.inverse_transform(dummy)[0, 3]


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


def get_position(symbol="^GSPC"):
    resp = requests.get(f"{ALPACA_URL}/v2/positions/{symbol}", headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    return None


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


def close_position(symbol="^GSPC"):
    resp = requests.delete(f"{ALPACA_URL}/v2/positions/{symbol}", headers=HEADERS)
    return resp.json() if resp.status_code in [200, 204] else {"error": resp.text}


def log_signal(action, price, predicted, change_pct):
    log_file = os.path.join(PROJECT_DIR, "logs", "signals.log")
    os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(
            f"{timestamp} | {action} | Price: ${price:.2f} | Predicted: ${predicted:.2f} | Change: {change_pct:+.2f}%\n"
        )


def main():
    print("=" * 50)
    print("Alpaca Paper Trading Bot")
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

    model, scaler = load_model_and_scaler()

    print("\nFetching market data...")
    data = get_yfinance_data()
    current_price = data["Close"].iloc[-1]
    predicted_price = predict_next(model, scaler, data)
    change_pct = (predicted_price - current_price) / current_price * 100

    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted:    ${predicted_price:.2f} ({change_pct:+.2f}%)")

    position = get_position("^GSPC")
    shares = int(float(position["qty"])) if position else 0

    if change_pct > THRESHOLD * 100 and shares == 0:
        action = "BUY"
        qty = int(cash / current_price)
        print(f"\n>>> {action} {qty} shares of ^GSPC")
        result = place_order("^GSPC", qty, "buy")
        print(f"    Result: {result.get('status', result)}")
        log_signal(action, current_price, predicted_price, change_pct)

    elif change_pct < -THRESHOLD * 100 and shares > 0:
        action = "SELL"
        print(f"\n>>> {action} {shares} shares of ^GSPC")
        result = close_position("^GSPC")
        print(f"    Result: {result.get('status', result)}")
        log_signal(action, current_price, predicted_price, change_pct)

    else:
        action = "HOLD"
        print(f"\n>>> {action} - No clear signal")
        log_signal(action, current_price, predicted_price, change_pct)

    print(f"Position: {shares} shares")


if __name__ == "__main__":
    main()
