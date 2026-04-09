import argparse
import json
import os
import pickle
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import yfinance as yf
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
import config

DATA_DIR = config.DATA_DIR
MODEL_DIR = config.MODEL_DIR

env_file = os.path.join(PROJECT_DIR, ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

LOOK_BACK = 60
# TICKER is now handled via config and command line argument
THRESHOLD = config.THRESHOLD  # Classifier threshold
POSITION_SIZE = getattr(config, "POSITION_SIZE", 0.15)
LONG_ENTRY_THRESHOLD = getattr(config, "LONG_ENTRY_THRESHOLD", 0.58)
SHORT_ENTRY_THRESHOLD = getattr(config, "SHORT_ENTRY_THRESHOLD", 0.42)
MIN_CONFIDENCE_GAP = getattr(config, "MIN_CONFIDENCE_GAP", 0.08)
TRADE_MODE = getattr(config, "TRADE_MODE", "equity").lower()
ALLOW_SHORTS = getattr(config, "ALLOW_SHORTS", True)
USE_TREND_FILTER = getattr(config, "USE_TREND_FILTER", True)
OPTIONS_POSITION_SIZE = getattr(config, "OPTIONS_POSITION_SIZE", 0.05)
OPTIONS_ENABLED_UNDERLYINGS = set(
    getattr(config, "OPTIONS_ENABLED_UNDERLYINGS", config.TICKERS)
)
OPTIONS_MIN_DTE = getattr(config, "OPTIONS_MIN_DTE", 7)
OPTIONS_MAX_DTE = getattr(config, "OPTIONS_MAX_DTE", 45)
OPTIONS_STRIKE_WINDOW = getattr(config, "OPTIONS_STRIKE_WINDOW", 0.08)
AUTO_MODE_MIN_CONFIDENCE_GAP = getattr(config, "AUTO_MODE_MIN_CONFIDENCE_GAP", 0.05)
AUTO_MODE_ATR_MULTIPLIER = getattr(config, "AUTO_MODE_ATR_MULTIPLIER", 1.0)
AUTO_MODE_OPTION_COST_PENALTY = getattr(config, "AUTO_MODE_OPTION_COST_PENALTY", 0.015)
AUTO_MODE_MAX_OPTION_LEVERAGE = getattr(config, "AUTO_MODE_MAX_OPTION_LEVERAGE", 4.0)
MODE_SELECTOR_FILENAME = "mode_selector.pkl"

ALPACA_URL = os.getenv("ALPACA_URL")
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")

HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}
NON_EXECUTING_ORDER_STATUSES = {"canceled", "expired", "rejected"}


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
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["RSI"] = compute_rsi(close)

    macd, signal, hist = compute_macd(close)
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Hist"] = hist

    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()
    df["SMA_200"] = close.rolling(window=200).mean()

    df["Price_SMA20_Ratio"] = close / df["SMA_20"]
    df["Price_SMA50_Ratio"] = close / df["SMA_50"]

    sma, upper, lower = compute_bollinger_bands(close)
    df["BB_Upper"] = upper
    df["BB_Lower"] = lower
    df["BB_Width"] = (upper - lower) / sma

    df["ATR"] = compute_atr(high, low, close)

    df["Momentum_5"] = close / close.shift(5) - 1
    df["Momentum_10"] = close / close.shift(10) - 1
    df["Momentum_20"] = close / close.shift(20) - 1

    df["Volume_SMA_20"] = volume.rolling(window=20).mean()
    df["Volume_Ratio"] = volume / df["Volume_SMA_20"]

    df["Daily_Return"] = close.pct_change()
    df["Volatility_10"] = df["Daily_Return"].rolling(window=10).std()
    df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

    df = df.fillna(0)
    return df


def load_models(ticker):
    scaler = pickle.load(open(f"{config.get_data_dir(ticker)}/scaler.pkl", "rb"))
    classifier = load_model(f"{config.get_model_dir(ticker)}/classifier_model.keras")
    return classifier, scaler


def get_yfinance_data(ticker, days=120):
    stock = yf.Ticker(ticker)
    end = datetime.now()
    start = end - timedelta(days=days)
    df = stock.history(start=start, end=end)
    # Remove rows with NaN in Close price
    df = df.dropna(subset=["Close"])
    return df


def prepare_features(df):
    df = add_technical_indicators(df)

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


def ticker_print(ticker, message):
    print(f"[{ticker}] {message}")


def get_account():
    resp = requests.get(f"{ALPACA_URL}/v2/account", headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    return None


def get_asset(symbol):
    resp = requests.get(f"{ALPACA_URL}/v2/assets/{symbol}", headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    return None


def get_position(symbol="SPY"):
    resp = requests.get(f"{ALPACA_URL}/v2/positions/{symbol}", headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    return None


def get_all_positions():
    resp = requests.get(f"{ALPACA_URL}/v2/positions", headers=HEADERS)
    if resp.status_code == 200:
        return resp.json()
    return []


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


def get_orders(status="all", limit=500, after=None):
    params = {"status": status, "limit": limit, "direction": "desc", "nested": "true"}
    if after:
        params["after"] = after
    resp = requests.get(f"{ALPACA_URL}/v2/orders", headers=HEADERS, params=params)
    if resp.status_code == 200:
        return resp.json()
    return []


def place_order(symbol, qty, side, asset_class="us_equity"):
    order = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    if asset_class == "option":
        order["order_class"] = "simple"
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


def get_option_contract(symbol_or_id):
    resp = requests.get(
        f"{ALPACA_URL}/v2/options/contracts/{symbol_or_id}", headers=HEADERS
    )
    if resp.status_code == 200:
        return resp.json()
    return None


def get_today_order_window_start():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")


def order_belongs_to_ticker(order, ticker, option_contract_cache):
    if order.get("symbol") == ticker:
        return True

    if order.get("asset_class") != "option":
        return False

    contract_symbol = order.get("symbol")
    if not contract_symbol:
        return False

    if contract_symbol not in option_contract_cache:
        option_contract_cache[contract_symbol] = get_option_contract(contract_symbol)

    contract = option_contract_cache.get(contract_symbol)
    return bool(contract and contract.get("underlying_symbol") == ticker)


def has_traded_today(ticker):
    option_contract_cache = {}
    today_orders = get_orders(after=get_today_order_window_start())

    for order in today_orders:
        if order.get("status") in NON_EXECUTING_ORDER_STATUSES:
            continue
        if order_belongs_to_ticker(order, ticker, option_contract_cache):
            return True

    return False


def list_option_contracts(ticker, contract_type, current_price):
    start_date = (datetime.now() + timedelta(days=OPTIONS_MIN_DTE)).date().isoformat()
    end_date = (datetime.now() + timedelta(days=OPTIONS_MAX_DTE)).date().isoformat()
    strike_min = round(current_price * (1 - OPTIONS_STRIKE_WINDOW), 2)
    strike_max = round(current_price * (1 + OPTIONS_STRIKE_WINDOW), 2)
    params = {
        "underlying_symbols": ticker,
        "type": contract_type,
        "expiration_date_gte": start_date,
        "expiration_date_lte": end_date,
        "strike_price_gte": strike_min,
        "strike_price_lte": strike_max,
        "status": "active",
        "limit": 100,
    }
    resp = requests.get(
        f"{ALPACA_URL}/v2/options/contracts", headers=HEADERS, params=params
    )
    if resp.status_code != 200:
        return []
    payload = resp.json()
    return payload.get("option_contracts", payload.get("contracts", payload))


def select_option_contract(ticker, contract_type, current_price):
    contracts = list_option_contracts(ticker, contract_type, current_price)
    valid_contracts = []
    for contract in contracts:
        close_price = float(contract.get("close_price") or 0)
        strike_price = float(contract.get("strike_price") or 0)
        expiration_date = contract.get("expiration_date")
        if close_price <= 0 or strike_price <= 0 or not expiration_date:
            continue
        dte = (datetime.fromisoformat(expiration_date) - datetime.now()).days
        if dte < OPTIONS_MIN_DTE or dte > OPTIONS_MAX_DTE:
            continue
        valid_contracts.append(contract)

    if not valid_contracts:
        return None

    return min(
        valid_contracts,
        key=lambda contract: (
            abs(
                (
                    datetime.fromisoformat(contract["expiration_date"]) - datetime.now()
                ).days
                - OPTIONS_MIN_DTE
            ),
            abs(float(contract["strike_price"]) - current_price),
        ),
    )


def estimate_expected_move_pct(features, current_price, confidence_gap):
    atr = float(features["ATR"].iloc[-1]) if "ATR" in features.columns else 0
    atr_move_pct = (atr / current_price) if current_price > 0 else 0
    confidence_scale = max(confidence_gap, 0)
    return max(atr_move_pct * AUTO_MODE_ATR_MULTIPLIER * confidence_scale * 2, 0)


def load_mode_selector(ticker):
    selector_path = os.path.join(config.get_model_dir(ticker), MODE_SELECTOR_FILENAME)
    if not os.path.exists(selector_path):
        return None
    with open(selector_path, "rb") as f:
        return pickle.load(f)


def build_live_selector_features(features, current_price, prob_up):
    latest = features.iloc[-1]
    feature_map = {
        "prob_up": prob_up,
        "prob_down": 1 - prob_up,
        "confidence_gap": abs(prob_up - 0.5),
        "current_price": current_price,
        "atr_pct": float(latest.get("ATR", 0) / current_price) if current_price > 0 else 0,
        "volatility_10": float(latest.get("Volatility_10", 0) or 0),
        "volatility_20": float(latest.get("Volatility_20", 0) or 0),
        "rsi": float(latest.get("RSI", 0) or 0),
        "macd_hist": float(latest.get("MACD_Hist", 0) or 0),
        "momentum_5": float(latest.get("Momentum_5", 0) or 0),
        "momentum_10": float(latest.get("Momentum_10", 0) or 0),
        "bb_width": float(latest.get("BB_Width", 0) or 0),
    }
    return feature_map


def estimate_equity_edge(prob_up, expected_move_pct):
    confidence_gap = abs(prob_up - 0.5)
    return expected_move_pct * confidence_gap * 2


def estimate_option_edge(contract, current_price, expected_move_pct):
    premium = float(contract.get("close_price") or 0)
    strike_price = float(contract.get("strike_price") or current_price)
    if premium <= 0 or current_price <= 0:
        return float("-inf")

    premium_pct = premium / current_price
    leverage = min(1 / max(premium_pct, 0.01), AUTO_MODE_MAX_OPTION_LEVERAGE)
    moneyness_penalty = abs(strike_price - current_price) / current_price
    return (
        expected_move_pct * leverage
        - premium_pct
        - moneyness_penalty
        - AUTO_MODE_OPTION_COST_PENALTY
    )


def choose_trade_mode(ticker, account, current_price, prob_up, features):
    selector_artifact = load_mode_selector(ticker)
    confidence_gap = abs(prob_up - 0.5)
    direction = "UP" if prob_up > 0.5 else "DOWN"
    expected_move_pct = estimate_expected_move_pct(features, current_price, confidence_gap)
    signal = evaluate_signal(features, current_price, prob_up)
    signal_is_actionable = signal["bullish_signal"] or signal["bearish_signal"]

    if not signal_is_actionable:
        return {
            "mode": "hold",
            "reason": describe_signal_rejection(signal, prob_up),
            "contract": None,
            "expected_move_pct": expected_move_pct,
            "equity_edge": None,
            "option_edge": None,
            "direction": direction,
        }

    if selector_artifact:
        feature_map = build_live_selector_features(features, current_price, prob_up)
        feature_columns = selector_artifact["feature_columns"]
        X = pd.DataFrame([[feature_map[col] for col in feature_columns]], columns=feature_columns)
        predicted_label = int(selector_artifact["model"].predict(X)[0])
        predicted_mode = "option" if predicted_label == 1 else "equity"

        if predicted_mode == "option":
            if ticker not in OPTIONS_ENABLED_UNDERLYINGS or not account_supports_options(
                account
            ):
                predicted_mode = "equity"
                reason = "selector preferred option, but options are unavailable; using equity"
                selected_contract = None
            else:
                contract_type = "call" if prob_up > 0.5 else "put"
                selected_contract = select_option_contract(ticker, contract_type, current_price)
                if not selected_contract:
                    predicted_mode = "equity"
                    selected_contract = None
                    reason = "selector preferred option, but no suitable live contract was found"
                else:
                    reason = (
                        "learned selector chose option "
                        f"(training accuracy {selector_artifact['metrics']['training_accuracy']:.2%})"
                    )
        else:
            selected_contract = None
            reason = (
                "learned selector chose equity "
                f"(training accuracy {selector_artifact['metrics']['training_accuracy']:.2%})"
            )

        return {
            "mode": predicted_mode,
            "reason": reason,
            "contract": selected_contract,
            "expected_move_pct": expected_move_pct,
            "equity_edge": None,
            "option_edge": None,
            "direction": direction,
        }

    if confidence_gap < AUTO_MODE_MIN_CONFIDENCE_GAP:
        return {
            "mode": "equity",
            "reason": (
                f"confidence gap {confidence_gap:.2%} below auto threshold; "
                "defaulting to equity"
            ),
            "contract": None,
            "expected_move_pct": expected_move_pct,
            "equity_edge": estimate_equity_edge(prob_up, expected_move_pct),
            "option_edge": float("-inf"),
            "direction": direction,
        }

    equity_edge = estimate_equity_edge(prob_up, expected_move_pct)
    option_edge = float("-inf")
    selected_contract = None

    if ticker in OPTIONS_ENABLED_UNDERLYINGS and account_supports_options(account):
        contract_type = "call" if prob_up > 0.5 else "put"
        selected_contract = select_option_contract(ticker, contract_type, current_price)
        if selected_contract:
            option_edge = estimate_option_edge(
                selected_contract, current_price, expected_move_pct
            )

    chosen_mode = "option" if option_edge > equity_edge else "equity"
    reason = (
        f"estimated option edge {option_edge:.4f} vs equity edge {equity_edge:.4f}"
        if selected_contract
        else "no suitable option contract or account approval; using equity"
    )
    return {
        "mode": chosen_mode,
        "reason": reason,
        "contract": selected_contract,
        "expected_move_pct": expected_move_pct,
        "equity_edge": equity_edge,
        "option_edge": option_edge,
        "direction": direction,
    }


def find_option_position_for_underlying(ticker):
    for position in get_all_positions():
        if position.get("asset_class") != "option":
            continue
        contract = get_option_contract(position["symbol"])
        if contract and contract.get("underlying_symbol") == ticker:
            return position, contract
    return None, None


def has_pending_option_order(ticker):
    for order in get_open_orders():
        if order.get("asset_class") != "option":
            continue
        contract = get_option_contract(order["symbol"])
        if contract and contract.get("underlying_symbol") == ticker:
            return True
    return False


def account_supports_options(account):
    approved_level = int(account.get("options_approved_level") or 0)
    trading_level = int(account.get("options_trading_level") or 0)
    return approved_level >= 1 and trading_level >= 1


def get_equity_trade_context(ticker, cash):
    asset = get_asset(ticker)
    position = get_position(ticker)
    shares = int(float(position["qty"])) if position else 0
    open_orders = get_open_orders()
    pending = [o for o in open_orders if o.get("symbol") == ticker]

    is_shortable = bool(asset and asset.get("shortable"))
    is_easy_to_borrow = bool(asset and asset.get("easy_to_borrow"))
    max_notional = cash * POSITION_SIZE

    return {
        "asset": asset,
        "position": position,
        "shares": shares,
        "pending_count": len(pending),
        "has_pending": len(pending) > 0,
        "is_shortable": is_shortable,
        "is_easy_to_borrow": is_easy_to_borrow,
        "max_notional": max_notional,
    }


def evaluate_signal(features, current_price, prob_up):
    latest = features.iloc[-1]
    prob_down = 1 - prob_up
    confidence_gap = abs(prob_up - 0.5)
    bullish_trend = (
        current_price > float(latest.get("SMA_20", current_price))
        and float(latest.get("SMA_20", current_price))
        >= float(latest.get("SMA_50", current_price))
    )
    bearish_trend = (
        current_price < float(latest.get("SMA_20", current_price))
        and float(latest.get("SMA_20", current_price))
        <= float(latest.get("SMA_50", current_price))
    )

    bullish_signal = (
        prob_up >= LONG_ENTRY_THRESHOLD
        and confidence_gap >= MIN_CONFIDENCE_GAP
        and (bullish_trend or not USE_TREND_FILTER)
    )
    bearish_signal = (
        prob_up <= SHORT_ENTRY_THRESHOLD
        and confidence_gap >= MIN_CONFIDENCE_GAP
        and (bearish_trend or not USE_TREND_FILTER)
    )

    return {
        "prob_down": prob_down,
        "confidence_gap": confidence_gap,
        "bullish_trend": bullish_trend,
        "bearish_trend": bearish_trend,
        "bullish_signal": bullish_signal,
        "bearish_signal": bearish_signal,
    }


def describe_signal_rejection(signal, prob_up):
    reasons = []
    if prob_up > 0.5:
        if prob_up < LONG_ENTRY_THRESHOLD:
            reasons.append(
                f"prob_up {prob_up:.2%} below long threshold {LONG_ENTRY_THRESHOLD:.0%}"
            )
        if signal["confidence_gap"] < MIN_CONFIDENCE_GAP:
            reasons.append(
                f"confidence gap {signal['confidence_gap']:.2%} below minimum {MIN_CONFIDENCE_GAP:.0%}"
            )
        if USE_TREND_FILTER and not signal["bullish_trend"]:
            reasons.append("bullish trend filter not aligned")
    else:
        if prob_up > SHORT_ENTRY_THRESHOLD:
            reasons.append(
                f"prob_up {prob_up:.2%} above short threshold {SHORT_ENTRY_THRESHOLD:.0%}"
            )
        if signal["confidence_gap"] < MIN_CONFIDENCE_GAP:
            reasons.append(
                f"confidence gap {signal['confidence_gap']:.2%} below minimum {MIN_CONFIDENCE_GAP:.0%}"
            )
        if USE_TREND_FILTER and not signal["bearish_trend"]:
            reasons.append("bearish trend filter not aligned")

    return "; ".join(reasons) if reasons else "signal did not meet execution criteria"


def trade_equity(ticker, account, current_price, prob_up, features):
    cash = float(account.get("cash", 0))
    context = get_equity_trade_context(ticker, cash)
    shares = context["shares"]
    position = context["position"]
    signal = evaluate_signal(features, current_price, prob_up)

    ticker_print(ticker, f"Current position: {shares} shares")
    ticker_print(ticker, f"Pending orders: {context['pending_count']}")
    ticker_print(
        ticker,
        f"Long/Short thresholds: {LONG_ENTRY_THRESHOLD:.0%}/{SHORT_ENTRY_THRESHOLD:.0%}",
    )
    ticker_print(ticker, f"Min confidence gap: {MIN_CONFIDENCE_GAP:.0%}")
    ticker_print(ticker, f"Shortable: {context['is_shortable']}")
    ticker_print(ticker, f"Easy to borrow: {context['is_easy_to_borrow']}")
    ticker_print(
        ticker,
        f"Trend filter: bullish={signal['bullish_trend']} bearish={signal['bearish_trend']}",
    )

    market_open = get_market_status()
    if not market_open:
        ticker_print(ticker, ">>> HOLD - Market closed, skipping trade")
        log_signal("HOLD", current_price, prob_up, "UP" if prob_up > 0.5 else "DOWN")
        return

    prob_down = signal["prob_down"]
    position_type = position.get("side", "flat") if position else "flat"

    if signal["bullish_signal"] and shares == 0 and not context["has_pending"]:
        action = "BUY (LONG)"
        qty = int(context["max_notional"] / current_price)
        if qty <= 0:
            ticker_print(ticker, ">>> HOLD - Not enough buying power for equity long")
            log_signal("HOLD", current_price, prob_up, "UP")
            return
        ticker_print(ticker, f">>> {action} {qty} shares of {ticker}")
        result = place_order(ticker, qty, "buy")
        if "error" in result:
            ticker_print(ticker, f"FAILED: {result.get('error')}")
            action = "FAILED"
        else:
            ticker_print(ticker, f"Result: {result.get('status', result)}")
        log_signal(action, current_price, prob_up, "UP")

    elif signal["bearish_signal"] and shares == 0 and not context["has_pending"]:
        if not ALLOW_SHORTS:
            ticker_print(ticker, ">>> HOLD - Short selling disabled in config")
            log_signal("HOLD", current_price, prob_up, "DOWN")
            return
        if not context["is_shortable"] or not context["is_easy_to_borrow"]:
            ticker_print(ticker, ">>> HOLD - Asset is not shortable or not easy to borrow")
            log_signal("HOLD", current_price, prob_up, "DOWN")
            return
        action = "SHORT"
        qty = int(context["max_notional"] / current_price)
        if qty <= 0:
            ticker_print(ticker, ">>> HOLD - Not enough buying power for equity short")
            log_signal("HOLD", current_price, prob_up, "DOWN")
            return
        ticker_print(ticker, f">>> {action} {qty} shares of {ticker}")
        result = place_order(ticker, qty, "sell")
        if "error" in result:
            ticker_print(ticker, f"FAILED: {result.get('error')}")
            action = "FAILED"
        else:
            ticker_print(ticker, f"Result: {result.get('status', result)}")
        log_signal(action, current_price, prob_up, "DOWN")

    elif signal["bullish_signal"] and shares > 0 and position_type == "short":
        action = "COVER SHORT"
        ticker_print(ticker, f">>> {action} {shares} shares of {ticker}")
        result = close_position(ticker)
        ticker_print(ticker, f"Result: {result.get('status', result)}")
        log_signal(action, current_price, prob_up, "UP")

    elif signal["bearish_signal"] and shares > 0 and position_type == "long":
        action = "SELL (STOP LOSS)"
        ticker_print(ticker, f">>> {action} {shares} shares of {ticker}")
        result = close_position(ticker)
        ticker_print(ticker, f"Result: {result.get('status', result)}")
        log_signal(action, current_price, prob_up, "DOWN")

    else:
        direction = "UP" if prob_up > 0.5 else "DOWN"
        action = "HOLD"
        ticker_print(
            ticker,
            f">>> {action} - No clear equity signal "
            f"(predicted: {direction}, gap: {signal['confidence_gap']:.2%}; "
            f"reason: {describe_signal_rejection(signal, prob_up)})",
        )
        log_signal(action, current_price, prob_up, direction)

    ticker_print(ticker, f"Position: {shares} shares of {ticker}")


def trade_option(ticker, account, current_price, prob_up, features, selected_contract=None):
    signal = evaluate_signal(features, current_price, prob_up)
    if ticker not in OPTIONS_ENABLED_UNDERLYINGS:
        ticker_print(ticker, ">>> HOLD - Options disabled for this underlying")
        log_signal("HOLD", current_price, prob_up, "UP" if prob_up > 0.5 else "DOWN")
        return

    if not account_supports_options(account):
        ticker_print(ticker, ">>> HOLD - Account is not approved for options trading")
        log_signal("HOLD", current_price, prob_up, "UP" if prob_up > 0.5 else "DOWN")
        return

    position, contract = find_option_position_for_underlying(ticker)
    has_pending = has_pending_option_order(ticker)
    contracts_held = int(float(position["qty"])) if position else 0
    options_buying_power = float(
        account.get("options_buying_power")
        or account.get("buying_power")
        or account.get("cash")
        or 0
    )

    held_label = contract.get("symbol") if contract else "None"
    ticker_print(ticker, f"Current option position: {held_label} x {contracts_held}")
    ticker_print(ticker, f"Pending option orders: {int(has_pending)}")
    ticker_print(ticker, f"Threshold: {THRESHOLD:.0%}")
    ticker_print(ticker, f"Options buying power: ${options_buying_power:.2f}")

    market_open = get_market_status()
    if not market_open:
        ticker_print(ticker, ">>> HOLD - Market closed, skipping trade")
        log_signal("HOLD", current_price, prob_up, "UP" if prob_up > 0.5 else "DOWN")
        return

    prob_down = signal["prob_down"]
    desired_type = "call" if prob_up > 0.5 else "put"
    held_type = contract.get("type") if contract else None

    if position and held_type != desired_type and not has_pending:
        ticker_print(ticker, f">>> CLOSE OPTION {position['symbol']} x {contracts_held}")
        result = close_position(position["symbol"])
        ticker_print(ticker, f"Result: {result.get('status', result)}")
        log_signal("CLOSE OPTION", current_price, prob_up, desired_type.upper())
        return

    if position:
        direction = "UP" if prob_up > 0.5 else "DOWN"
        ticker_print(ticker, f">>> HOLD - Existing {held_type} position already aligned")
        log_signal("HOLD", current_price, prob_up, direction)
        return

    if has_pending:
        direction = "UP" if prob_up > 0.5 else "DOWN"
        ticker_print(ticker, ">>> HOLD - Pending option order already exists")
        log_signal("HOLD", current_price, prob_up, direction)
        return

    if not signal["bullish_signal"] and not signal["bearish_signal"]:
        direction = "UP" if prob_up > 0.5 else "DOWN"
        ticker_print(
            ticker,
            f">>> HOLD - No clear option signal (predicted: {direction}; "
            f"reason: {describe_signal_rejection(signal, prob_up)})",
        )
        log_signal("HOLD", current_price, prob_up, direction)
        return

    contract = selected_contract or select_option_contract(
        ticker, desired_type, current_price
    )
    if not contract:
        ticker_print(ticker, ">>> HOLD - No suitable option contract found")
        log_signal("HOLD", current_price, prob_up, desired_type.upper())
        return

    premium = float(contract.get("close_price") or 0)
    cost_per_contract = premium * 100
    budget = options_buying_power * OPTIONS_POSITION_SIZE
    qty = int(budget / cost_per_contract) if cost_per_contract > 0 else 0
    if qty <= 0:
        ticker_print(ticker, ">>> HOLD - Not enough buying power for selected option")
        log_signal("HOLD", current_price, prob_up, desired_type.upper())
        return

    action = f"BUY {desired_type.upper()}"
    ticker_print(
        ticker,
        f"\n>>> {action} {qty} contract(s) of {contract['symbol']} "
        f"(strike ${float(contract['strike_price']):.2f}, exp {contract['expiration_date']})",
    )
    result = place_order(contract["symbol"], qty, "buy", asset_class="option")
    if "error" in result:
        ticker_print(ticker, f"FAILED: {result.get('error')}")
        action = "FAILED"
    else:
        ticker_print(ticker, f"Result: {result.get('status', result)}")
    log_signal(action, current_price, prob_up, desired_type.upper())


def main(ticker):
    ticker_print(ticker, "=" * 50)
    ticker_print(ticker, f"Enhanced Alpaca Paper Trading Bot - {ticker}")
    ticker_print(ticker, "=" * 50)

    if not check_alpaca_keys():
        return

    account = get_account()
    if account:
        cash = float(account.get("cash", 0))
        equity = float(account.get("portfolio_value", cash))
        ticker_print(ticker, f"Connected - Cash: ${cash:.2f} | Equity: ${equity:.2f}")
        if account.get("options_buying_power") is not None:
            ticker_print(
                ticker,
                "Options - "
                f"Buying Power: ${float(account.get('options_buying_power') or 0):.2f} | "
                f"Approved Level: {account.get('options_approved_level', 0)} | "
                f"Trading Level: {account.get('options_trading_level', 0)}",
            )
    else:
        ticker_print(ticker, "ERROR: Could not connect to Alpaca")
        ticker_print(ticker, "Check your API keys in .env")
        return

    if has_traded_today(ticker):
        ticker_print(
            ticker,
            ">>> HOLD - Already placed a non-canceled order for this ticker today",
        )
        return

    classifier, scaler = load_models(ticker)

    ticker_print(ticker, "Fetching market data with technical indicators...")
    raw_data = get_yfinance_data(ticker)
    features = prepare_features(raw_data)

    current_price = raw_data["Close"].iloc[-1]
    prob_up = predict_direction(classifier, scaler, features)

    ticker_print(ticker, f"Current price: ${current_price:.2f}")
    ticker_print(ticker, f"Probability of UP: {prob_up:.2%}")
    ticker_print(ticker, f"Probability of DOWN: {1 - prob_up:.2%}")
    if TRADE_MODE == "auto":
        decision = choose_trade_mode(ticker, account, current_price, prob_up, features)
        ticker_print(
            ticker,
            f"Auto mode: {decision['mode'].upper()} | "
            f"Expected move: {decision['expected_move_pct']:.2%} | "
            f"Reason: {decision['reason']}",
        )
        if decision["mode"] == "hold":
            log_signal("HOLD", current_price, prob_up, decision["direction"])
            ticker_print(
                ticker,
                ">>> HOLD - Auto mode rejected trade before execution "
                f"(predicted: {decision['direction']})",
            )
        elif decision["mode"] == "option":
            trade_option(
                ticker, account, current_price, prob_up, features, decision["contract"]
            )
        else:
            trade_equity(ticker, account, current_price, prob_up, features)
    elif TRADE_MODE == "option":
        trade_option(ticker, account, current_price, prob_up, features)
    else:
        trade_equity(ticker, account, current_price, prob_up, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpaca paper trading bot")
    parser.add_argument(
        "--ticker", type=str, default=None, help="Specific ticker to trade"
    )
    parser.add_argument(
        "--all", action="store_true", help="Trade all tickers from config"
    )
    parser.add_argument(
        "--mode",
        choices=["equity", "option", "auto"],
        default=TRADE_MODE,
        help="Trading mode override",
    )
    args = parser.parse_args()
    TRADE_MODE = args.mode

    if args.all:
        import concurrent.futures

        tickers = config.TICKERS
        print(f"\n{'#' * 60}")
        print(f"# MrKrabs Trading Bot - All Tickers")
        print(f"# Tickers: {tickers}")
        print(f"{'#' * 60}\n")

        def trade_single_ticker(ticker):
            main(ticker)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(tickers)
        ) as executor:
            futures = [executor.submit(trade_single_ticker, t) for t in tickers]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"ERROR: Ticker worker failed: {exc}")
    else:
        if args.ticker:
            ticker = args.ticker
        elif hasattr(config, "DEFAULT_TICKER"):
            ticker = config.DEFAULT_TICKER
        else:
            ticker = "SPY"
        main(ticker)
