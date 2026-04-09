import argparse
import json
import os
import pickle
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Add project directory to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
import config

DATA_DIR = config.DATA_DIR
MODEL_DIR = config.MODEL_DIR
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore", category=UserWarning)


LOOK_BACK = 60
THRESHOLD = 0.45  # Classifier threshold (45% - trade on any upward signal)
LONG_ENTRY_THRESHOLD = getattr(config, "LONG_ENTRY_THRESHOLD", 0.52)
SHORT_ENTRY_THRESHOLD = getattr(config, "SHORT_ENTRY_THRESHOLD", 0.48)
MIN_CONFIDENCE_GAP = getattr(config, "MIN_CONFIDENCE_GAP", 0.02)
USE_TREND_FILTER = getattr(config, "USE_TREND_FILTER", False)
STOP_LOSS_PCT = getattr(config, "STOP_LOSS_PCT", 0.03)
TAKE_PROFIT_PCT = getattr(config, "TAKE_PROFIT_PCT", 0.06)
MAX_HOLD_DAYS = getattr(config, "MAX_HOLD_DAYS", 20)
AUTO_TUNE_ENABLED = getattr(config, "AUTO_TUNE_ENABLED", True)
MIN_SIGNALS_PER_TICKER = getattr(config, "MIN_SIGNALS_PER_TICKER", 30)
TARGET_WIN_RATE = getattr(config, "TARGET_WIN_RATE", 0.50)
SELECTOR_FEATURE_COLUMNS = [
    "prob_up",
    "prob_down",
    "confidence_gap",
    "current_price",
    "atr_pct",
    "volatility_10",
    "volatility_20",
    "rsi",
    "macd_hist",
    "momentum_5",
    "momentum_10",
    "bb_width",
]


def load_scaler(ticker=None):
    if ticker is None:
        ticker = config.TICKERS[0] if config.TICKERS else "SPY"
    with open(f"{config.get_data_dir(ticker)}/scaler.pkl", "rb") as f:
        return pickle.load(f)


def load_data(ticker=None):
    if ticker is None:
        ticker = config.TICKERS[0] if config.TICKERS else "SPY"
    raw_df = pd.read_csv(
        f"{config.get_data_dir(ticker)}/raw_data.csv", parse_dates=["Date"]
    )
    y_dir_test = np.load(f"{config.get_data_dir(ticker)}/y_dir_test.npy")
    return raw_df, y_dir_test


def get_feature_columns():
    return [
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


def load_classifier_model(ticker=None):
    if ticker is None:
        ticker = config.TICKERS[0] if config.TICKERS else "SPY"
    return load_model(f"{config.get_model_dir(ticker)}/classifier_model.keras")


def predict_directions(model, scaler, raw_df, look_back=LOOK_BACK):
    print("\nGenerating predictions for each timestep...")

    df = add_technical_indicators(raw_df)
    feature_cols = get_feature_columns()
    features = df[feature_cols].values
    scaled_features = scaler.transform(features)

    # Generate predictions for each time step (need look_back window)
    probabilities = []
    actual_directions = []

    for i in range(look_back, len(scaled_features)):
        seq = scaled_features[i - look_back : i].reshape(1, look_back, 25)
        prob = model.predict(seq, verbose=0)[0, 0]
        probabilities.append(prob)

        # Actual direction for the predicted session: today's close vs yesterday's close.
        previous_close = raw_df["Close"].iloc[i - 1]
        current_close = raw_df["Close"].iloc[i]
        actual_directions.append(1 if current_close > previous_close else 0)

    return np.array(probabilities), np.array(actual_directions)


def backtest_classifier(
    probabilities, actual_directions, prices, features_df, ticker=None
):
    if ticker is None:
        ticker = config.TICKERS[0] if config.TICKERS else "SPY"
    initial_capital = 10000
    capital = initial_capital
    position = 0
    trades = []
    portfolio_value = [capital]

    entry_price = None
    entry_day = None

    for i in range(len(probabilities)):
        prob_up = probabilities[i]
        prob_down = 1 - prob_up
        current_price = prices[i]
        row = features_df.iloc[i]
        confidence_gap = abs(prob_up - 0.5)
        reference_price = float(row.get("Close", current_price))
        bullish_trend = reference_price > float(
            row.get("SMA_20", reference_price)
        ) and float(row.get("SMA_20", reference_price)) >= float(
            row.get("SMA_50", reference_price)
        )
        bearish_trend = reference_price < float(
            row.get("SMA_20", reference_price)
        ) and float(row.get("SMA_20", reference_price)) <= float(
            row.get("SMA_50", reference_price)
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

        should_exit = False
        exit_reason = None

        if position != 0 and entry_price is not None:
            price_change = (current_price - entry_price) / entry_price
            if position > 0:
                if price_change <= -STOP_LOSS_PCT:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                elif price_change >= TAKE_PROFIT_PCT:
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"
            else:
                if price_change >= STOP_LOSS_PCT:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                elif price_change <= -TAKE_PROFIT_PCT:
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"

            if entry_day is not None and (i - entry_day) >= MAX_HOLD_DAYS:
                should_exit = True
                exit_reason = "MAX_HOLD"

        if should_exit:
            if position > 0:
                capital = capital + (position * current_price)
                trades.append(("SELL", i, current_price, exit_reason))
            elif position < 0:
                shares = abs(position)
                capital = capital - (shares * current_price)
                trades.append(("COVER", i, current_price, exit_reason))
            position = 0
            entry_price = None
            entry_day = None

        if bullish_signal and position <= 0:
            if position < 0:
                shares = abs(position)
                capital = capital - (shares * current_price)
                trades.append(("COVER", i, current_price))
                position = 0

            shares = int((capital * 0.15) / current_price)
            position = shares
            capital = capital - (shares * current_price)
            trades.append(("LONG", i, current_price))
            entry_price = current_price
            entry_day = i

        elif bearish_signal and position >= 0:
            if position > 0:
                capital = capital + (position * current_price)
                trades.append(("SELL", i, current_price))
                position = 0

            shares = int((capital * 0.15) / current_price)
            position = -shares
            capital = capital + (shares * current_price)
            trades.append(("SHORT", i, current_price))
            entry_price = current_price
            entry_day = i

        if position != 0:
            value = capital + (position * current_price)
        else:
            value = capital
        portfolio_value.append(value)

    # Close final position
    if position != 0:
        final_price = prices[-1]
        if position > 0:
            capital = capital + (position * final_price)
            trades.append(("SELL", len(probabilities) - 1, final_price))
        elif position < 0:
            shares = abs(position)
            capital = capital - (shares * final_price)
            trades.append(("COVER", len(probabilities) - 1, final_price))
        portfolio_value[-1] = capital

    return capital, trades, portfolio_value


def analyze_model_confidence(probabilities, actuals):
    """Analyze model confidence distribution"""
    mean_confidence = np.mean(np.abs(probabilities - 0.5)) * 2
    std_confidence = np.std(np.abs(probabilities - 0.5)) * 2
    
    accuracy = (predictions := (probabilities > 0.5).astype(int)) == actuals
    accuracy_at_50 = accuracy.mean()
    accuracy_at_52 = (probabilities >= 0.52).astype(int) == actuals
    accuracy_at_54 = (probabilities >= 0.54).astype(int) == actuals
    
    high_conf_count = np.sum(np.abs(probabilities - 0.5) >= 0.02)
    very_high_conf_count = np.sum(np.abs(probabilities - 0.5) >= 0.04)
    
    return {
        "mean_confidence": mean_confidence,
        "std_confidence": std_confidence,
        "accuracy_at_50": accuracy_at_50,
        "accuracy_at_52": accuracy_at_52.mean() if len(accuracy_at_52) > 0 else 0,
        "accuracy_at_54": accuracy_at_54.mean() if len(accuracy_at_54) > 0 else 0,
        "high_conf_signals": high_conf_count,
        "very_high_conf_signals": very_high_conf_count,
    }


def run_backtest_with_thresholds(probabilities, prices, features_df, long_entry, short_entry, min_conf_gap):
    """Run backtest with specific thresholds - returns metrics"""
    initial_capital = 10000
    capital = initial_capital
    position = 0
    trades = []
    portfolio_value = [capital]

    entry_price = None
    entry_day = None

    for i in range(len(probabilities)):
        prob_up = probabilities[i]
        current_price = prices[i]
        row = features_df.iloc[i]
        confidence_gap = abs(prob_up - 0.5)

        bullish_signal = prob_up >= long_entry and confidence_gap >= min_conf_gap
        bearish_signal = prob_up <= short_entry and confidence_gap >= min_conf_gap

        should_exit = False
        exit_reason = None

        if position != 0 and entry_price is not None:
            price_change = (current_price - entry_price) / entry_price
            if position > 0:
                if price_change <= -STOP_LOSS_PCT:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                elif price_change >= TAKE_PROFIT_PCT:
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"
            else:
                if price_change >= STOP_LOSS_PCT:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                elif price_change <= -TAKE_PROFIT_PCT:
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"

            if entry_day is not None and (i - entry_day) >= MAX_HOLD_DAYS:
                should_exit = True
                exit_reason = "MAX_HOLD"

        if should_exit:
            if position > 0:
                capital = capital + (position * current_price)
                trades.append(("SELL", i, current_price, exit_reason))
            elif position < 0:
                shares = abs(position)
                capital = capital - (shares * current_price)
                trades.append(("COVER", i, current_price, exit_reason))
            position = 0
            entry_price = None
            entry_day = None

        if bullish_signal and position <= 0:
            if position < 0:
                shares = abs(position)
                capital = capital - (shares * current_price)
                trades.append(("COVER", i, current_price))
                position = 0

            shares = int((capital * 0.15) / current_price)
            if shares > 0:
                position = shares
                capital = capital - (shares * current_price)
                trades.append(("LONG", i, current_price))
                entry_price = current_price
                entry_day = i

        elif bearish_signal and position >= 0:
            if position > 0:
                capital = capital + (position * current_price)
                trades.append(("SELL", i, current_price))
                position = 0

            shares = int((capital * 0.15) / current_price)
            if shares > 0:
                position = -shares
                capital = capital + (shares * current_price)
                trades.append(("SHORT", i, current_price))
                entry_price = current_price
                entry_day = i

        if position != 0:
            value = capital + (position * current_price)
        else:
            value = capital
        portfolio_value.append(value)

    if position != 0:
        final_price = prices[-1]
        if position > 0:
            capital = capital + (position * final_price)
            trades.append(("SELL", len(probabilities) - 1, final_price))
        elif position < 0:
            shares = abs(position)
            capital = capital - (shares * final_price)
            trades.append(("COVER", len(probabilities) - 1, final_price))
        portfolio_value[-1] = capital

    metrics = calculate_metrics(portfolio_value, trades)
    return metrics, trades, portfolio_value


def auto_tune_thresholds(ticker, probabilities, actuals, prices, features_df):
    """Find optimal thresholds by testing combinations - optimize for Sharpe"""
    print(f"\nAuto-tuning thresholds for {ticker}...")
    
    analysis = analyze_model_confidence(probabilities, actuals)
    print(f"  Model accuracy: {analysis['accuracy_at_50']:.2%}")
    print(f"  High confidence signals: {analysis['high_conf_signals']}")
    
    best_sharpe = -float('inf')
    best_config = None
    best_metrics = None

    for long_entry in np.arange(0.50, 0.56, 0.01):
        short_entry = 1.0 - long_entry
        for min_conf_gap in np.arange(0.00, 0.06, 0.01):
            metrics, trades, _ = run_backtest_with_thresholds(
                probabilities, prices, features_df, 
                long_entry, short_entry, min_conf_gap
            )
            
            total_trades = metrics.get("Total Trades", 0)
            total_return = float(metrics.get("Total Return", "0%").replace("%", "")) / 100
            sharpe = float(metrics.get("Sharpe Ratio", "-999"))
            
            if total_return > 0 and total_trades >= MIN_SIGNALS_PER_TICKER:
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_config = {
                        "long_entry": round(long_entry, 2),
                        "short_entry": round(short_entry, 2),
                        "min_conf_gap": round(min_conf_gap, 2),
                    }
                    best_metrics = {
                        "sharpe_ratio": sharpe,
                        "total_return": total_return,
                        "total_trades": total_trades,
                        "wins": metrics.get("Wins", 0),
                        "losses": metrics.get("Losses", 0),
                        "win_rate": metrics.get("Wins", 0) / total_trades if total_trades > 0 else 0,
                    }

    if best_config is None:
        print(f"  No valid configuration found (requires return > 0 and >= {MIN_SIGNALS_PER_TICKER} trades)")
        return None

    print(f"  Best config: long={best_config['long_entry']}, min_gap={best_config['min_conf_gap']}")
    print(f"  Sharpe: {best_metrics['sharpe_ratio']:.2f}, Return: {best_metrics['total_return']:.2%}, Trades: {best_metrics['total_trades']}")
    
    return {**best_config, **best_metrics}


def save_tuned_thresholds(ticker, thresholds):
    """Save tuned thresholds to JSON file"""
    model_dir = config.get_model_dir(ticker)
    filepath = os.path.join(model_dir, "tuned_thresholds.json")
    
    data = {
        "ticker": ticker,
        "long_entry": thresholds["long_entry"],
        "short_entry": thresholds["short_entry"],
        "min_conf_gap": thresholds["min_conf_gap"],
        "sharpe_ratio": thresholds.get("sharpe_ratio", 0),
        "total_return": thresholds.get("total_return", 0),
        "total_trades": thresholds.get("total_trades", 0),
        "win_rate": thresholds.get("win_rate", 0),
        "tuned_date": datetime.now().strftime("%Y-%m-%d"),
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved tuned thresholds to {filepath}")


def load_tuned_thresholds(ticker):
    """Load tuned thresholds from JSON file"""
    filepath = os.path.join(config.get_model_dir(ticker), "tuned_thresholds.json")
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = json.load(f)
        print(f"Loaded tuned thresholds: long={data['long_entry']}, min_gap={data['min_conf_gap']}")
        return data
    return None


def align_prediction_frame(test_df, probabilities):
    df = add_technical_indicators(test_df.copy())
    aligned = df.iloc[LOOK_BACK:].copy().reset_index(drop=True)
    signal_reference = df.iloc[LOOK_BACK - 1 : -1].copy().reset_index(drop=True)
    aligned["reference_close"] = signal_reference["Close"]
    aligned["decision_open"] = aligned["Open"]
    aligned["prob_up"] = probabilities
    aligned["prob_down"] = 1 - probabilities
    aligned["confidence_gap"] = np.abs(aligned["prob_up"] - 0.5)
    aligned["session_return"] = (
        aligned["Close"] - aligned["reference_close"]
    ) / aligned["reference_close"]
    for column in [
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
    ]:
        aligned[column] = signal_reference[column]
    return aligned


def simulate_option_return(row):
    direction = 1 if row["prob_up"] >= 0.5 else -1
    directional_move = direction * row["session_return"]
    atr_pct = max(
        float(row["ATR"] / row["reference_close"]) if row["reference_close"] > 0 else 0,
        0,
    )
    vol_component = max(float(row.get("Volatility_10", 0) or 0), 0)

    # Synthetic one-step option premium and payoff proxy built from realized move.
    premium_pct = max(0.01, 0.40 * atr_pct + 0.15 * vol_component + 0.005)
    strike_offset_pct = min(0.02, max(0.0, 0.20 * atr_pct))
    intrinsic_pct = max(directional_move - strike_offset_pct, 0)
    gross_return = (intrinsic_pct - premium_pct) / premium_pct
    return float(np.clip(gross_return, -1.0, 4.0))


def build_mode_selector_dataset(test_df, probabilities, threshold):
    aligned = align_prediction_frame(test_df, probabilities)
    aligned["signal_active"] = (
        (
            (aligned["prob_up"] >= LONG_ENTRY_THRESHOLD)
            | (aligned["prob_up"] <= SHORT_ENTRY_THRESHOLD)
        )
        & (aligned["confidence_gap"] >= MIN_CONFIDENCE_GAP)
    ).astype(int)
    if USE_TREND_FILTER:
        bullish_trend = (aligned["Close"] > aligned["SMA_20"]) & (
            aligned["SMA_20"] >= aligned["SMA_50"]
        )
        bearish_trend = (aligned["Close"] < aligned["SMA_20"]) & (
            aligned["SMA_20"] <= aligned["SMA_50"]
        )
        aligned["signal_active"] = (
            aligned["signal_active"].astype(bool)
            & (
                ((aligned["prob_up"] >= LONG_ENTRY_THRESHOLD) & bullish_trend)
                | ((aligned["prob_up"] <= SHORT_ENTRY_THRESHOLD) & bearish_trend)
            )
        ).astype(int)
    selector_df = aligned[aligned["signal_active"] == 1].copy().reset_index(drop=True)

    if selector_df.empty:
        return selector_df

    selector_df["equity_return"] = np.where(
        selector_df["prob_up"] >= selector_df["prob_down"],
        selector_df["session_return"],
        -selector_df["session_return"],
    )
    selector_df["option_return"] = selector_df.apply(simulate_option_return, axis=1)
    selector_df["best_mode"] = np.where(
        selector_df["option_return"] > selector_df["equity_return"], "option", "equity"
    )
    selector_df["best_mode_label"] = np.where(
        selector_df["best_mode"] == "option", 1, 0
    )
    selector_df["current_price"] = selector_df["decision_open"]
    selector_df["atr_pct"] = selector_df["ATR"] / selector_df["reference_close"]
    selector_df["volatility_10"] = selector_df["Volatility_10"]
    selector_df["volatility_20"] = selector_df["Volatility_20"]
    selector_df["rsi"] = selector_df["RSI"]
    selector_df["macd_hist"] = selector_df["MACD_Hist"]
    selector_df["momentum_5"] = selector_df["Momentum_5"]
    selector_df["momentum_10"] = selector_df["Momentum_10"]
    selector_df["bb_width"] = selector_df["BB_Width"]
    return selector_df


def train_mode_selector(selector_df, ticker):
    if len(selector_df) < 25:
        print(
            f"Skipping selector training for {ticker}: not enough signal rows ({len(selector_df)})"
        )
        return None

    X = selector_df[SELECTOR_FEATURE_COLUMNS].fillna(0)
    y = selector_df["best_mode_label"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    selector_df = selector_df.copy()
    selector_df["predicted_mode_label"] = predictions
    selector_df["predicted_mode"] = np.where(predictions == 1, "option", "equity")
    selector_df["chosen_return"] = np.where(
        selector_df["predicted_mode_label"] == 1,
        selector_df["option_return"],
        selector_df["equity_return"],
    )

    artifact = {
        "model": model,
        "feature_columns": SELECTOR_FEATURE_COLUMNS,
        "metrics": {
            "training_accuracy": float(accuracy),
            "sample_count": int(len(selector_df)),
            "mean_equity_return": float(selector_df["equity_return"].mean()),
            "mean_option_return": float(selector_df["option_return"].mean()),
            "mean_chosen_return": float(selector_df["chosen_return"].mean()),
            "option_label_rate": float(selector_df["best_mode_label"].mean()),
        },
    }

    with open(f"{config.get_model_dir(ticker)}/mode_selector.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print(
        f"Saved mode selector to {config.get_model_dir(ticker)}/mode_selector.pkl "
        f"(training accuracy: {accuracy:.2%})"
    )
    return artifact, selector_df


def backtest_single_ticker(ticker):
    """Backtest a single ticker"""
    print(f"\n{'=' * 60}")
    print(f"Backtesting ticker: {ticker}")
    print(f"{'=' * 60}")

    print("\nLoading classifier model...")
    model = load_classifier_model(ticker)
    print(f"Loaded model from {config.get_model_dir(ticker)}/classifier_model.keras")

    raw_df, y_dir_test = load_data(ticker)
    scaler = load_scaler(ticker)

    print(f"Raw data shape: {raw_df.shape}")
    print(
        f"Direction labels: up={sum(y_dir_test)}, down={len(y_dir_test) - sum(y_dir_test)}"
    )

    # Get predictions and actuals for test period
    test_start_idx = int(len(raw_df) * 0.8)
    test_df = raw_df.iloc[test_start_idx:].reset_index(drop=True)

    probabilities, actuals = predict_directions(model, scaler, test_df)
    prices = test_df["Open"].iloc[config.LOOK_BACK :].values
    features_df = add_technical_indicators(test_df.copy()).iloc[
        config.LOOK_BACK - 1 : -1
    ]
    features_df = features_df.reset_index(drop=True)

    preds = (probabilities > 0.5).astype(int)
    actuals_arr = actuals
    accuracy = (preds == actuals_arr).mean()
    print(f"Predictions: {len(probabilities)}")
    print(f"Prediction accuracy: {accuracy:.2%}")

    # Auto-tune thresholds
    tuned = load_tuned_thresholds(ticker)
    if tuned is None and AUTO_TUNE_ENABLED:
        tuned = auto_tune_thresholds(ticker, probabilities, actuals, prices, features_df)
        if tuned:
            save_tuned_thresholds(ticker, tuned)
    
    if tuned:
        global LONG_ENTRY_THRESHOLD, SHORT_ENTRY_THRESHOLD, MIN_CONFIDENCE_GAP
        LONG_ENTRY_THRESHOLD = tuned["long_entry"]
        SHORT_ENTRY_THRESHOLD = tuned["short_entry"]
        MIN_CONFIDENCE_GAP = tuned["min_conf_gap"]
        print(f"\nUsing tuned thresholds: long={LONG_ENTRY_THRESHOLD}, min_gap={MIN_CONFIDENCE_GAP}")
    else:
        if AUTO_TUNE_ENABLED:
            print(f"\nSkipping {ticker}: no valid threshold configuration found")
            return
        print(f"\nUsing default thresholds: long={LONG_ENTRY_THRESHOLD}, min_gap={MIN_CONFIDENCE_GAP}")

    print("\n--- Backtesting (Classifier) ---")
    capital, trades, portfolio_value = backtest_classifier(
        probabilities, actuals, prices, features_df, ticker=ticker
    )

    metrics = calculate_metrics(portfolio_value, trades)
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Save results
    ticker_results_dir = os.path.join(RESULTS_DIR, ticker)
    os.makedirs(ticker_results_dir, exist_ok=True)
    portfolio_dates = test_df["Date"].iloc[config.LOOK_BACK :].reset_index(drop=True)
    df_results = pd.DataFrame(
        {
            "Date": portfolio_dates.iloc[: len(portfolio_value)],
            "Portfolio_Value": portfolio_value[1:],
        }
    )
    df_results.to_csv(f"{ticker_results_dir}/backtest_results.csv", index=False)
    print(f"\nSaved backtest results to {ticker_results_dir}/backtest_results.csv")

    selector_df = build_mode_selector_dataset(
        test_df, probabilities, threshold=config.THRESHOLD
    )
    if not selector_df.empty:
        selector_df.to_csv(
            f"{ticker_results_dir}/mode_selector_dataset.csv", index=False
        )
        print(
            f"Saved selector dataset to {ticker_results_dir}/mode_selector_dataset.csv"
        )

        selector_result = train_mode_selector(selector_df, ticker)
        if selector_result:
            artifact, trained_selector_df = selector_result
            metrics_path = f"{ticker_results_dir}/mode_selector_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(artifact["metrics"], f, indent=2)
            trained_selector_df.to_csv(
                f"{ticker_results_dir}/mode_selector_predictions.csv", index=False
            )
            print(f"Saved selector metrics to {metrics_path}")

    print(f"\nTrades made: {len(trades)}")
    for t in trades[-10:]:
        print(f"  {t}")


def calculate_metrics(portfolio_value, trades, initial_capital=10000):
    portfolio = np.array(portfolio_value)

    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(portfolio) / portfolio[:-1]
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    total_return = (portfolio[-1] - initial_capital) / initial_capital

    valid_returns = returns[returns != 0]
    if len(valid_returns) > 0 and valid_returns.std() > 0:
        sharpe_ratio = (valid_returns.mean() / valid_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    peak = initial_capital
    max_drawdown = 0
    for value in portfolio:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Calculate wins/losses by matching entries with exits
    wins = 0
    losses = 0
    entry_prices = {}  # Track entry price for each position

    for t in trades:
        action = t[0]
        price = t[2]

        if action in ("LONG", "SHORT"):
            # Store entry price
            entry_prices[action] = price

        elif action in ("SELL", "COVER"):
            # Find matching entry
            entry_price = None
            if action == "SELL" and "LONG" in entry_prices:
                entry_price = entry_prices.pop("LONG")
            elif action == "COVER" and "SHORT" in entry_prices:
                entry_price = entry_prices.pop("SHORT")

            if entry_price is not None:
                is_win = (
                    price > entry_price if action == "SELL" else price < entry_price
                )
                if is_win:
                    wins += 1
                else:
                    losses += 1

    win_loss_ratio = wins / losses if losses > 0 else float("inf")

    return {
        "Total Return": f"{total_return * 100:.2f}%",
        "Final Value": f"${portfolio[-1]:.2f}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"-{max_drawdown * 100:.2f}%",
        "Total Trades": len([t for t in trades if t[0] in ("LONG", "SHORT")]),
        "Wins": wins,
        "Losses": losses,
        "Win/Loss Ratio": f"{win_loss_ratio:.2f}"
        if win_loss_ratio != float("inf")
        else "inf",
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest trading strategy")
    parser.add_argument(
        "--ticker", type=str, default=None, help="Specific ticker to backtest"
    )
    parser.add_argument(
        "--all", action="store_true", help="Backtest all tickers from config"
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

    print(f"\n{'#' * 60}")
    print(f"# MrKrabs Backtesting")
    print(f"# Tickers: {tickers}")
    print(f"{'#' * 60}")

    # Backtest each ticker
    for ticker in tickers:
        backtest_single_ticker(ticker)

    print(f"\n\n{'#' * 60}")
    print(f"# Backtesting Complete")
    print(f"# Processed tickers: {tickers}")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
