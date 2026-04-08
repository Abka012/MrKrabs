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
LONG_ENTRY_THRESHOLD = getattr(config, "LONG_ENTRY_THRESHOLD", 0.58)
SHORT_ENTRY_THRESHOLD = getattr(config, "SHORT_ENTRY_THRESHOLD", 0.42)
MIN_CONFIDENCE_GAP = getattr(config, "MIN_CONFIDENCE_GAP", 0.08)
USE_TREND_FILTER = getattr(config, "USE_TREND_FILTER", True)
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

    for i in range(look_back, len(scaled_features) - 1):
        seq = scaled_features[i - look_back : i].reshape(1, look_back, 25)
        prob = model.predict(seq, verbose=0)[0, 0]
        probabilities.append(prob)

        # Actual direction: price went up or down next day
        current_price = raw_df["Close"].iloc[i]
        next_price = raw_df["Close"].iloc[i + 1]
        actual_directions.append(1 if next_price > current_price else 0)

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

    for i in range(len(probabilities)):
        prob_up = probabilities[i]
        prob_down = 1 - prob_up
        current_price = prices[i]
        row = features_df.iloc[i]
        confidence_gap = abs(prob_up - 0.5)
        bullish_trend = (
            current_price > float(row.get("SMA_20", current_price))
            and float(row.get("SMA_20", current_price))
            >= float(row.get("SMA_50", current_price))
        )
        bearish_trend = (
            current_price < float(row.get("SMA_20", current_price))
            and float(row.get("SMA_20", current_price))
            <= float(row.get("SMA_50", current_price))
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

        # Trading logic with classifier
        if bullish_signal and position <= 0:
            # Enter LONG
            if position < 0:
                # Cover short first
                shares = abs(position)
                capital = capital - (shares * current_price)
                trades.append(("COVER", i, current_price))
                position = 0

            shares = int((capital * 0.15) / current_price)
            position = shares
            capital = capital - (shares * current_price)
            trades.append(("LONG", i, current_price))

        elif bearish_signal and position >= 0:
            # Enter SHORT
            if position > 0:
                # Sell long first
                capital = capital + (position * current_price)
                trades.append(("SELL", i, current_price))
                position = 0

            shares = int((capital * 0.15) / current_price)
            position = -shares
            capital = capital + (shares * current_price)
            trades.append(("SHORT", i, current_price))

        elif (bullish_signal and position < 0) or (bearish_signal and position > 0):
            # Exit positions
            if position > 0:
                capital = capital + (position * current_price)
                trades.append(("SELL", i, current_price))
            elif position < 0:
                shares = abs(position)
                capital = capital - (shares * current_price)
                trades.append(("COVER", i, current_price))
            position = 0

        # Calculate portfolio value
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


def align_prediction_frame(test_df, probabilities):
    df = add_technical_indicators(test_df.copy())
    aligned = df.iloc[LOOK_BACK:-1].copy().reset_index(drop=True)
    aligned["next_close"] = df["Close"].iloc[LOOK_BACK + 1 :].reset_index(drop=True)
    aligned["prob_up"] = probabilities
    aligned["prob_down"] = 1 - probabilities
    aligned["confidence_gap"] = np.abs(aligned["prob_up"] - 0.5)
    aligned["next_return"] = (
        aligned["next_close"] - aligned["Close"]
    ) / aligned["Close"]
    return aligned


def simulate_option_return(row):
    direction = 1 if row["prob_up"] >= 0.5 else -1
    directional_move = direction * row["next_return"]
    atr_pct = max(float(row["ATR"] / row["Close"]) if row["Close"] > 0 else 0, 0)
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
        selector_df["next_return"],
        -selector_df["next_return"],
    )
    selector_df["option_return"] = selector_df.apply(simulate_option_return, axis=1)
    selector_df["best_mode"] = np.where(
        selector_df["option_return"] > selector_df["equity_return"], "option", "equity"
    )
    selector_df["best_mode_label"] = np.where(
        selector_df["best_mode"] == "option", 1, 0
    )
    selector_df["current_price"] = selector_df["Close"]
    selector_df["atr_pct"] = selector_df["ATR"] / selector_df["Close"]
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
    prices = test_df["Close"].iloc[config.LOOK_BACK : -1].values
    features_df = add_technical_indicators(test_df.copy()).iloc[config.LOOK_BACK : -1]
    features_df = features_df.reset_index(drop=True)

    preds = (probabilities > 0.5).astype(int)
    actuals_arr = actuals
    accuracy = (preds == actuals_arr).mean()
    print(f"Predictions: {len(probabilities)}")
    print(f"Prediction accuracy: {accuracy:.2%}")

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
    portfolio_dates = (
        test_df["Date"].iloc[config.LOOK_BACK - 1 : -1].reset_index(drop=True)
    )
    df_results = pd.DataFrame(
        {
            "Date": portfolio_dates.iloc[: len(portfolio_value)],
            "Portfolio_Value": portfolio_value,
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
                    price > entry_price
                    if action == "SELL"
                    else price < entry_price
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
