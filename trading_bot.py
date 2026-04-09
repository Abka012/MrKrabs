import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging
import pickle
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOOK_BACK = 60
TICKERS = ["SPY"]
THRESHOLD = 0.45  # Classifier threshold (45% - trade on any upward signal)

# Import config for ticker support
import sys

sys.path.insert(0, PROJECT_DIR)
import config

# Use TICKERS from config if available
if hasattr(config, "TICKERS"):
    TICKERS = config.TICKERS
INITIAL_CAPITAL = 10000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/trading_bot.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


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


class TradingBot:
    def __init__(self, ticker=None):
        self.ticker = ticker
        if ticker is None:
            ticker = "SPY"

        logger.info(f"Initializing Trading Bot for {ticker}...")
        self.scaler = self.load_scaler(ticker)
        self.model = self.load_model(ticker)
        self.position = 0  # positive = long, negative = short
        self.capital = INITIAL_CAPITAL
        self.portfolio_value = INITIAL_CAPITAL

    def load_scaler(self, ticker=None):
        if ticker is None:
            ticker = self.ticker if self.ticker else "SPY"
        data_dir = os.path.join(DATA_DIR, ticker)
        with open(f"{data_dir}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded scaler for {ticker}")
        return scaler

    def load_model(self, ticker=None):
        if ticker is None:
            ticker = self.ticker if self.ticker else "SPY"
        model_dir = os.path.join(MODEL_DIR, ticker)
        model = load_model(f"{model_dir}/classifier_model.keras")
        logger.info(f"Loaded classifier model for {ticker}")
        return model

    def fetch_recent_data(self, ticker, days=120):
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = stock.history(start=start_date, end=end_date)
        df = df.dropna(subset=["Close"])

        if len(df) < LOOK_BACK:
            logger.warning(f"Insufficient data for {ticker}: {len(df)} rows")
            return None

        logger.info(f"Fetched {len(df)} days of data for {ticker}")
        return df

    def fetch_recent_data_single(self, days=120):
        """Fetch data for the single ticker set in __init__"""
        return self.fetch_recent_data(self.ticker, days)

    def prepare_features(self, df):
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

        return df[feature_cols]

    def predict_direction(self, features):
        scaled = self.scaler.transform(features.values)
        seq = scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 25)
        prob_up = self.model.predict(seq, verbose=0)[0, 0]
        return prob_up

    def run_strategy(self, ticker):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Running strategy for {ticker}")
        logger.info(f"{'=' * 50}")

        data = self.fetch_recent_data(ticker)
        if data is None:
            logger.error(f"Skipping {ticker} due to insufficient data")
            return

        current_price = data["Close"].iloc[-1]
        logger.info(f"Current price: ${current_price:.2f}")

        features = self.prepare_features(data)
        prob_up = self.predict_direction(features)
        prob_down = 1 - prob_up

        logger.info(f"Probability UP: {prob_up:.2%}")
        logger.info(f"Probability DOWN: {prob_down:.2%}")

        # Trading logic
        # LONG when prob_up > THRESHOLD
        # SHORT when prob_down > THRESHOLD

        if prob_up > THRESHOLD and self.position <= 0:
            # Buy LONG
            available = self.capital * 0.15
            shares = int(available / current_price)
            self.position = shares
            self.capital = self.capital - (shares * current_price)
            logger.info(
                f"BUY (LONG) {shares} shares of {ticker} at ${current_price:.2f}"
            )

        elif prob_down > THRESHOLD and self.position >= 0:
            # Sell existing long and go SHORT
            if self.position > 0:
                self.capital = self.capital + (self.position * current_price)
                logger.info(f"SELL {self.position} shares at ${current_price:.2f}")

            # Enter short
            available = self.capital * 0.15
            shares = int(available / current_price)
            self.position = -shares
            self.capital = self.capital + (shares * current_price)
            logger.info(f"SHORT {shares} shares of {ticker} at ${current_price:.2f}")

        elif (prob_up > THRESHOLD and self.position < 0) or (
            prob_down > THRESHOLD and self.position > 0
        ):
            # Cover positions
            if self.position > 0:
                self.capital = self.capital + (self.position * current_price)
                logger.info(f"SELL {self.position} shares at ${current_price:.2f}")
                self.position = 0
            elif self.position < 0:
                shares = abs(self.position)
                self.capital = self.capital + (shares * current_price)
                logger.info(f"COVER {shares} shares at ${current_price:.2f}")
                self.position = 0
        else:
            logger.info(f"HOLD - No clear signal")

        self.portfolio_value = self.capital + (abs(self.position) * current_price)
        logger.info(f"Portfolio Value: ${self.portfolio_value:.2f}")
        logger.info(f"  Cash: ${self.capital:.2f}")
        logger.info(f"  Position: {self.position:.2f} shares")

    def run_all_tickers(self):
        for ticker in TICKERS:
            try:
                self.run_strategy(ticker)
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
            time.sleep(1)

    def run_single_ticker(self, ticker=None):
        """Run strategy for a single ticker"""
        if ticker is None:
            ticker = self.ticker
        self.run_strategy(ticker)

    def get_performance_summary(self):
        total_return = (
            (self.portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
        ) * 100
        logger.info(f"\n{'=' * 50}")
        logger.info("PERFORMANCE SUMMARY")
        logger.info(f"{'=' * 50}")
        logger.info(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
        logger.info(f"Current Value: ${self.portfolio_value:.2f}")
        logger.info(f"Total Return: {total_return:+.2f}%")
        logger.info(
            f"Current Position: {'Long' if self.position > 0 else 'Short' if self.position < 0 else 'Flat'}"
        )


def main():
    parser = argparse.ArgumentParser(description="LSTM Classifier Trading Bot")
    parser.add_argument(
        "--ticker", type=str, default=None, help="Specific ticker to trade"
    )
    parser.add_argument(
        "--all", action="store_true", help="Trade all tickers from config"
    )
    args = parser.parse_args()

    logger.info("Starting LSTM Classifier Trading Bot")

    if args.all:
        # Run all tickers
        bot = TradingBot()
        bot.run_all_tickers()
    elif args.ticker:
        # Run specific ticker
        bot = TradingBot(ticker=args.ticker)
        bot.run_single_ticker(args.ticker)
    else:
        # Default: run all tickers
        bot = TradingBot()
        bot.run_all_tickers()

    bot.get_performance_summary()

    logger.info("Trading bot cycle complete")


if __name__ == "__main__":
    main()
