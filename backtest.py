import os
import pickle
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

warnings.filterwarnings('ignore', category=UserWarning)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LOOK_BACK = 60
THRESHOLD = 0.45  # Classifier threshold (45% - trade on any upward signal)


def load_scaler():
    with open(f"{DATA_DIR}/scaler.pkl", "rb") as f:
        return pickle.load(f)


def load_data():
    raw_df = pd.read_csv(f"{DATA_DIR}/raw_data.csv", parse_dates=["Date"])
    y_dir_test = np.load(f"{DATA_DIR}/y_dir_test.npy")
    return raw_df, y_dir_test


def get_feature_columns():
    return [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'SMA_20', 'SMA_50', 'SMA_200',
        'Price_SMA20_Ratio', 'Price_SMA50_Ratio',
        'BB_Upper', 'BB_Lower', 'BB_Width',
        'ATR', 'Momentum_5', 'Momentum_10', 'Momentum_20',
        'Volume_Ratio', 'Daily_Return', 'Volatility_10', 'Volatility_20'
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


def load_classifier_model():
    return load_model(f"{MODEL_DIR}/classifier_model.keras")


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
        seq = scaled_features[i-look_back:i].reshape(1, look_back, 25)
        prob = model.predict(seq, verbose=0)[0, 0]
        probabilities.append(prob)
        
        # Actual direction: price went up or down next day
        current_price = raw_df['Close'].iloc[i]
        next_price = raw_df['Close'].iloc[i+1]
        actual_directions.append(1 if next_price > current_price else 0)
    
    return np.array(probabilities), np.array(actual_directions)


def backtest_classifier(probabilities, actual_directions, prices, threshold=0.5):
    initial_capital = 10000
    capital = initial_capital
    position = 0
    trades = []
    portfolio_value = [capital]

    for i in range(len(probabilities)):
        prob_up = probabilities[i]
        prob_down = 1 - prob_up
        current_price = prices[i]
        
        # Trading logic with classifier
        if prob_up > threshold and position <= 0:
            # Enter LONG
            if position < 0:
                # Cover short first
                shares = abs(position)
                pnl = shares * (trades[-1][2] - current_price)
                capital = capital + pnl
                trades.append(("COVER", i, current_price))
            
            shares = int((capital * 0.15) / current_price)
            position = shares
            capital = capital - (shares * current_price)
            trades.append(("LONG", i, current_price))
            
        elif prob_down > threshold and position >= 0:
            # Enter SHORT
            if position > 0:
                # Sell long first
                capital = capital + (position * current_price)
                trades.append(("SELL", i, current_price))
            
            shares = int((capital * 0.15) / current_price)
            position = -shares
            capital = capital + (shares * current_price)
            trades.append(("SHORT", i, current_price))
            
        elif (prob_up > threshold and position < 0) or (prob_down > threshold and position > 0):
            # Exit positions
            if position > 0:
                capital = capital + (position * current_price)
                trades.append(("SELL", i, current_price))
            elif position < 0:
                shares = abs(position)
                pnl = shares * (trades[-1][2] - current_price)
                capital = capital + pnl
                trades.append(("COVER", i, current_price))
            position = 0
        
        # Calculate portfolio value
        if position != 0:
            value = capital + abs(position) * current_price
        else:
            value = capital
        portfolio_value.append(value)
    
    # Close final position
    if position != 0:
        final_price = prices[-1]
        if position > 0:
            capital = capital + (position * final_price)
            trades.append(("SELL", len(probabilities)-1, final_price))
        elif position < 0:
            shares = abs(position)
            pnl = shares * (trades[-1][2] - final_price)
            capital = capital + pnl
            trades.append(("COVER", len(probabilities)-1, final_price))
        portfolio_value[-1] = capital
    
    return capital, trades, portfolio_value


def calculate_metrics(portfolio_value, trades, initial_capital=10000):
    portfolio = np.array(portfolio_value)
    
    with np.errstate(divide='ignore', invalid='ignore'):
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
                if price > entry_price:
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
        "Win/Loss Ratio": f"{win_loss_ratio:.2f}" if win_loss_ratio != float("inf") else "inf",
    }


def main():
    print("=" * 60)
    print("=" * 60)
    
    print("\nLoading classifier model...")
    model = load_classifier_model()
    print(f"Loaded model from {MODEL_DIR}/classifier_model.keras")
    
    raw_df, y_dir_test = load_data()
    scaler = load_scaler()
    
    print(f"Raw data shape: {raw_df.shape}")
    print(f"Direction labels: up={sum(y_dir_test)}, down={len(y_dir_test)-sum(y_dir_test)}")
    
    # Get predictions and actuals for test period
    # Start from test data portion (after 80% of data)
    test_start_idx = int(len(raw_df) * 0.8)
    test_df = raw_df.iloc[test_start_idx:].reset_index(drop=True)
    
    probabilities, actuals = predict_directions(model, scaler, test_df)
    prices = test_df['Close'].iloc[LOOK_BACK:].values
    
    preds = (probabilities > 0.5).astype(int)
    actuals_arr = actuals
    accuracy = (preds == actuals_arr).mean()
    print(f"Predictions: {len(probabilities)}")
    print(f"Prediction accuracy: {accuracy:.2%}")
    
    print("\n--- Backtesting (Classifier) ---")
    capital, trades, portfolio_value = backtest_classifier(probabilities, actuals, prices, threshold=THRESHOLD)
    
    metrics = calculate_metrics(portfolio_value, trades)
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Save results
    df_results = pd.DataFrame({
        "Date": test_df['Close'].iloc[LOOK_BACK:].index[:len(portfolio_value)],
        "Portfolio_Value": portfolio_value,
    })
    df_results.to_csv(f"{RESULTS_DIR}/backtest_results.csv", index=False)
    print(f"\nSaved backtest results to {RESULTS_DIR}/backtest_results.csv")
    
    print(f"\nTrades made: {len(trades)}")
    for t in trades[-10:]:
        print(f"  {t}")


if __name__ == "__main__":
    main()