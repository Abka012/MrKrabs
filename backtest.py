import os
import pickle
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore', category=UserWarning)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LOOK_BACK = 60


def load_scaler():
    with open(f"{DATA_DIR}/scaler.pkl", "rb") as f:
        return pickle.load(f)


def load_data():
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")
    raw_df = pd.read_csv(f"{DATA_DIR}/raw_data.csv", parse_dates=["Date"])
    return X_test, y_test, raw_df


def inverse_transform_close(scaler, scaled_values):
    dummy = np.zeros((len(scaled_values), 5))
    dummy[:, 3] = scaled_values
    return scaler.inverse_transform(dummy)[:, 3]


def predict_prices(model, X_test, y_test, scaler):
    predictions_scaled = model.predict(X_test, verbose=0).flatten()
    predictions = inverse_transform_close(scaler, predictions_scaled)
    actuals = inverse_transform_close(scaler, y_test)
    return predictions, actuals


def backtest(predictions, actuals, threshold=0.01):
    capital = 10000
    position = 0
    trades = []
    portfolio_value = [capital]

    for i in range(len(predictions) - 1):
        price_change = (actuals[i + 1] - actuals[i]) / actuals[i]
        pred_change = (predictions[i + 1] - actuals[i]) / actuals[i]

        if pred_change > threshold and position == 0:
            position = capital / actuals[i]
            trades.append(("BUY", i, actuals[i]))
            capital = 0
        elif pred_change < -threshold and position > 0:
            capital = position * actuals[i]
            trades.append(("SELL", i, actuals[i]))
            position = 0

        portfolio_value.append(position * actuals[i] if position > 0 else capital)

    if position > 0:
        capital = position * actuals[-1]
        trades.append(("SELL", len(predictions) - 1, actuals[-1]))
        portfolio_value[-1] = capital

    return capital, trades, portfolio_value


def calculate_metrics(portfolio_value, trades):
    portfolio = np.array(portfolio_value)
    returns = np.diff(portfolio) / portfolio[:-1]

    total_return = (portfolio[-1] - portfolio[0]) / portfolio[0]

    if len(returns) > 0 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    cummax = np.maximum.accumulate(portfolio)
    drawdowns = (portfolio - cummax) / cummax
    max_drawdown = drawdowns.min()

    wins = sum(
        1
        for t in trades
        if t[0] == "SELL"
        and len([x for x in trades if x[0] == "BUY" and x[1] < t[1]]) > 0
        and any(
            trades[j][2] > trades[k][2]
            for k, j in [
                (i, j)
                for i, t in enumerate(trades)
                if t[0] == "BUY"
                for j in range(i + 1, len(trades))
                if trades[j][0] == "SELL"
            ]
        )
    )
    wins = sum(
        1
        for i, t in enumerate(trades)
        if t[0] == "SELL"
        and any(trades[k][2] < t[2] for k in range(i) if trades[k][0] == "BUY")
    )
    losses = len([t for t in trades if t[0] == "SELL"]) - wins

    win_loss_ratio = wins / losses if losses > 0 else float("inf")

    return {
        "Total Return": f"{total_return * 100:.2f}%",
        "Final Value": f"${portfolio[-1]:.2f}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown * 100:.2f}%",
        "Total Trades": len([t for t in trades if t[0] == "BUY"]),
        "Wins": wins,
        "Losses": losses,
        "Win/Loss Ratio": f"{win_loss_ratio:.2f}"
        if win_loss_ratio != float("inf")
        else "inf",
    }


def walk_forward_analysis(model, raw_df, scaler, window_size=120):
    print("\nWalk-Forward Analysis (Paper Trading)...")

    test_data = raw_df.iloc[-window_size:][
        ["Open", "High", "Low", "Close", "Volume"]
    ].values

    predictions = []
    for i in range(len(test_data) - LOOK_BACK):
        seq = test_data[i : i + LOOK_BACK]
        seq_scaled = scaler.transform(seq)
        seq_scaled = seq_scaled.reshape(1, LOOK_BACK, 5)
        pred = model.predict(seq_scaled, verbose=0)[0, 0]
        predictions.append(pred)

    actual_prices = test_data[LOOK_BACK:, 3]

    wf_predictions = []
    for pred_scaled in predictions:
        dummy = np.zeros((1, 5))
        dummy[0, 3] = pred_scaled
        wf_predictions.append(scaler.inverse_transform(dummy)[0, 3])

    wf_return = (wf_predictions[-1] - wf_predictions[0]) / wf_predictions[0] * 100
    print(f"  Walk-forward return: {wf_return:.2f}%")

    return wf_predictions, actual_prices


def main():
    print("=" * 60)
    print("=" * 60)

    scaler = load_scaler()
    X_test, y_test, raw_df = load_data()

    model = load_model(f"{MODEL_DIR}/best_model.keras")
    print(f"Loaded model from {MODEL_DIR}/best_model.keras")

    predictions, actuals = predict_prices(model, X_test, y_test, scaler)

    print("\n--- Backtesting (Historical Simulation) ---")
    capital, trades, portfolio_value = backtest(predictions, actuals, threshold=0.005)

    metrics = calculate_metrics(portfolio_value, trades)
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    df_results = pd.DataFrame(
        {
            "Date": raw_df["Date"].iloc[-len(portfolio_value) :].values,
            "Portfolio_Value": portfolio_value,
        }
    )
    df_results.to_csv(f"{RESULTS_DIR}/backtest_results.csv", index=False)
    print(f"\nSaved backtest results to {RESULTS_DIR}/backtest_results.csv")

    print("\n--- Walk-Forward Analysis (Paper Trading) ---")
    wf_preds, wf_actuals = walk_forward_analysis(model, raw_df, scaler)

    df_wf = pd.DataFrame({"Predicted": wf_preds, "Actual": wf_actuals})
    df_wf.to_csv(f"{RESULTS_DIR}/walk_forward_results.csv", index=False)
    print(f"Saved walk-forward results to {RESULTS_DIR}/walk_forward_results.csv")

    return metrics


if __name__ == "__main__":
    main()
