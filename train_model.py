import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

# Add project directory to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
import config

DATA_DIR = config.DATA_DIR
MODEL_DIR = config.MODEL_DIR
os.makedirs(MODEL_DIR, exist_ok=True)

LOOK_BACK = 60
FEATURES = 25  # Now includes technical indicators
EPOCHS = 50
BATCH_SIZE = 32


def load_data(ticker=None, flat=False):
    """Load data for a specific ticker (or use first ticker from config)"""
    if ticker is None:
        # Use first ticker from config
        ticker = config.TICKERS[0] if config.TICKERS else "SPY"

    ticker_dir = config.get_data_dir(ticker)

    if flat:
        X_train = np.load(f"{ticker_dir}/X_train_flat.npy")
        X_test = np.load(f"{ticker_dir}/X_test_flat.npy")
    else:
        X_train = np.load(f"{ticker_dir}/X_train.npy")
        X_test = np.load(f"{ticker_dir}/X_test.npy")

    y_train = np.load(f"{ticker_dir}/y_train.npy")
    y_test = np.load(f"{ticker_dir}/y_test.npy")

    # Direction labels (for classifier)
    y_dir_train = np.load(f"{ticker_dir}/y_dir_train.npy")
    y_dir_test = np.load(f"{ticker_dir}/y_dir_test.npy")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(
        f"Direction labels: up={sum(y_dir_train)}, down={len(y_dir_train) - sum(y_dir_train)}"
    )

    return X_train, X_test, y_train, y_test, y_dir_train, y_dir_test


def build_regression_model():
    """LSTM for price regression"""
    print("\nBuilding LSTM Regression Model...")

    model = Sequential(
        [
            LSTM(
                128,
                return_sequences=True,
                input_shape=(LOOK_BACK, FEATURES),
                kernel_regularizer=l2(0.001),
            ),
            Dropout(0.3),
            LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    return model


def build_classifier_model():
    """Bidirectional LSTM for directional classification"""
    print("\nBuilding Directional Classifier Model...")

    model = Sequential(
        [
            Bidirectional(
                LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, FEATURES))
            ),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )
    model.summary()
    return model


def build_classifier_model():
    """Bidirectional LSTM for directional classification"""
    print("\nBuilding Directional Classifier Model...")

    model = Sequential(
        [
            Bidirectional(
                LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, FEATURES))
            ),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )
    model.summary()
    return model


def train_regression(model, X_train, y_train, X_test, y_test, ticker):
    print("\nTraining Regression Model...")

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            f"{config.get_model_dir(ticker)}/regression_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )
    return history


def train_classifier(model, X_train, y_dir_train, X_test, y_dir_test, ticker):
    print("\nTraining Classifier Model...")

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            f"{config.get_model_dir(ticker)}/classifier_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_dir_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_dir_test),
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate_model(model, X_test, y_test, model_type="regression"):
    print(f"\nEvaluating {model_type} Model...")

    if model_type == "classifier":
        from sklearn.metrics import accuracy_score, classification_report

        preds = model.predict(X_test, verbose=0)
        preds_binary = (preds > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, preds_binary)
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Classification Report:")
        print(classification_report(y_test, preds_binary, target_names=["Down", "Up"]))
        return preds
    else:
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"  Test Loss (MSE): {test_loss:.6f}")
        print(f"  Test MAE: {test_mae:.6f}")

        predictions = model.predict(X_test, verbose=0)

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        return predictions


def train_xgboost(ticker):
    """Train XGBoost classifier on flat features"""
    print("\n" + "=" * 50)
    print("Training XGBoost Classifier")
    print("=" * 50)

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("Installing xgboost...")
        import subprocess

        subprocess.run(["pip", "install", "xgboost"], check=True)
        from xgboost import XGBClassifier

    X_train, X_test, y_train, y_test, y_dir_train, y_dir_test = load_data(
        ticker=ticker, flat=True
    )

    print(f"\nTraining XGBoost on {X_train.shape[1]} features...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    model.fit(
        X_train,
        y_dir_train,
        eval_set=[(X_test, y_dir_test)],
        verbose=50,
    )

    # Save model
    import pickle

    ticker_model_dir = config.get_model_dir(ticker)
    os.makedirs(ticker_model_dir, exist_ok=True)

    with open(f"{ticker_model_dir}/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Saved XGBoost model to {ticker_model_dir}/xgboost_model.pkl")

    # Evaluate
    from sklearn.metrics import accuracy_score

    preds = model.predict(X_test)
    acc = accuracy_score(y_dir_test, preds)
    print(f"XGBoost Test Accuracy: {acc:.4f}")

    return model


def train_single_ticker(ticker):
    """Train models for a single ticker"""
    print(f"\n{'=' * 60}")
    print(f"Training models for ticker: {ticker}")
    print(f"{'=' * 60}")

    # Ensure ticker model directory exists
    os.makedirs(config.get_model_dir(ticker), exist_ok=True)

    X_train, X_test, y_train, y_test, y_dir_train, y_dir_test = load_data(ticker=ticker)

    # 1. Train Regression Model
    print("\n" + "=" * 50)
    print("1. Training Regression Model")
    print("=" * 50)
    reg_model = build_regression_model()
    reg_history = train_regression(reg_model, X_train, y_train, X_test, y_test, ticker)
    reg_preds = evaluate_model(reg_model, X_test, y_test, "regression")

    # 2. Train Classifier Model
    print("\n" + "=" * 50)
    print("2. Training Directional Classifier")
    print("=" * 50)
    clf_model = build_classifier_model()
    clf_history = train_classifier(
        clf_model, X_train, y_dir_train, X_test, y_dir_test, ticker
    )
    clf_preds = evaluate_model(clf_model, X_test, y_dir_test, "classifier")

    # 3. Train XGBoost
    print("\n" + "=" * 50)
    print("3. Training XGBoost")
    print("=" * 50)
    xgb_model = train_xgboost(ticker)

    print("\n" + "=" * 50)
    print(f"Training Complete for {ticker}!")
    print("=" * 50)
    print("Models saved:")
    print(f"  - {config.get_model_dir(ticker)}/regression_model.keras")
    print(f"  - {config.get_model_dir(ticker)}/classifier_model.keras")
    print(f"  - {config.get_model_dir(ticker)}/xgboost_model.pkl")


def main():
    parser = argparse.ArgumentParser(description="Train trading models")
    parser.add_argument(
        "--ticker", type=str, default=None, help="Specific ticker to train"
    )
    parser.add_argument(
        "--all", action="store_true", help="Train all tickers from config"
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
    print(f"# MrKrabs Model Training")
    print(f"# Tickers: {tickers}")
    print(f"{'#' * 60}")

    # Train models for each ticker
    for ticker in tickers:
        train_single_ticker(ticker)

    print(f"\n\n{'#' * 60}")
    print(f"# All Training Complete")
    print(f"# Processed tickers: {tickers}")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
