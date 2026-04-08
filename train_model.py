import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LOOK_BACK = 60
FEATURES = 5
EPOCHS = 50
BATCH_SIZE = 32


def load_data():
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def build_model():
    print("\nBuilding LSTM model...")

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, FEATURES)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    model.summary()
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    print("\nTraining model...")

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            f"{MODEL_DIR}/best_model.keras",
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


def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model...")

    train_loss, train_mae = model.evaluate(X_test[:1000], y_test[:1000], verbose=0)
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


def main():
    print("=" * 50)
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()

    model = build_model()

    history = train_model(model, X_train, y_train, X_test, y_test)

    predictions = evaluate_model(model, X_test, y_test)

    return model, history, predictions, y_test


if __name__ == "__main__":
    main()
