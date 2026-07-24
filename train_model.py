"""
train_model.py — Train an improved Bidirectional LSTM model for stock price prediction.

Usage:
    python train_model.py --ticker AAPL --epochs 25 --lookback 60

The trained model is saved to: models/<TICKER>_lstm.h5
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend for saving plots
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

from utils import (
    fetch_stock_data,
    add_moving_averages,
    create_sequences,
    compute_metrics,
    forecast_future,
)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train LSTM stock price predictor")
    p.add_argument("--ticker",   default="AAPL",  help="Stock ticker symbol")
    p.add_argument("--period",   default="5y",    help="Data history period (e.g. 5y, 2y)")
    p.add_argument("--lookback", type=int, default=60,  help="Lookback window size")
    p.add_argument("--epochs",   type=int, default=25,  help="Training epochs")
    p.add_argument("--batch",    type=int, default=32,  help="Batch size")
    p.add_argument("--test_ratio", type=float, default=0.2, help="Test split ratio")
    return p.parse_args()


# ──────────────────────────────────────────────
# Build Model
# ──────────────────────────────────────────────

def build_lstm_model(lookback: int) -> tf.keras.Model:
    """
    Improved LSTM model:
      • Bidirectional LSTM for richer context
      • Two stacked LSTM layers
      • Dropout for regularisation
      • Dense output
    """
    model = Sequential([
        Input(shape=(lookback, 1)),
        Bidirectional(LSTM(units=128, return_sequences=True)),
        Dropout(0.2),
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=32),
        Dropout(0.1),
        Dense(units=16, activation="relu"),
        Dense(units=1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    return model


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{args.ticker}_lstm.keras"

    print(f"\n{'='*55}")
    print(f"  Stock Price Prediction — Training ({args.ticker})")
    print(f"{'='*55}\n")

    # 1. Fetch data ─────────────────────────────────────────
    print(f"[1/7] Downloading {args.ticker} data ({args.period})…")
    df = fetch_stock_data(args.ticker, period=args.period)
    df = add_moving_averages(df)
    print(f"      Records: {len(df)}  |  Range: {df.index[0].date()} → {df.index[-1].date()}")

    # 2. Prepare data ────────────────────────────────────────
    print("[2/7] Preparing sequences…")
    close_prices = df[["Close"]].values
    scaler       = MinMaxScaler(feature_range=(0, 1))
    scaled       = scaler.fit_transform(close_prices)

    split_idx    = int(len(scaled) * (1 - args.test_ratio))
    train_scaled = scaled[:split_idx]
    test_scaled  = scaled[split_idx - args.lookback :]   # include lookback overlap

    X_train, y_train = create_sequences(train_scaled, args.lookback)
    X_test,  y_test  = create_sequences(test_scaled,  args.lookback)
    print(f"      Train: {X_train.shape}  |  Test: {X_test.shape}")

    # 3. Build model ─────────────────────────────────────────
    print("[3/7] Building Bidirectional LSTM model…")
    model = build_lstm_model(args.lookback)

    # 4. Callbacks ───────────────────────────────────────────
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    # 5. Train ───────────────────────────────────────────────
    print(f"[4/7] Training for up to {args.epochs} epochs…")
    history = model.fit(
        X_train, y_train,
        epochs          = args.epochs,
        batch_size      = args.batch,
        validation_split= 0.1,
        callbacks       = callbacks,
        verbose         = 1,
    )

    # 6. Evaluate ────────────────────────────────────────────
    print("[5/7] Evaluating on test set…")
    preds_scaled = model.predict(X_test, verbose=0)
    preds        = scaler.inverse_transform(preds_scaled)
    actual       = scaler.inverse_transform(y_test.reshape(-1, 1))

    metrics = compute_metrics(actual, preds)
    print("\n  ── Metrics ──────────────────────────────")
    for k, v in metrics.items():
        print(f"    {k:10s}: {v:.4f}")
    print()

    # 7. Save plots ──────────────────────────────────────────
    print("[6/7] Saving evaluation plots…")

    # Loss curve
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#e6edf3")

    axes[0].plot(history.history["loss"],      color="#58a6ff", label="Train Loss")
    axes[0].plot(history.history["val_loss"],  color="#f78166", label="Val Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
    axes[0].legend(facecolor="#21262d", labelcolor="#c9d1d9")

    # Actual vs Predicted
    test_dates = df.index[split_idx:]
    n = min(len(test_dates), len(preds))
    axes[1].plot(test_dates[:n], actual[:n], color="#58a6ff", label="Actual",    linewidth=1.5)
    axes[1].plot(test_dates[:n], preds[:n],  color="#f78166", label="Predicted", linewidth=1.5)
    axes[1].set_title(f"{args.ticker} — Actual vs. LSTM Predicted")
    axes[1].set_xlabel("Date"); axes[1].set_ylabel("Price (USD)")
    axes[1].legend(facecolor="#21262d", labelcolor="#c9d1d9")
    fig.tight_layout()
    plot_path = f"models/{args.ticker}_evaluation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 8. Future forecast ─────────────────────────────────────
    print("[7/7] Generating 30-day future forecast…")
    last_seq     = scaled[-args.lookback:]            # (60, 1)
    future_preds = forecast_future(model, last_seq, scaler, n_days=30)
    print(f"      30-day forecast preview: {future_preds[:5].round(2)} …")

    np.save(f"models/{args.ticker}_future_forecast.npy", future_preds)

    print(f"\n✅ Model saved → {model_path}")
    print(f"   Plot  saved → {plot_path}")
    print(f"   Run `python app.py` to launch the dashboard.\n")


if __name__ == "__main__":
    main()
