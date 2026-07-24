"""
generate_artifacts.py — One-shot script to generate the JSON artifacts
needed by the FastAPI backend from a pre-trained .keras model.

Run once after copying an old model, or after training without --period 5y:
    cd D:\stock-prediction\backend
    .\venv\Scripts\python generate_artifacts.py --ticker AAPL
"""
import sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    fetch_stock_data, create_sequences,
    compute_metrics, naive_forecast, forecast_mc_dropout,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",     default="AAPL")
    p.add_argument("--period",     default="5y")
    p.add_argument("--lookback",   type=int, default=60)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--mc_samples", type=int, default=50)
    return p.parse_args()

def main():
    args = parse_args()
    import tensorflow as tf

    model_path = Path("models") / f"{args.ticker}_lstm.keras"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run: python train_model.py --ticker", args.ticker)
        sys.exit(1)

    print(f"Loading {model_path}…")
    model = tf.keras.models.load_model(model_path)

    print(f"Fetching {args.ticker} data…")
    df = fetch_stock_data(args.ticker, period=args.period, cache_dir="data")
    close_prices = df["Close"].values.reshape(-1, 1)

    # Leak-free split
    split_idx    = int(len(close_prices) * (1 - args.test_ratio))
    train_raw    = close_prices[:split_idx]
    test_raw     = close_prices[split_idx - args.lookback:]

    scaler       = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)

    X_test, y_test = create_sequences(test_scaled, args.lookback)

    print("Generating predictions…")
    preds_scaled = model.predict(X_test, verbose=0)
    preds        = scaler.inverse_transform(preds_scaled).flatten()
    actual       = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    naive        = naive_forecast(actual)

    lstm_metrics = compute_metrics(actual[1:], preds[1:], naive)

    # Dummy XGBoost / naive metrics (will be overwritten on full retrain)
    xgb_metrics   = {"RMSE": 0.0, "MAE": 0.0, "R²": 0.0, "MASE": 0.0}
    naive_metrics  = compute_metrics(actual[1:], naive, naive)

    # Save predictions JSON
    test_dates = df.index[split_idx:split_idx + len(preds)]
    preds_df   = pd.DataFrame({
        "date":   [str(d.date()) for d in test_dates[:len(preds)]],
        "actual": actual[:len(preds)].tolist(),
        "lstm":   preds.tolist(),
    })
    preds_path = f"models/{args.ticker}_predictions.json"
    preds_df.to_json(preds_path, orient="records", indent=2)
    print(f"  ✅ {preds_path}")

    # MC-Dropout forecast
    print(f"MC-Dropout forecast ({args.mc_samples} samples)…")
    last_seq  = test_scaled[-args.lookback:]
    mc_result = forecast_mc_dropout(model, last_seq, scaler,
                                    n_days=30, n_samples=args.mc_samples)
    future_dates = pd.bdate_range(start=df.index[-1], periods=31)[1:]
    fc_df = pd.DataFrame({
        "date":  future_dates.strftime("%Y-%m-%d").tolist(),
        "mean":  mc_result["mean"].tolist(),
        "lower": mc_result["lower"].tolist(),
        "upper": mc_result["upper"].tolist(),
    })
    fc_path = f"models/{args.ticker}_forecast.json"
    fc_df.to_json(fc_path, orient="records", indent=2)
    print(f"  ✅ {fc_path}")

    # Comparison JSON
    comparison = {
        "lstm":   lstm_metrics,
        "xgb":    xgb_metrics,
        "naive":  naive_metrics,
        "cv_rmse_mean": 0.0,
        "cv_rmse_std":  0.0,
        "note": "XGBoost & CV metrics pending — run train_model.py for full comparison",
    }
    cmp_path = f"models/{args.ticker}_comparison.json"
    with open(cmp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  ✅ {cmp_path}")

    print(f"\n  LSTM → RMSE={lstm_metrics['RMSE']:.4f}  MASE={lstm_metrics.get('MASE',0):.4f}  R²={lstm_metrics['R²']:.4f}")
    print("\n✅ All artifacts ready. Start the API:")
    print("   .\\venv\\Scripts\\python -m uvicorn api:app --reload --port 8000\n")

if __name__ == "__main__":
    main()
