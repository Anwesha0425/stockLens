"""
generate_mock_artifacts.py — Generates JSON artifacts for the API without requiring TensorFlow.
Useful when the local CPU lacks AVX instructions to load the saved .keras model.
"""
import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from datetime import date
from utils import fetch_stock_data, compute_metrics, naive_forecast

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()
    
    ticker = args.ticker
    period = args.period
    
    print(f"Generating mock artifacts for {ticker}…")
    Path("models").mkdir(exist_ok=True)
    df = fetch_stock_data(ticker, period=period, cache_dir="data")
    close = df["Close"].values

    test_ratio = 0.2
    split_idx = int(len(close) * (1 - test_ratio))
    test_dates = df.index[split_idx:]
    actual = close[split_idx:]

    # Mock LSTM predictions (actual + some noise and slight lag)
    np.random.seed(42)
    noise = np.random.normal(0, np.std(actual) * 0.05, size=len(actual))
    preds = actual * 0.98 + noise + (np.std(actual) * 0.02)
    
    # Save predictions.json
    preds_df = pd.DataFrame({
        "date": [str(d.date()) for d in test_dates],
        "actual": actual.tolist(),
        "lstm": preds.tolist(),
    })
    preds_df.to_json(f"models/{ticker}_predictions.json", orient="records", indent=2)

    # Mock 30-day MC-Dropout forecast
    last_price = actual[-1]
    future_dates = pd.bdate_range(start=df.index[-1], periods=31)[1:]
    
    # Random walk with slight upward drift for mock mean
    drift = 0.0005
    volatility = 0.015
    steps = 30
    
    mean_forecast = [last_price]
    for _ in range(steps):
        ret = np.random.normal(drift, volatility)
        mean_forecast.append(mean_forecast[-1] * (1 + ret))
    mean_forecast = np.array(mean_forecast[1:])
    
    # Expanding confidence band
    expanding_std = last_price * volatility * np.sqrt(np.arange(1, steps + 1))
    lower = mean_forecast - 1.645 * expanding_std  # ~90% CI
    upper = mean_forecast + 1.645 * expanding_std

    fc_df = pd.DataFrame({
        "date": future_dates.strftime("%Y-%m-%d").tolist(),
        "mean": mean_forecast.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
    })
    fc_df.to_json(f"models/{ticker}_forecast.json", orient="records", indent=2)

    # Comparison metrics
    naive = naive_forecast(actual)
    # Re-align actual and naive for metrics
    lstm_metrics = compute_metrics(actual[1:], preds[1:], naive)
    
    # Mock XGBoost (slightly worse than LSTM for demonstration)
    xgb_preds = actual * 0.97 + np.random.normal(0, np.std(actual)*0.08, size=len(actual))
    xgb_metrics = compute_metrics(actual[1:], xgb_preds[1:], naive)
    
    naive_metrics = compute_metrics(actual[1:], naive, naive)

    comparison = {
        "lstm": lstm_metrics,
        "xgb": xgb_metrics,
        "naive": naive_metrics,
        "cv_rmse_mean": float(lstm_metrics["RMSE"] * 1.05),
        "cv_rmse_std": float(lstm_metrics["RMSE"] * 0.1),
        "note": "Mock generated due to CPU AVX limits",
    }
    with open(f"models/{ticker}_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("✅ Mock artifacts generated successfully!")

if __name__ == "__main__":
    main()
