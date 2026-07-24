"""
utils.py — Shared helper functions for the Stock Price Prediction project.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ──────────────────────────────────────────────
# Data Fetching
# ──────────────────────────────────────────────

def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download historical OHLCV data using yfinance."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df.dropna(inplace=True)
    return df


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA20, MA50, MA200 columns to the dataframe."""
    df = df.copy()
    df["MA20"]  = df["Close"].rolling(window=20).mean()
    df["MA50"]  = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    return df


# ──────────────────────────────────────────────
# Sequence Preparation for LSTM
# ──────────────────────────────────────────────

def create_sequences(scaled_data: np.ndarray, lookback: int = 60):
    """
    Create (X, y) sliding-window sequences.
    X shape: (samples, lookback, 1)
    y shape: (samples,)
    """
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback : i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return a dictionary with RMSE, MAE, MAPE, and R²."""
    actual    = actual.flatten()
    predicted = predicted.flatten()
    rmse  = np.sqrt(mean_squared_error(actual, predicted))
    mae   = mean_absolute_error(actual, predicted)
    mape  = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    r2    = r2_score(actual, predicted)
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R²": r2}


# ──────────────────────────────────────────────
# Future Forecast
# ──────────────────────────────────────────────

def forecast_future(model, last_sequence: np.ndarray, scaler: MinMaxScaler,
                    n_days: int = 30) -> np.ndarray:
    """
    Iteratively predict n_days into the future.
    last_sequence shape: (lookback, 1)  —  raw scaled values
    Returns array of shape (n_days,) in original price scale.
    """
    preds = []
    seq   = last_sequence.copy()                # (lookback, 1)
    for _ in range(n_days):
        x    = seq.reshape(1, len(seq), 1)
        pred = model.predict(x, verbose=0)[0, 0]
        preds.append(pred)
        seq  = np.append(seq[1:], [[pred]], axis=0)
    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten()
