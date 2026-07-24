"""
utils.py — Core ML helpers: data fetch, sequences, metrics, MC-Dropout forecast.

Fixes applied vs v1:
  • Scaler is NEVER fit_transform'd on full data — callers must pass pre-split arrays.
  • MASE replaces epsilon-hacked MAPE.
  • MC-Dropout forecast returns mean + confidence band instead of a single point.
  • Moving averages are computed on a given series slice, not globally.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Data Fetching ──────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, period: str = "5y",
                     cache_dir: str = "data") -> pd.DataFrame:
    """
    Download historical OHLCV data via yfinance.
    Saves a dated Parquet snapshot for reproducibility.
    """
    Path(cache_dir).mkdir(exist_ok=True)
    snap_path = Path(cache_dir) / f"{ticker}_{period}_{date.today()}.parquet"

    if snap_path.exists():
        df = pd.read_parquet(snap_path)
        print(f"  [cache] Loaded {ticker} ({period}) from {snap_path}")
        return df

    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    # Flatten MultiIndex if present (yfinance ≥ 0.2 may return one)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    df.to_parquet(snap_path)
    print(f"  [cache] Saved {ticker} snapshot → {snap_path}")
    return df


# ── Feature Engineering ────────────────────────────────────────────────────────

def add_moving_averages(series: pd.Series) -> pd.DataFrame:
    """
    Compute MA20 / MA50 / MA200 causally on a 1-D price series.
    Returns a DataFrame aligned to the series index.
    NOTE: Do NOT call this on the full dataframe before split — compute it
          per-split slice to guarantee no future information leakage if MAs
          are ever used as LSTM features.
    """
    df = pd.DataFrame({"Close": series})
    df["MA20"]  = series.rolling(window=20,  min_periods=1).mean()
    df["MA50"]  = series.rolling(window=50,  min_periods=1).mean()
    df["MA200"] = series.rolling(window=200, min_periods=1).mean()
    return df


def build_lag_features(series: pd.Series, n_lags: int = 60) -> pd.DataFrame:
    """
    Build a tabular lag feature matrix for tree-based models (XGBoost).
    Also includes: RSI-14, MACD, daily return.
    All features are computed causally (no look-ahead).
    """
    df = pd.DataFrame({"close": series.values}, index=series.index)

    # Lag features
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["close"].shift(lag)

    # Technical indicators
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    df["rsi"]     = 100 - (100 / (1 + rs))
    ema12         = df["close"].ewm(span=12, adjust=False).mean()
    ema26         = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]    = ema12 - ema26
    df["return1"] = df["close"].pct_change()

    df.dropna(inplace=True)
    return df


# ── Sequence Preparation for LSTM ─────────────────────────────────────────────

def create_sequences(scaled_data: np.ndarray, lookback: int = 60):
    """
    Create (X, y) sliding-window sequences from a SCALED 1-D array.

    Args:
        scaled_data: shape (N, 1) — already scaled, NO raw/full-data mix.
        lookback:    window length.

    Returns:
        X: shape (N-lookback, lookback, 1)
        y: shape (N-lookback,)
    """
    assert scaled_data.ndim == 2 and scaled_data.shape[1] == 1, \
        "scaled_data must be shape (N, 1)"
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback: i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray,
                    naive: np.ndarray | None = None) -> dict:
    """
    Compute evaluation metrics.

    Args:
        actual:    true prices (1-D)
        predicted: model predictions (1-D)
        naive:     naive (lag-1) predictions for MASE denominator.
                   If None, MASE is skipped.

    Returns dict with: RMSE, MAE, R², MASE (if naive provided).
    """
    actual    = actual.flatten()
    predicted = predicted.flatten()

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae  = float(mean_absolute_error(actual, predicted))
    r2   = float(r2_score(actual, predicted))

    out = {"RMSE": rmse, "MAE": mae, "R²": r2}

    if naive is not None:
        naive = naive.flatten()
        # MASE = MAE(model) / MAE(naive)
        mae_naive = float(mean_absolute_error(actual, naive))
        out["MASE"] = mae / (mae_naive + 1e-9)

    return out


def naive_forecast(prices: np.ndarray) -> np.ndarray:
    """Shift-by-1 naive baseline: predict tomorrow = today."""
    return prices[:-1]   # predicted[i] corresponds to actual[i+1]


# ── MC-Dropout Forecast with Confidence Band ───────────────────────────────────

def forecast_mc_dropout(model, last_sequence: np.ndarray,
                        scaler: MinMaxScaler,
                        n_days: int = 30,
                        n_samples: int = 100,
                        ci: float = 0.90) -> dict:
    """
    Monte-Carlo Dropout forecast — runs the model n_samples times with
    dropout ACTIVE at inference to estimate prediction uncertainty.

    Args:
        model:         Keras model (must have Dropout layers).
        last_sequence: shape (lookback, 1), SCALED values (train-scaler).
        scaler:        fitted on train only.
        n_days:        forecast horizon.
        n_samples:     MC iterations (100 is typical).
        ci:            confidence interval width (0.90 → 5th–95th percentile).

    Returns dict with keys: 'mean', 'lower', 'upper' — all shape (n_days,).
    """
    import tensorflow as tf

    all_preds = []
    for _ in range(n_samples):
        seq  = last_sequence.copy()   # (lookback, 1)
        run  = []
        for _ in range(n_days):
            x    = seq.reshape(1, len(seq), 1)
            # training=True keeps dropout active
            pred = model(x, training=True).numpy()[0, 0]
            run.append(pred)
            seq  = np.append(seq[1:], [[pred]], axis=0)
        all_preds.append(run)

    all_preds = np.array(all_preds)   # (n_samples, n_days)

    lower_q = (1 - ci) / 2
    upper_q = 1 - lower_q

    mean_scaled  = all_preds.mean(axis=0).reshape(-1, 1)
    lower_scaled = np.percentile(all_preds, lower_q * 100, axis=0).reshape(-1, 1)
    upper_scaled = np.percentile(all_preds, upper_q * 100, axis=0).reshape(-1, 1)

    return {
        "mean":  scaler.inverse_transform(mean_scaled).flatten(),
        "lower": scaler.inverse_transform(lower_scaled).flatten(),
        "upper": scaler.inverse_transform(upper_scaled).flatten(),
    }
