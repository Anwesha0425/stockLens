"""
api.py — FastAPI backend serving ML predictions to the Next.js frontend.

Endpoints:
  GET /api/stock/{ticker}?period=2y   → OHLCV + moving averages
  GET /api/predict/{ticker}           → LSTM predictions + MC forecast band
  GET /api/compare?tickers=AAPL,MSFT  → multi-stock close prices
  GET /api/models/{ticker}            → metrics comparison table
  GET /api/returns/{ticker}?period=1y → daily returns data
  GET /health                         → liveness check
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from utils import fetch_stock_data, add_moving_averages

app = FastAPI(title="Stock Predictor API", version="2.0.0")

# Allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

MODELS_DIR = Path("models")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | list:
    if not path.exists():
        raise HTTPException(404, f"File not found: {path.name}. Run train_model.py first.")
    with open(path) as f:
        return json.load(f)

def _slice_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Helper to slice a dataframe to the requested period."""
    if period == "1mo": return df.tail(22)
    if period == "3mo": return df.tail(63)
    if period == "6mo": return df.tail(126)
    if period == "1y":  return df.tail(252)
    if period == "2y":  return df.tail(504)
    return df

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "date": str(date.today())}


@app.get("/api/stock/{ticker}")
def get_stock(ticker: str, period: str = "2y"):
    """OHLCV data + moving averages for a single ticker."""
    try:
        # Fetch 5y always so moving averages (like MA200) have enough history
        df = fetch_stock_data(ticker.upper(), period="5y", cache_dir="data")
    except Exception as e:
        raise HTTPException(500, f"yfinance error: {e}")

    close  = df["Close"]
    ma_df  = add_moving_averages(close)

    out_df = pd.DataFrame({
        "open":   df["Open"],
        "high":   df["High"],
        "low":    df["Low"],
        "close":  df["Close"],
        "volume": df["Volume"],
        "ma20":   ma_df["MA20"],
        "ma50":   ma_df["MA50"],
        "ma200":  ma_df["MA200"],
    })
    
    # Slice down to requested window AFTER computing MAs
    out_df = _slice_period(out_df, period)
    out_df.index = out_df.index.strftime("%Y-%m-%d")
    
    # Replace NaN with None (JSON null) instead of string "null"
    records = out_df.reset_index().rename(columns={"Date": "date", "index": "date"})
    records = records.replace({np.nan: None}).to_dict("records")
    return {"ticker": ticker.upper(), "data": records}


@app.get("/api/predict/{ticker}")
def get_predictions(ticker: str):
    """LSTM test-set predictions + optional MC-Dropout 30-day forecast."""
    t = ticker.upper()
    preds = _load_json(MODELS_DIR / f"{t}_predictions.json")
    
    forecast_path = MODELS_DIR / f"{t}_forecast.json"
    forecast = None
    if forecast_path.exists():
        with open(forecast_path) as f:
            forecast = json.load(f)
            
    return {
        "ticker":    t,
        "predictions": preds,
        "forecast":    forecast,
    }


@app.get("/api/models/{ticker}")
def get_models(ticker: str):
    """LSTM vs XGBoost vs Naive metrics table."""
    t = ticker.upper()
    comparison = _load_json(MODELS_DIR / f"{t}_comparison.json")
    return comparison


@app.get("/api/compare")
def get_compare(tickers: str = Query("AAPL,MSFT,NVDA,TSLA"), period: str = Query("1y")):
    """Normalised close prices for multiple tickers."""
    result = {}
    for t in tickers.split(","):
        t = t.strip().upper()
        try:
            # Fetch 5y then slice for consistency
            df = fetch_stock_data(t, period="5y", cache_dir="data")
            df = _slice_period(df, period)
            close = df["Close"]
            norm  = (close / close.iloc[0] * 100).round(3)
            result[t] = {
                "dates":  close.index.strftime("%Y-%m-%d").tolist(),
                "values": norm.tolist(),
                "raw":    close.round(2).tolist(),
            }
        except Exception as e:
            result[t] = {"error": str(e)}
    return result


@app.get("/api/returns/{ticker}")
def get_returns(ticker: str, period: str = "1y"):
    """Daily returns and rolling 21-day annualised volatility."""
    try:
        df = fetch_stock_data(ticker.upper(), period="5y", cache_dir="data")
    except Exception as e:
        raise HTTPException(500, str(e))

    close   = df["Close"]
    returns = close.pct_change().dropna() * 100
    vol21   = returns.rolling(21).std() * (252 ** 0.5)

    out = pd.DataFrame({
        "date":       returns.index.strftime("%Y-%m-%d"),
        "return_pct": returns.round(4).values,
        "vol21":      vol21.round(4).values,
    }).dropna()
    
    out = _slice_period(out, period)
    return {"ticker": ticker.upper(), "data": out.to_dict("records")}
