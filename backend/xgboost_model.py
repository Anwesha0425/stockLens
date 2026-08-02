"""
xgboost_model.py — XGBoost baseline for stock price prediction.

Uses lag features (lag-1 … lag-N) + RSI + MACD as inputs.
No scaler leakage: fit only on train split.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from utils import build_lag_features


def train_xgboost(close_series: pd.Series, split_idx: int,
                  n_lags: int = 60) -> tuple:
    """
    Train XGBoost on lag features from the train slice.

    Returns:
        model:   fitted XGBRegressor
        scaler:  fitted StandardScaler (for features only — target is return)
        cols:    feature column names (for consistent ordering at predict time)
    """
    feat_df = build_lag_features(close_series, n_lags=n_lags)

    # Align split to feature DataFrame index
    feat_train = feat_df[feat_df.index < close_series.index[split_idx]]

    X_train = feat_train.drop(columns=["close", "return1"]).values
    y_train = feat_train["return1"].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = XGBRegressor(
        n_estimators   = 500,
        learning_rate  = 0.05,
        max_depth      = 6,
        subsample      = 0.8,
        colsample_bytree=0.8,
        random_state   = 42,
        n_jobs         = -1,
    )
    model.fit(X_train, y_train,
              eval_set     = [(X_train, y_train)],
              verbose      = 0)

    cols = feat_df.drop(columns=["close", "return1"]).columns.tolist()
    return model, scaler, cols


def predict_xgboost(model, close_series: pd.Series,
                    split_idx: int, scaler: StandardScaler,
                    cols: list, n_lags: int = 60) -> np.ndarray:
    """
    Predict on the test slice using the already-fitted model + scaler.
    """
    feat_df  = build_lag_features(close_series, n_lags=n_lags)
    feat_test = feat_df[feat_df.index >= close_series.index[split_idx]]

    if feat_test.empty:
        return np.array([])

    X_test = feat_test[cols].values
    X_test = scaler.transform(X_test)   # ← transform only, no re-fit
    return model.predict(X_test)
