"""
tests/test_utils.py — Unit tests for core ML utilities.

Run:
    cd D:\\stock-prediction\\backend
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

# Add backend to path when running from tests/ dir
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils import (
    create_sequences,
    compute_metrics,
    naive_forecast,
    add_moving_averages,
    build_lag_features,
)


# ── create_sequences ───────────────────────────────────────────────────────────

class TestCreateSequences:

    def test_output_shapes(self):
        data = np.arange(100).reshape(-1, 1).astype(float)
        lookback = 20
        X, y = create_sequences(data, lookback)
        assert X.shape == (80, 20, 1), f"Bad X shape: {X.shape}"
        assert y.shape == (80,),       f"Bad y shape: {y.shape}"

    def test_no_boundary_leakage(self):
        """The sequence at index i must NOT contain value at i+lookback or beyond."""
        data = np.arange(50).reshape(-1, 1).astype(float)
        lookback = 10
        X, y = create_sequences(data, lookback)
        # X[0] should be [0..9], y[0] should be 10
        np.testing.assert_array_equal(X[0, :, 0], np.arange(0, 10))
        assert y[0] == 10.0

    def test_last_sequence_correct(self):
        data = np.arange(50).reshape(-1, 1).astype(float)
        lookback = 10
        X, y = create_sequences(data, lookback)
        # Last X should be [39..48], last y should be 49
        np.testing.assert_array_equal(X[-1, :, 0], np.arange(39, 49))
        assert y[-1] == 49.0

    def test_requires_2d_input(self):
        with pytest.raises(AssertionError):
            create_sequences(np.arange(50), lookback=10)   # 1-D → should fail

    def test_single_feature_col(self):
        data = np.zeros((30, 2))  # wrong shape
        with pytest.raises(AssertionError):
            create_sequences(data, lookback=5)


# ── compute_metrics ────────────────────────────────────────────────────────────

class TestComputeMetrics:

    def test_perfect_predictions(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = compute_metrics(arr, arr)
        assert m["RMSE"] == pytest.approx(0.0, abs=1e-9)
        assert m["MAE"]  == pytest.approx(0.0, abs=1e-9)
        assert m["R²"]   == pytest.approx(1.0, abs=1e-6)

    def test_mase_vs_naive(self):
        actual    = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        predicted = np.array([10.5, 11.5, 12.5, 13.5, 14.5])
        # naive predicts previous value
        naive = actual[:-1]    # [10, 11, 12, 13]
        m = compute_metrics(actual[1:], predicted[1:], naive)
        # MAE of model = 0.5 every step
        # MAE of naive = 1.0 every step (each step goes up by 1)
        assert m["MASE"] == pytest.approx(0.5, rel=1e-3)

    def test_known_rmse(self):
        actual    = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])   # each off by 1
        m = compute_metrics(actual, predicted)
        assert m["RMSE"] == pytest.approx(1.0, abs=1e-9)
        assert m["MAE"]  == pytest.approx(1.0, abs=1e-9)

    def test_no_naive_skips_mase(self):
        arr = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(arr, arr)
        assert "MASE" not in m


# ── naive_forecast ─────────────────────────────────────────────────────────────

class TestNaiveForecast:

    def test_shift_by_one(self):
        prices = np.array([10.0, 11.0, 12.0, 13.0])
        naive  = naive_forecast(prices)
        np.testing.assert_array_equal(naive, np.array([10.0, 11.0, 12.0]))

    def test_length(self):
        prices = np.arange(100, dtype=float)
        assert len(naive_forecast(prices)) == 99


# ── add_moving_averages ────────────────────────────────────────────────────────

class TestMovingAverages:

    def test_columns_present(self):
        s = pd.Series(np.random.randn(300).cumsum() + 100)
        df = add_moving_averages(s)
        assert "MA20"  in df.columns
        assert "MA50"  in df.columns
        assert "MA200" in df.columns

    def test_ma20_value(self):
        s = pd.Series(np.ones(50))
        df = add_moving_averages(s)
        # MA20 of all-ones should be 1.0 everywhere (after warmup)
        assert df["MA20"].iloc[-1] == pytest.approx(1.0, abs=1e-9)


# ── build_lag_features ─────────────────────────────────────────────────────────

class TestLagFeatures:

    def test_n_lag_columns(self):
        s = pd.Series(np.arange(200, dtype=float),
                      index=pd.date_range("2020-01-01", periods=200))
        df = build_lag_features(s, n_lags=10)
        lag_cols = [c for c in df.columns if c.startswith("lag_")]
        assert len(lag_cols) == 10

    def test_no_future_leakage_in_lags(self):
        """lag_1 at row i should equal close at row i-1."""
        s = pd.Series(np.arange(100, dtype=float),
                      index=pd.date_range("2020-01-01", periods=100))
        df = build_lag_features(s, n_lags=5)
        # At any row, lag_1 == close.shift(1)
        for i in range(1, len(df)):
            assert df["lag_1"].iloc[i] == pytest.approx(df["close"].iloc[i - 1])
