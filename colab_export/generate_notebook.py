"""
generate_notebook_v3.py
Writes the v3 Stock_Prediction_Colab_Standalone.ipynb.

Root cause analysis of v1/v2 failures:
  - Models predicted absolute price → MinMaxScaler trained on train-set prices
    (~$130-190 for AAPL) cannot generalize to test-set prices ($200-230).
    Model outputs ~$160, actual is $220 → RMSE = $60, MASE = 15.

v3 fix:
  - Scale-invariant features (returns, ratios, oscillators)
  - Target: log-return (stationary ~N(0, 0.015)), not raw price
  - Evaluate in RETURN SPACE: MASE vs naive=0
  - Visualize: cumulative-return price reconstruction
"""

import json, os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [src],
    }

# ══════════════════════════════════════════════════════════════════════════════
CELL_TITLE = """\
# 📈 Stock Prediction — LSTM & XGBoost v3 (Stationary Return Prediction)

**Self-contained. Run All — no file uploads needed.**

## Why Previous Versions Failed (Root Cause)

Stock prices are **non-stationary** — AAPL drifted from ~$130 to ~$220 over 5 years.

| Version | Target | Why it failed |
|---|---|---|
| v1 | pct_change return | Reconstructed prices amplified tiny errors 10-25× |
| v2 | Close price directly | MinMaxScaler trained on $130-190 can't predict test prices at $200-230. Model output ~$160, actual ~$220 → RMSE=$53 |
| **v3** | **log-return (stationary)** | **Scale-invariant features + stationary target = no price-level drift** |

## v3 Architecture

- **Features**: 8 scale-invariant indicators (returns, price/MA ratios, RSI, normalized MACD/ATR, Bollinger %B, volume ratio)
- **Target**: `log(Close[t]/Close[t-1])` — stationary, ~N(0, 0.015)
- **Naive baseline**: predict 0 return (no change)
- **Evaluation**: MASE in return space → MASE < 1.0 = beats naive
"""

CELL_INSTALL = "!pip install -q yfinance xgboost scikit-learn tensorflow pyarrow matplotlib numpy pandas\n"

CELL_IMPORTS = """\
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, Input, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
TICKER     = 'AAPL'
PERIOD     = '5y'
LOOKBACK   = 60        # LSTM window (trading days)
EPOCHS     = 100
BATCH      = 32
TEST_RATIO = 0.2
N_SPLITS   = 5

os.makedirs('models', exist_ok=True)
os.makedirs('data',   exist_ok=True)
print(f'TF: {tf.__version__} | Ticker: {TICKER} | Lookback: {LOOKBACK}')
"""

CELL_DATA = """\
# ── Data & Scale-Invariant Feature Engineering ───────────────────────────────
#
# KEY: All features must be scale-invariant so the model works the same
# whether AAPL is at $130 or $220.  We achieve this by using:
#   - Returns (log-scale changes)
#   - Price/MA ratios (deviation from moving averages)
#   - RSI, Bollinger %B (already 0-100 / 0-1)
#   - MACD / Close, ATR / Close (normalized by current price)
#   - Volume ratio (vs rolling average)

def fetch_stock_data(ticker, period='5y'):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


def build_features(df):
    out = df.copy()

    # ── Moving Averages ───────────────────────────────────────────────────────
    out['MA20'] = out['Close'].rolling(20, min_periods=1).mean()
    out['MA50'] = out['Close'].rolling(50, min_periods=1).mean()

    # ── Log Return (target + feature) ─────────────────────────────────────────
    # log(Close[t] / Close[t-1]): stationary, scale-invariant
    out['log_return'] = np.log(out['Close'] / out['Close'].shift(1))

    # ── Price deviation from moving averages (scale-invariant) ────────────────
    out['close_vs_ma20'] = out['Close'] / out['MA20'] - 1   # e.g. +0.02 = 2% above MA20
    out['close_vs_ma50'] = out['Close'] / out['MA50'] - 1

    # ── RSI-14 (Wilder smoothing) — already 0..100 ────────────────────────────
    delta    = out['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    out['RSI'] = (100 - (100 / (1 + rs))) / 100.0   # scale to [0,1]

    # ── MACD normalized by Close (scale-invariant) ────────────────────────────
    ema12        = out['Close'].ewm(span=12, adjust=False).mean()
    ema26        = out['Close'].ewm(span=26, adjust=False).mean()
    out['macd_norm'] = (ema12 - ema26) / out['Close']   # ~[-0.05, +0.05]

    # ── Bollinger Band %B (already 0..1) ──────────────────────────────────────
    std20    = out['Close'].rolling(20, min_periods=1).std().fillna(0)
    upper_bb = out['MA20'] + 2 * std20
    lower_bb = out['MA20'] - 2 * std20
    out['BB_pct'] = (out['Close'] - lower_bb) / (upper_bb - lower_bb + 1e-9)

    # ── ATR normalized by Close (scale-invariant volatility) ──────────────────
    hl  = out['High'] - out['Low']
    hc  = (out['High'] - out['Close'].shift()).abs()
    lc  = (out['Low']  - out['Close'].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    out['atr_norm'] = tr.ewm(com=13, adjust=False).mean() / out['Close']  # ~[0.005, 0.03]

    # ── Volume ratio vs 20-day average ────────────────────────────────────────
    out['Vol_ratio'] = out['Volume'] / (out['Volume'].rolling(20, min_periods=1).mean() + 1)

    out.dropna(inplace=True)
    return out


print('[1/7] Fetching data...')
df_raw = fetch_stock_data(TICKER, PERIOD)
df     = build_features(df_raw)
print(f'  {len(df)} trading days  {df.index[0].date()} -> {df.index[-1].date()}')
mean_ret = df['log_return'].mean()
std_ret  = df['log_return'].std()
print(f'  Daily log-return: mean={mean_ret*100:.3f}%  std={std_ret*100:.3f}%')
"""

CELL_SPLIT = """\
# ── Train / Test Split ────────────────────────────────────────────────────────
#
# Features: 8 scale-invariant columns (no raw prices — they drift!)
# Target:   log_return  (stationary, scale-invariant)
# Scaler:   StandardScaler (zero-mean, unit-variance per feature)
#           Fits ONLY on train rows — no leakage.

FEATURE_COLS = [
    'log_return',       # yesterday's return (window features)
    'close_vs_ma20',    # % above/below MA20
    'close_vs_ma50',    # % above/below MA50
    'RSI',              # momentum [0, 1]
    'macd_norm',        # MACD / Close
    'BB_pct',           # Bollinger position [0, 1]
    'atr_norm',         # ATR / Close (normalised volatility)
    'Vol_ratio',        # volume anomaly
]
TARGET_COL = 'log_return'   # predict TOMORROW's log return

features = df[FEATURE_COLS].values          # (N, 8)
target   = df[TARGET_COL].values.reshape(-1, 1)  # (N, 1)
prices   = df['Close'].values               # keep raw prices for reconstruction plot

split_idx  = int(len(features) * (1 - TEST_RATIO))
split_date = df.index[split_idx]

train_feat = features[:split_idx]
train_targ = target[:split_idx]
test_feat  = features[split_idx - LOOKBACK:]   # include lookback context rows
test_targ  = target[split_idx - LOOKBACK:]

# StandardScaler fit on train only
feat_sc = StandardScaler()
targ_sc = StandardScaler()

train_feat_sc = feat_sc.fit_transform(train_feat)
test_feat_sc  = feat_sc.transform(test_feat)
train_targ_sc = targ_sc.fit_transform(train_targ)
test_targ_sc  = targ_sc.transform(test_targ)

print(f'[2/7] Data split')
print(f'  Train: {split_idx} rows ({df.index[0].date()} -> {split_date.date()})')
print(f'  Test:  {len(features)-split_idx} rows  ({split_date.date()} -> {df.index[-1].date()})')
print(f'  Features: {len(FEATURE_COLS)} (all scale-invariant)')
"""

CELL_SEQUENCES = """\
# ── Sequence Builder ──────────────────────────────────────────────────────────
def create_sequences(feat_sc, targ_sc, lookback):
    \"\"\"
    X[j] = feat_sc[j : j+lookback]   <- window of PAST features
    y[j] = targ_sc[j + lookback]     <- NEXT day's log-return to predict

    Note: feat_sc contains log_return for days j..j+lookback-1.
    Target is log_return for day j+lookback. No data leakage.
    \"\"\"
    X, y = [], []
    for i in range(lookback, len(feat_sc)):
        X.append(feat_sc[i - lookback: i, :])
        y.append(targ_sc[i, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


X_train, y_train = create_sequences(train_feat_sc, train_targ_sc, LOOKBACK)
X_test,  y_test  = create_sequences(test_feat_sc,  test_targ_sc,  LOOKBACK)
print(f'  X_train: {X_train.shape}  y_train: {y_train.shape}')
print(f'  X_test:  {X_test.shape}   y_test:  {y_test.shape}')
"""

CELL_METRICS = """\
# ── Metrics (evaluated in RETURN SPACE) ──────────────────────────────────────
#
# MASE = MAE(model) / MAE(naive)
# Naive for returns: always predict 0 (no change)
# MAE(naive) = mean(|actual_return - 0|) = mean(|actual_return|)
#
# MASE < 1.0  --> model beats "predict no change"
# MASE = 1.0  --> same as naive
# MASE > 1.0  --> worse than naive

def compute_return_metrics(actual_ret, pred_ret):
    \"\"\"Metrics computed in log-return space.\"\"\"
    actual_ret = np.array(actual_ret).flatten()
    pred_ret   = np.array(pred_ret).flatten()
    n = min(len(actual_ret), len(pred_ret))
    actual_ret, pred_ret = actual_ret[:n], pred_ret[:n]

    mae      = float(np.mean(np.abs(actual_ret - pred_ret)))
    rmse     = float(np.sqrt(np.mean((actual_ret - pred_ret)**2)))
    mae_pct  = mae * 100   # as percent per day
    rmse_pct = rmse * 100

    # MASE vs naive=0 (predict no change)
    mae_naive = float(np.mean(np.abs(actual_ret)))
    mase      = mae / (mae_naive + 1e-12)

    # Directional accuracy
    da = float(np.mean(np.sign(pred_ret) == np.sign(actual_ret))) * 100

    return {
        'MASE':      mase,
        'MAE_%':     mae_pct,
        'RMSE_%':    rmse_pct,
        'Dir_Acc_%': da,
    }


def reconstruct_prices(start_price, log_returns):
    \"\"\"Reconstruct price series from log returns: P[t] = P[0] * exp(sum(r[1..t]))\"\"\"
    return start_price * np.exp(np.cumsum(log_returns))


print('Return-space metrics helper loaded.')
"""

CELL_ARCH = """\
# ── LSTM Architecture ─────────────────────────────────────────────────────────
def build_lstm(lookback, n_features):
    \"\"\"
    Bidirectional LSTM predicting next-day log-return.

    Key design choices for return prediction:
    - BatchNorm after each LSTM block (handles varied return magnitudes)
    - Huber loss (robust to large return outliers e.g. earnings gaps)
    - Smaller dropout (0.15) — return signal is weak, don't over-regularise
    - tanh output capped via dense (natural for small ±0.05 returns)
    \"\"\"
    model = Sequential([
        Input(shape=(lookback, n_features)),

        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.15),

        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.15),

        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.10),

        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1),           # linear: predicts standardised log-return
    ])
    model.compile(
        optimizer=Adam(learning_rate=5e-4),  # lower LR for noisy return target
        loss='huber',
    )
    return model

print('LSTM architecture defined.')
build_lstm(LOOKBACK, len(FEATURE_COLS)).summary()
"""

CELL_CV = """\
# ── Walk-Forward Cross-Validation ─────────────────────────────────────────────
def walk_forward_cv(features_all, target_all, lookback, epochs, batch, n_splits):
    tscv         = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    print(f'\\n[3/7] Walk-Forward CV ({n_splits} folds) on log-return prediction')
    print('  ' + '-' * 62)

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(features_all)):
        if len(tr_idx) < lookback + 10 or len(va_idx) < 1:
            print(f'  Fold {fold_idx+1}: skipped (insufficient data)')
            continue

        fs = StandardScaler().fit(features_all[tr_idx])
        ts = StandardScaler().fit(target_all[tr_idx])

        va_start = max(0, va_idx[0] - lookback)
        X_tr, y_tr = create_sequences(
            fs.transform(features_all[tr_idx]),
            ts.transform(target_all[tr_idx]), lookback)
        X_va, y_va = create_sequences(
            fs.transform(features_all[va_start: va_idx[-1]+1]),
            ts.transform(target_all[va_start: va_idx[-1]+1]), lookback)

        if len(X_tr) == 0 or len(X_va) == 0:
            continue

        m_fold = build_lstm(lookback, features_all.shape[1])
        m_fold.fit(X_tr, y_tr, epochs=epochs, batch_size=batch,
                   validation_data=(X_va, y_va),
                   callbacks=[
                       EarlyStopping('val_loss', patience=12, restore_best_weights=True),
                       ReduceLROnPlateau('val_loss', factor=0.5, patience=6, verbose=0),
                   ], verbose=0)

        pred_sc = m_fold.predict(X_va, verbose=0)
        pred_ret  = ts.inverse_transform(pred_sc).flatten()
        actual_ret = ts.inverse_transform(y_va.reshape(-1,1)).flatten()

        fm = compute_return_metrics(actual_ret, pred_ret)
        fm['fold'] = fold_idx + 1
        fold_metrics.append(fm)

        print(f'  Fold {fold_idx+1}/{n_splits}  MASE={fm["MASE"]:.4f}  '
              f'MAE={fm["MAE_%"]:.4f}%  DA={fm["Dir_Acc_%"]:.1f}%')

    return fold_metrics


cv_metrics = walk_forward_cv(features, target, LOOKBACK, EPOCHS, BATCH, N_SPLITS)
if cv_metrics:
    mean_mase = np.mean([m['MASE'] for m in cv_metrics])
    mean_da   = np.mean([m['Dir_Acc_%'] for m in cv_metrics])
    print(f'\\n  CV  MASE={mean_mase:.4f}  Dir_Acc={mean_da:.1f}%')
    print(f'  (MASE < 1.0 = beats naive; Dir_Acc > 50% = predicts direction better than random)')
"""

CELL_TRAIN = """\
# ── Train Final LSTM on Full Train Set ────────────────────────────────────────
print('[4/7] Training final LSTM (target = log return)...')

lstm_model = build_lstm(LOOKBACK, len(FEATURE_COLS))
model_path = f'models/{TICKER}_lstm.keras'

history = lstm_model.fit(
    X_train, y_train,
    epochs           = EPOCHS,
    batch_size       = BATCH,
    validation_split = 0.1,
    callbacks        = [
        EarlyStopping('val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau('val_loss', factor=0.5, patience=8,
                          verbose=1, min_lr=1e-6),
    ],
    verbose = 1,
)
lstm_model.save(model_path)
print(f'  Model saved -> {model_path}')
"""

CELL_XGB = """\
# ── XGBoost on Log Returns ────────────────────────────────────────────────────
#
# XGBoost features: lag-1..20 daily log returns + yesterday's indicators
# Target: today's log return
# Naive: always predict 0 return

print('[5/7] Training XGBoost (target = log return)...')

XGB_LAGS = 20   # 20 days of lag-return features

def build_xgb_df(df, n_lags=20):
    \"\"\"
    Tabular feature matrix for XGBoost.
    All features are shifted by 1: yesterday's data predicts today's return.
    Target: today's log_return.
    \"\"\"
    out = pd.DataFrame(index=df.index)

    # Lag features of log returns
    for lag in range(1, n_lags + 1):
        out[f'ret_lag_{lag}'] = df['log_return'].shift(lag)

    # Yesterday's technical indicators
    for col in ['RSI', 'macd_norm', 'close_vs_ma20', 'close_vs_ma50',
                'BB_pct', 'atr_norm', 'Vol_ratio']:
        out[col] = df[col].shift(1)

    out['target'] = df['log_return']
    out.dropna(inplace=True)
    return out


xgb_df    = build_xgb_df(df, n_lags=XGB_LAGS)
feat_cols = [c for c in xgb_df.columns if c != 'target']

xgb_train  = xgb_df[xgb_df.index < split_date]
xgb_test   = xgb_df[xgb_df.index >= split_date]

xgb_sc     = StandardScaler()
X_xtr      = xgb_sc.fit_transform(xgb_train[feat_cols].values)
y_xtr      = xgb_train['target'].values
X_xte      = xgb_sc.transform(xgb_test[feat_cols].values)
y_xte      = xgb_test['target'].values

xgb_model = XGBRegressor(
    n_estimators     = 1000,
    learning_rate    = 0.01,
    max_depth        = 4,          # shallow trees for noisy returns
    subsample        = 0.7,
    colsample_bytree = 0.6,
    min_child_weight = 5,
    reg_lambda       = 2.0,
    reg_alpha        = 0.5,
    random_state     = 42,
    n_jobs           = -1,
)
xgb_model.fit(X_xtr, y_xtr,
              eval_set=[(X_xte, y_xte)],
              verbose=200)

xgb_ret_pred = xgb_model.predict(X_xte)
xgb_ret_actual = y_xte
print(f'  XGBoost trained | test days: {len(xgb_ret_pred)}')
"""

CELL_EVAL = """\
# ── Evaluate All Models in Return Space ──────────────────────────────────────
print('[6/7] Evaluating on holdout test set...')

# LSTM predictions (inverse-transform from standardised space)
lstm_pred_sc  = lstm_model.predict(X_test, verbose=0)
lstm_ret_pred = targ_sc.inverse_transform(lstm_pred_sc).flatten()
lstm_ret_actual = targ_sc.inverse_transform(y_test.reshape(-1,1)).flatten()

# Align LSTM and XGBoost by date
lstm_dates   = df.index[split_idx: split_idx + len(lstm_ret_actual)]
xgb_dates    = xgb_test.index
common_dates = lstm_dates.intersection(xgb_dates)

lstm_ret_s    = pd.Series(lstm_ret_pred,   index=lstm_dates)
actual_ret_s  = pd.Series(lstm_ret_actual, index=lstm_dates)
xgb_ret_s     = pd.Series(xgb_ret_pred,   index=xgb_dates)

actual_r = actual_ret_s.loc[common_dates].values
lstm_r   = lstm_ret_s.loc[common_dates].values
xgb_r    = xgb_ret_s.loc[common_dates].values
naive_r  = np.zeros(len(actual_r))   # naive always predicts 0 return

lstm_met  = compute_return_metrics(actual_r, lstm_r)
xgb_met   = compute_return_metrics(actual_r, xgb_r)
naive_met = compute_return_metrics(actual_r, naive_r)

print()
print('  -- Model Comparison (Return Space) --------------------------------')
print(f"  {'Metric':12s}  {'LSTM':>10s}  {'XGBoost':>10s}  {'Naive(0)':>10s}")
print('  ' + '-' * 50)
for k in ['MASE', 'MAE_%', 'RMSE_%', 'Dir_Acc_%']:
    lv = lstm_met.get(k, 0.0)
    xv = xgb_met.get(k, 0.0)
    nv = naive_met.get(k, 0.0)
    print(f'  {k:12s}  {lv:10.4f}  {xv:10.4f}  {nv:10.4f}')
print()
print('  MASE < 1.0  ->  model beats naive (predict-no-change) baseline')
print('  Dir_Acc > 50% -> model predicts direction better than random')
print()
for name, m in [('LSTM', lstm_met), ('XGBoost', xgb_met)]:
    mase = m['MASE']
    da   = m['Dir_Acc_%']
    mase_ok = 'BEATS naive' if mase < 1.0 else 'worse than naive'
    da_ok   = 'above random' if da > 50 else 'at or below random'
    icon_m = 'OK' if mase < 1.0 else 'FAIL'
    icon_d = 'OK' if da > 50 else 'INFO'
    print(f'  [{icon_m}] {name}: MASE={mase:.4f} ({mase_ok})')
    print(f'  [{icon_d}] {name}: Dir_Acc={da:.1f}% ({da_ok})')
    print()

# ── Reconstruct prices from predicted returns (for visualization) ─────────────
# price[t] = last_train_price * exp( sum of predicted log returns up to t )
start_price    = df['Close'].values[split_idx - 1]
actual_prices  = df['Close'].values[split_idx: split_idx + len(actual_r)]

actual_prices_from_ret  = reconstruct_prices(start_price, actual_r)
lstm_prices_from_ret    = reconstruct_prices(start_price, lstm_r)
xgb_prices_from_ret     = reconstruct_prices(start_price, xgb_r)
naive_prices_from_ret   = np.full(len(actual_r), start_price)  # naive: stays flat

print('  Reconstructed prices for visualization.')
print(f'  Start price: ${start_price:.2f} | Actual end: ${actual_prices[-1]:.2f}')
print(f'  LSTM reconstructed end:    ${lstm_prices_from_ret[-1]:.2f}')
print(f'  XGBoost reconstructed end: ${xgb_prices_from_ret[-1]:.2f}')
"""

CELL_SAVE = """\
# ── Save Outputs & Plot ───────────────────────────────────────────────────────
print('[7/7] Saving outputs...')

comparison = {
    'ticker': TICKER,
    'evaluation': 'log-return space (MASE vs naive=0)',
    'lstm':   {k: float(v) for k, v in lstm_met.items()},
    'xgb':    {k: float(v) for k, v in xgb_met.items()},
    'naive':  {k: float(v) for k, v in naive_met.items()},
}
cmp_path = f'models/{TICKER}_comparison.json'
with open(cmp_path, 'w') as fh:
    json.dump(comparison, fh, indent=2)

# Predictions JSON
n_save = min(len(common_dates), len(lstm_r), len(xgb_r))
pd.DataFrame({
    'date':          [str(d.date()) for d in common_dates[:n_save]],
    'actual_return': actual_r[:n_save].tolist(),
    'lstm_return':   lstm_r[:n_save].tolist(),
    'xgb_return':    xgb_r[:n_save].tolist(),
    'actual_price':  actual_prices[:n_save].tolist(),
    'lstm_price':    lstm_prices_from_ret[:n_save].tolist(),
    'xgb_price':     xgb_prices_from_ret[:n_save].tolist(),
}).to_json(f'models/{TICKER}_predictions.json', orient='records', indent=2)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 5))
fig.patch.set_facecolor('#0d1117')
for ax in axes:
    ax.set_facecolor('#161b22')
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.tick_params(colors='#c9d1d9')
    ax.xaxis.label.set_color('#c9d1d9')
    ax.yaxis.label.set_color('#c9d1d9')
    ax.title.set_color('#e6edf3')

# 1. Training loss
axes[0].plot(history.history['loss'],     color='#58a6ff', label='Train', lw=1.5)
axes[0].plot(history.history['val_loss'], color='#f78166', label='Val',   lw=1.5)
axes[0].set_title('LSTM Training Loss (Huber)')
axes[0].set_xlabel('Epoch')
axes[0].legend(facecolor='#21262d', labelcolor='#c9d1d9')

# 2. Predicted vs actual log returns (scatter)
pn = min(n_save, 200)  # first 200 days
axes[1].plot(common_dates[:pn], actual_r[:pn] * 100,
             color='#8b949e', label='Actual return %', lw=1.2, alpha=0.8)
axes[1].plot(common_dates[:pn], lstm_r[:pn] * 100,
             color='#58a6ff', label='LSTM predicted %', lw=1.2, alpha=0.8)
axes[1].plot(common_dates[:pn], xgb_r[:pn] * 100,
             color='#f0883e', label='XGB predicted %', lw=1.0, ls='--', alpha=0.8)
axes[1].axhline(0, color='#444', lw=0.8, ls=':')
axes[1].set_title('Daily Log Returns: Actual vs Predicted')
axes[1].set_ylabel('Return (%)')
axes[1].legend(facecolor='#21262d', labelcolor='#c9d1d9', fontsize=8)

# 3. Cumulative price reconstruction
pn2 = n_save
lstm_lbl = 'LSTM  (MASE={:.3f}, DA={:.1f}%)'.format(lstm_met['MASE'], lstm_met['Dir_Acc_%'])
xgb_lbl  = 'XGBoost (MASE={:.3f}, DA={:.1f}%)'.format(xgb_met['MASE'], xgb_met['Dir_Acc_%'])
axes[2].plot(common_dates[:pn2], actual_prices[:pn2],
             color='#8b949e', label='Actual price', lw=2)
axes[2].plot(common_dates[:pn2], lstm_prices_from_ret[:pn2],
             color='#58a6ff', label=lstm_lbl, lw=1.5)
axes[2].plot(common_dates[:pn2], xgb_prices_from_ret[:pn2],
             color='#f0883e', label=xgb_lbl, lw=1.5, ls='--')
axes[2].set_title(f'{TICKER} — Cumulative Price (from predicted returns)')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Price (USD)')
axes[2].legend(facecolor='#21262d', labelcolor='#c9d1d9', fontsize=8)

fig.tight_layout()
plot_path = f'models/{TICKER}_evaluation.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()

print(f'\\n  Done!')
print(f'  Model    -> models/{TICKER}_lstm.keras')
print(f'  Plot     -> {plot_path}')
print(f'  Metrics  -> {cmp_path}')
print()
print('  -- Final Metrics (Return Space) ---------------------------------')
print(f"  {'Metric':12s}  {'LSTM':>10s}  {'XGBoost':>10s}  {'Naive(0)':>10s}")
print('  ' + '-' * 50)
for k in ['MASE', 'MAE_%', 'RMSE_%', 'Dir_Acc_%']:
    lv = lstm_met.get(k, 0.0)
    xv = xgb_met.get(k, 0.0)
    nv = naive_met.get(k, 0.0)
    print(f'  {k:12s}  {lv:10.4f}  {xv:10.4f}  {nv:10.4f}')
"""

# ══════════════════════════════════════════════════════════════════════════════
nb = {
    "cells": [
        md(CELL_TITLE),
        code(CELL_INSTALL),
        code(CELL_IMPORTS),
        code(CELL_DATA),
        code(CELL_SPLIT),
        code(CELL_SEQUENCES),
        code(CELL_METRICS),
        code(CELL_ARCH),
        code(CELL_CV),
        code(CELL_TRAIN),
        code(CELL_XGB),
        code(CELL_EVAL),
        code(CELL_SAVE),
    ],
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Stock_Prediction_Colab_Standalone.ipynb")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=2, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells:   {len(nb['cells'])}")
