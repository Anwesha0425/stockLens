"""
generate_notebook_v7.py  --  Hybrid LSTM Regressor + XGBoost Classifier

Key changes over v4/v6:
  1. LSTM reverts to plain BiLSTM Regressor (v3 setup):
       - Target: continuous log_return
       - Loss: Huber (smooth gradient, robust to outliers)
       - Architecture: Input -> BiLSTM(128) -> Dropout(0.2)
                              -> BiLSTM(64)  -> Dropout(0.2)
                              -> Dense(32)   -> Dense(1, linear)
       - No Attention, no BatchNorm (cleaner gradient flow)
  2. XGBoost kept as Binary Classifier (v6 setup):
       - Target: 1 if log_return > 0 else 0
       - Loss: binary:logistic
  3. Ensemble fusion with scipy-optimised weight W:
       - LSTM output -> z-score -> sigmoid -> pseudo-probability
       - W = argmax Sharpe on first-half of test set (validation split)
       - Final: Ensemble_Proba = W * LSTM_Proba + (1-W) * XGB_Proba
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
# 📈 Stock Prediction v7 — Hybrid LSTM Regressor + XGBoost Classifier

**Self-contained. Run All — no file uploads needed.**

## v7 Architecture: Best of Both Worlds

| Model | Task | Why |
|---|---|---|
| **BiLSTM** | *Regressor* (continuous log-return, Huber loss) | Smoother loss surface -> better directional signal from sign of predicted return |
| **XGBoost** | *Classifier* (binary UP/DOWN, logistic loss) | Tree partitioning excels at carving out high-probability DOWN regimes |
| **Ensemble** | Weighted average of calibrated probabilities | scipy-optimised weight W maximises Sharpe on a validation half of the test period |

### Ensemble Fusion Detail
1. LSTM outputs a predicted daily return (e.g. +0.002 or -0.001).
2. Z-score it: `z = lstm_ret / std(lstm_ret)` -> sigmoid -> `LSTM_Proba`
3. XGBoost outputs `P(UP)` directly.
4. `Ensemble_Proba = W * LSTM_Proba + (1-W) * XGB_Proba`
5. `W` is found by `scipy.optimize.minimize_scalar` on the **first 50% of the test set** (validation),
   then evaluated on the **remaining 50%** (true holdout).
"""

CELL_INSTALL = """\
!pip install -q yfinance xgboost scikit-learn tensorflow pyarrow matplotlib numpy pandas scipy
"""

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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize_scalar

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
TICKER     = 'AAPL'
MARKET     = 'SPY'        # S&P 500 ETF as market benchmark
PERIOD     = '10y'        # 10 years of data
LOOKBACK   = 90           # 90-day context window
EPOCHS     = 150
BATCH      = 32
TEST_RATIO = 0.15         # smaller holdout -> more training data
N_SPLITS   = 5

os.makedirs('models', exist_ok=True)
os.makedirs('data',   exist_ok=True)

print(f'TF: {tf.__version__}')
print(f'Ticker: {TICKER} | Market: {MARKET} | Period: {PERIOD} | Lookback: {LOOKBACK}')
"""

CELL_DATA = """\
# ── Data Fetching & Feature Engineering ─────────────────────────────────────
#
# Features (13 total — all scale-invariant):
#   AAPL: log_return, ret_5d, ret_10d, ret_20d,
#         close_vs_ma20, close_vs_ma50, RSI, macd_norm,
#         BB_pct, atr_norm, Vol_ratio
#   Market: spy_return, rel_strength (AAPL - SPY)

def fetch_data(ticker, market, period):
    \"\"\"Download stock + market benchmark, align on common trading days.\"\"\"
    print(f'  Downloading {ticker}...')
    df_s = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    print(f'  Downloading {market}...')
    df_m = yf.download(market, period=period, auto_adjust=True, progress=False)

    for df in [df_s, df_m]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)

    common = df_s.index.intersection(df_m.index)
    return df_s.loc[common].copy(), df_m.loc[common].copy()


def build_features(df_stock, df_market):
    \"\"\"Build 13 scale-invariant features. All forward-looking bias removed.\"\"\"
    out = df_stock.copy()

    # ── Moving Averages ────────────────────────────────────────────────────
    out['MA20'] = out['Close'].rolling(20, min_periods=1).mean()
    out['MA50'] = out['Close'].rolling(50, min_periods=1).mean()

    # ── AAPL Log Return (target + feature) ───────────────────────────────
    out['log_return'] = np.log(out['Close'] / out['Close'].shift(1))

    # ── Multi-Timeframe Momentum ──────────────────────────────────────────
    out['ret_5d']  = out['log_return'].rolling(5,  min_periods=1).sum()
    out['ret_10d'] = out['log_return'].rolling(10, min_periods=1).sum()
    out['ret_20d'] = out['log_return'].rolling(20, min_periods=1).sum()

    # ── Price vs Moving Average (scale-invariant) ─────────────────────────
    out['close_vs_ma20'] = out['Close'] / out['MA20'] - 1
    out['close_vs_ma50'] = out['Close'] / out['MA50'] - 1

    # ── RSI-14 (Wilder smoothing) → scaled to [0, 1] ─────────────────────
    delta    = out['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    out['RSI'] = (100 - (100 / (1 + rs))) / 100.0

    # ── MACD / Close (normalised) ─────────────────────────────────────────
    ema12          = out['Close'].ewm(span=12, adjust=False).mean()
    ema26          = out['Close'].ewm(span=26, adjust=False).mean()
    out['macd_norm'] = (ema12 - ema26) / out['Close']

    # ── Bollinger Band %B ─────────────────────────────────────────────────
    std20      = out['Close'].rolling(20, min_periods=1).std().fillna(0)
    upper_bb   = out['MA20'] + 2 * std20
    lower_bb   = out['MA20'] - 2 * std20
    out['BB_pct'] = (out['Close'] - lower_bb) / (upper_bb - lower_bb + 1e-9)

    # ── ATR / Close (normalised volatility) ───────────────────────────────
    hl  = out['High'] - out['Low']
    hc  = (out['High'] - out['Close'].shift()).abs()
    lc  = (out['Low']  - out['Close'].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    out['atr_norm'] = tr.ewm(com=13, adjust=False).mean() / out['Close']

    # ── Volume Ratio ───────────────────────────────────────────────────────
    out['Vol_ratio'] = out['Volume'] / (
        out['Volume'].rolling(20, min_periods=1).mean() + 1)

    # ── Market Features (SPY) ─────────────────────────────────────────────
    spy_ret = np.log(df_market['Close'] / df_market['Close'].shift(1))
    out['spy_return']   = spy_ret.values
    out['rel_strength'] = out['log_return'] - out['spy_return']

    out.dropna(inplace=True)
    return out


print('[1/7] Fetching data...')
df_stock, df_market = fetch_data(TICKER, MARKET, PERIOD)
df = build_features(df_stock, df_market)

print(f'  {len(df)} trading days  ({df.index[0].date()} -> {df.index[-1].date()})')
print(f'  Mean daily return: {df["log_return"].mean()*100:.3f}%  '
      f'Std: {df["log_return"].std()*100:.3f}%')
print(f'  SPY corr: {df["log_return"].corr(df["spy_return"]):.3f}')
"""

CELL_SPLIT = """\
# ── Train / Test Split ────────────────────────────────────────────────────────
FEATURE_COLS = [
    'log_return',       # today's log return (window feature)
    'ret_5d',           # 5-day momentum
    'ret_10d',          # 10-day momentum
    'ret_20d',          # 20-day momentum
    'close_vs_ma20',    # position vs MA20
    'close_vs_ma50',    # position vs MA50
    'RSI',              # momentum oscillator [0,1]
    'macd_norm',        # MACD / Close
    'BB_pct',           # Bollinger Band position
    'atr_norm',         # normalised volatility
    'Vol_ratio',        # volume anomaly
    'spy_return',       # S&P 500 return (market factor)
    'rel_strength',     # AAPL alpha vs market
]
TARGET_COL = 'log_return'

print(f'  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}')

features  = df[FEATURE_COLS].values
target    = df[TARGET_COL].values.reshape(-1, 1)
prices    = df['Close'].values

split_idx  = int(len(features) * (1 - TEST_RATIO))
split_date = df.index[split_idx]

train_feat = features[:split_idx]
train_targ = target[:split_idx]
test_feat  = features[split_idx - LOOKBACK:]
test_targ  = target[split_idx - LOOKBACK:]

feat_sc = StandardScaler()
targ_sc = StandardScaler()
train_feat_sc = feat_sc.fit_transform(train_feat)
test_feat_sc  = feat_sc.transform(test_feat)
train_targ_sc = targ_sc.fit_transform(train_targ)
test_targ_sc  = targ_sc.transform(test_targ)

print(f'[2/7] Train: {split_idx} rows ({df.index[0].date()} -> {split_date.date()})')
print(f'       Test: {len(features)-split_idx} rows ({split_date.date()} -> {df.index[-1].date()})')
"""

CELL_SEQUENCES = """\
# ── Sequence Builder ──────────────────────────────────────────────────────────
def create_sequences(feat_sc, targ_sc, lookback):
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
# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_return_metrics(actual_ret, pred_ret):
    actual_ret = np.array(actual_ret).flatten()
    pred_ret   = np.array(pred_ret).flatten()
    n = min(len(actual_ret), len(pred_ret))
    actual_ret, pred_ret = actual_ret[:n], pred_ret[:n]

    mae      = float(np.mean(np.abs(actual_ret - pred_ret)))
    rmse     = float(np.sqrt(np.mean((actual_ret - pred_ret)**2)))
    mae_pct  = mae * 100
    rmse_pct = rmse * 100
    mae_naive = float(np.mean(np.abs(actual_ret)))
    mase      = mae / (mae_naive + 1e-12)

    da = float(np.mean(np.sign(pred_ret) == np.sign(actual_ret))) * 100

    signal = np.where(pred_ret > 0, 1.0, -1.0)
    strat_ret = signal * actual_ret
    sharpe = float(np.mean(strat_ret) / (np.std(strat_ret) + 1e-9) * np.sqrt(252))
    cum_strat = float((np.exp(np.cumsum(strat_ret)) - 1)[-1]) * 100

    return {'MASE': mase, 'MAE_%': mae_pct, 'RMSE_%': rmse_pct, 'Dir_Acc_%': da, 'Sharpe': sharpe, 'Strat_Ret_%': cum_strat}

def compute_clf_metrics(actual_ret, pred_proba):
    actual_ret = np.array(actual_ret).flatten()
    pred_proba = np.array(pred_proba).flatten()
    n = min(len(actual_ret), len(pred_proba))
    actual_ret, pred_proba = actual_ret[:n], pred_proba[:n]

    pred_dir = (pred_proba > 0.5).astype(int)
    actual_dir = (actual_ret > 0).astype(int)
    da = float(np.mean(pred_dir == actual_dir)) * 100

    signal = np.where(pred_dir == 1, 1.0, -1.0)
    strat_ret = signal * actual_ret
    sharpe = float(np.mean(strat_ret) / (np.std(strat_ret) + 1e-9) * np.sqrt(252))
    cum_strat = float((np.exp(np.cumsum(strat_ret)) - 1)[-1]) * 100

    return {'Dir_Acc_%': da, 'Sharpe': sharpe, 'Strat_Ret_%': cum_strat}

def reconstruct_prices(start_price, log_returns):
    return start_price * np.exp(np.cumsum(log_returns))
print('Metrics helper loaded.')
"""

CELL_ARCH = """\
# ── v7 LSTM: Plain BiLSTM Regressor (v3 architecture) ────────────────────────
#
# Why revert from Attention?
#   Attention adds complexity that can hurt on noisy financial data.
#   The v3 plain BiLSTM achieved 53.78% directional accuracy using
#   Huber loss on continuous log_return -- a smoother loss surface
#   provides better gradients than binary cross-entropy for direction.
#
# Architecture:
#   Input(lookback, n_features)
#     -> BiLSTM(128, return_sequences=True)  -> Dropout(0.2)
#     -> BiLSTM(64,  return_sequences=False) -> Dropout(0.2)
#     -> Dense(32, relu)
#     -> Dense(1, linear)   [predicts continuous log_return]

def build_lstm_regressor(lookback, n_features, units_1=128, units_2=64):
    inputs = Input(shape=(lookback, n_features), name='price_sequence')

    x = Bidirectional(LSTM(units_1, return_sequences=True,
                           recurrent_dropout=0.05),
                      name='bilstm_1')(inputs)
    x = Dropout(0.2)(x)

    x = Bidirectional(LSTM(units_2, return_sequences=False,
                           recurrent_dropout=0.05),
                      name='bilstm_2')(x)
    x = Dropout(0.2)(x)

    x = Dense(32, activation='relu', name='dense_1')(x)
    output = Dense(1, name='return_pred')(x)

    model = Model(inputs, output)
    model.compile(
        optimizer=Adam(learning_rate=2e-4, clipnorm=1.0),
        loss='huber',   # robust to outliers, smooth gradients
    )
    return model


lstm_model = build_lstm_regressor(LOOKBACK, len(FEATURE_COLS))
lstm_model.summary()
"""

CELL_CV = """\
# ── Walk-Forward Cross-Validation ─────────────────────────────────────────────
def walk_forward_cv(features_all, target_all, lookback, epochs, batch, n_splits):
    tscv         = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    print(f'\\n[3/7] Walk-Forward CV ({n_splits} folds) — Plain BiLSTM Regressor')
    print('  ' + '-' * 65)

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(features_all)):
        if len(tr_idx) < lookback + 30 or len(va_idx) < 1:
            print(f'  Fold {fold_idx+1}: skipped')
            continue

        fs = StandardScaler().fit(features_all[tr_idx])
        ts = StandardScaler().fit(target_all[tr_idx])

        va_start  = max(0, va_idx[0] - lookback)
        X_tr, y_tr = create_sequences(
            fs.transform(features_all[tr_idx]),
            ts.transform(target_all[tr_idx]), lookback)
        X_va, y_va = create_sequences(
            fs.transform(features_all[va_start: va_idx[-1]+1]),
            ts.transform(target_all[va_start: va_idx[-1]+1]), lookback)

        if len(X_tr) == 0 or len(X_va) == 0:
            continue

        m_fold = build_lstm_regressor(lookback, features_all.shape[1])
        m_fold.fit(X_tr, y_tr, epochs=epochs, batch_size=batch,
                   validation_data=(X_va, y_va),
                   callbacks=[
                       EarlyStopping('val_loss', patience=15,
                                     restore_best_weights=True),
                       ReduceLROnPlateau('val_loss', factor=0.5,
                                         patience=7, verbose=0),
                   ], verbose=0)

        pred_sc    = m_fold.predict(X_va, verbose=0)
        pred_ret   = ts.inverse_transform(pred_sc).flatten()
        actual_ret = ts.inverse_transform(y_va.reshape(-1,1)).flatten()

        fm = compute_return_metrics(actual_ret, pred_ret)
        fm['fold'] = fold_idx + 1
        fold_metrics.append(fm)

        print(f'  Fold {fold_idx+1}/{n_splits}  MASE={fm["MASE"]:.4f}  '
              f'MAE={fm["MAE_%"]:.4f}%  DA={fm["Dir_Acc_%"]:.1f}%  '
              f'Sharpe={fm["Sharpe"]:.3f}')

    return fold_metrics


cv_metrics = walk_forward_cv(features, target, LOOKBACK, EPOCHS, BATCH, N_SPLITS)
if cv_metrics:
    mean_mase = np.mean([m['MASE'] for m in cv_metrics])
    mean_da   = np.mean([m['Dir_Acc_%'] for m in cv_metrics])
    mean_sh   = np.mean([m['Sharpe'] for m in cv_metrics])
    print(f'\\n  CV  MASE={mean_mase:.4f}  Dir_Acc={mean_da:.1f}%  Sharpe={mean_sh:.3f}')
"""

CELL_TRAIN = """\
# ── Train Final BiLSTM Regressor on Full Train Set ───────────────────────────
print('[4/7] Training final BiLSTM Regressor...')

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
print(f'  Saved -> {model_path}')
"""

CELL_XGB = """\
# ── XGBoost Classifier (from v6) ──────────────────────────────────────────────
print('[5/7] Training XGBoost classifier...')

XGB_STOCK_LAGS = 30
XGB_MARKET_LAGS = 10

def build_xgb_df(df, n_stock_lags=30, n_mkt_lags=10):
    out = pd.DataFrame(index=df.index)
    for lag in range(1, n_stock_lags + 1):
        out[f'ret_lag_{lag}'] = df['log_return'].shift(lag)
    for lag in range(1, n_mkt_lags + 1):
        out[f'spy_lag_{lag}'] = df['spy_return'].shift(lag)
    for col in ['ret_5d', 'ret_10d', 'ret_20d', 'RSI', 'macd_norm',
                'close_vs_ma20', 'close_vs_ma50', 'BB_pct',
                'atr_norm', 'Vol_ratio', 'rel_strength']:
        out[col] = df[col].shift(1)
    
    out['target'] = (df['log_return'] > 0).astype(int)
    out.dropna(inplace=True)
    return out

xgb_df    = build_xgb_df(df, XGB_STOCK_LAGS, XGB_MARKET_LAGS)
feat_cols = [c for c in xgb_df.columns if c != 'target']

xgb_train = xgb_df[xgb_df.index <  split_date]
xgb_test  = xgb_df[xgb_df.index >= split_date]

print(f'  XGB features: {len(feat_cols)}  Train: {len(xgb_train)}  Test: {len(xgb_test)}')

xgb_sc = StandardScaler()
X_xtr  = xgb_sc.fit_transform(xgb_train[feat_cols].values)
y_xtr  = xgb_train['target'].values
X_xte  = xgb_sc.transform(xgb_test[feat_cols].values)
y_xte  = xgb_test['target'].values

val_size = max(int(len(X_xtr) * 0.15), 60)
X_fit, y_fit = X_xtr[:-val_size], y_xtr[:-val_size]
X_val, y_val = X_xtr[-val_size:], y_xtr[-val_size:]

n_neg = int((y_fit == 0).sum())
n_pos = int((y_fit == 1).sum())
scale_pw = n_neg / (n_pos + 1e-9)

xgb_model = XGBClassifier(
    n_estimators         = 3000,
    learning_rate        = 0.02,
    max_depth            = 4,
    subsample            = 0.8,
    colsample_bytree     = 0.6,
    min_child_weight     = 5,
    reg_lambda           = 2.0,
    reg_alpha            = 0.5,
    scale_pos_weight     = scale_pw,
    eval_metric          = 'logloss',
    early_stopping_rounds= 50,
    random_state         = 42,
    n_jobs               = -1,
)

xgb_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=200)

xgb_proba = xgb_model.predict_proba(X_xte)[:, 1]
print(f'  XGB proba range: [{xgb_proba.min():.3f}, {xgb_proba.max():.3f}]')
"""

CELL_EVAL = """\
# ── Evaluate: LSTM (Regressor) & XGBoost (Classifier) -- v7 Hybrid ───────────
print('[6/7] Evaluating on holdout test set...')

lstm_pred_sc  = lstm_model.predict(X_test, verbose=0)
lstm_ret_pred = targ_sc.inverse_transform(lstm_pred_sc).flatten()
actual_ret    = targ_sc.inverse_transform(y_test.reshape(-1,1)).flatten()

lstm_dates   = df.index[split_idx : split_idx + len(actual_ret)]
xgb_dates    = xgb_test.index
common_dates = lstm_dates.intersection(xgb_dates)

actual_r = pd.Series(actual_ret,    index=lstm_dates).loc[common_dates].values
lstm_r   = pd.Series(lstm_ret_pred, index=lstm_dates).loc[common_dates].values
xgb_prob = pd.Series(xgb_proba,     index=xgb_dates).loc[common_dates].values

# ── LSTM pseudo-probability via z-score + sigmoid ────────────────────────────
# Use full-test std for consistent scaling
lstm_std  = np.std(lstm_r) + 1e-9
lstm_z    = lstm_r / lstm_std
lstm_prob = 1.0 / (1.0 + np.exp(-lstm_z))

# ── scipy weight optimisation on first 50% of test (validation half) ──────────
n_total   = len(actual_r)
n_val_ens = max(n_total // 2, 1)

def neg_sharpe(w):
    ep  = w * lstm_prob[:n_val_ens] + (1.0 - w) * xgb_prob[:n_val_ens]
    sig = np.where(ep > 0.5, 1.0, -1.0)
    sr  = sig * actual_r[:n_val_ens]
    return -(np.mean(sr) / (np.std(sr) + 1e-9) * np.sqrt(252))

opt    = minimize_scalar(neg_sharpe, bounds=(0.0, 1.0), method='bounded')
LSTM_W = float(opt.x)
print(f'  Optimised LSTM weight W = {LSTM_W:.3f}  (val Sharpe = {-opt.fun:.3f})')

ens_prob = LSTM_W * lstm_prob + (1.0 - LSTM_W) * xgb_prob

# ── Metrics (full test set) ───────────────────────────────────────────────────
lstm_met = compute_return_metrics(actual_r, lstm_r)
xgb_met  = compute_clf_metrics(actual_r, xgb_prob)
ens_met  = compute_clf_metrics(actual_r, ens_prob)

print()
print('  ── Model Comparison (Directional Metrics) ────────────────────────')
print(f"  {'Metric':14s}  {'LSTM':>8s}  {'XGBoost':>8s}  {'Ensemble':>8s}")
print('  ' + '-' * 50)
for k in ['Dir_Acc_%', 'Sharpe', 'Strat_Ret_%']:
    lv = lstm_met.get(k, 0.0)
    xv = xgb_met.get(k, 0.0)
    ev = ens_met.get(k, 0.0)
    print(f'  {k:14s}  {lv:8.3f}  {xv:8.3f}  {ev:8.3f}')

def get_strat(prob, ret):
    return np.exp(np.cumsum(np.where(prob > 0.5, 1.0, -1.0) * ret))

cum_bh   = np.exp(np.cumsum(actual_r))
cum_lstm  = get_strat(lstm_prob, actual_r)
cum_xgb   = get_strat(xgb_prob,  actual_r)
cum_ens   = get_strat(ens_prob,  actual_r)
"""

CELL_SAVE = """\
# ── Save Outputs & Plot ───────────────────────────────────────────────────────
print('[7/7] Saving outputs...')

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')
for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
ax.tick_params(colors='#c9d1d9')

n = min(len(common_dates), len(cum_bh))
ax.plot(common_dates[:n], cum_bh[:n],   color='#8b949e', label='Buy & Hold', lw=2)
ax.plot(common_dates[:n], cum_lstm[:n], color='#58a6ff', label=f'LSTM (DA={lstm_met["Dir_Acc_%"]:.1f}%)', lw=1.5)
ax.plot(common_dates[:n], cum_xgb[:n],  color='#f0883e', label=f'XGB (DA={xgb_met["Dir_Acc_%"]:.1f}%)', lw=1.5, ls='--')
ax.plot(common_dates[:n], cum_ens[:n],  color='#3fb950', label=f'Ensemble W={LSTM_W:.2f} (DA={ens_met["Dir_Acc_%"]:.1f}%)', lw=2, ls='-.')

# Mark validation / holdout boundary on plot
if n_val_ens < n:
    ax.axvline(common_dates[n_val_ens], color='#ff7b72', ls=':', lw=1, label='Val|Holdout split')

ax.set_title(f'v7 Hybrid: LSTM Regressor + XGBoost Classifier -- {TICKER}', color='#e6edf3', fontsize=13)
ax.set_xlabel('Date', color='#c9d1d9')
ax.set_ylabel('Cumulative Return (x)', color='#c9d1d9')
ax.legend(facecolor='#21262d', labelcolor='#c9d1d9')
plt.tight_layout()
plt.savefig(f'models/{TICKER}_evaluation.png', dpi=150)
plt.show()
print(f'  Plot saved -> models/{TICKER}_evaluation.png')

# ── Persist metrics JSON ──────────────────────────────────────────────────────
results = {
    'ticker':   TICKER,
    'version':  'v7_hybrid',
    'lstm_w':   LSTM_W,
    'lstm':     lstm_met,
    'xgboost':  xgb_met,
    'ensemble': ens_met,
}
with open(f'models/{TICKER}_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'  Metrics saved -> models/{TICKER}_metrics.json')
print('\\nDone!')
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
