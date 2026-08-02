"""
generate_notebook_v4.py  —  Industry-grade stock return prediction

Key upgrades over v3:
  1. 10y data (2× more training samples)
  2. LOOKBACK = 90 days (captures monthly/quarterly patterns)
  3. Market context: SPY log-return + relative strength (AAPL vs S&P500)
     → ~60-70% of AAPL's daily move IS the market; giving the model this
       directly is the single biggest accuracy improvement possible.
  4. Multi-timeframe momentum: 5d, 10d, 20d cumulative returns
  5. BiLSTM + Additive Attention (focus on most relevant past days)
  6. XGBoost: early stopping on train-val split (optimal tree count)
  7. Ensemble: weighted average of LSTM + XGBoost
  8. Better XGBoost features: lag SPY returns + multi-timeframe momentum
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
# 📈 Stock Prediction v4 — LSTM+Attention & XGBoost (Industry-Grade)

**Self-contained. Run All — no file uploads needed.**

## v4 Upgrades Over v3

| Upgrade | v3 | v4 |
|---|---|---|
| **Training data** | 5 years | **10 years** (2× more) |
| **Lookback window** | 60 days | **90 days** (quarterly patterns) |
| **Market context** | None | **SPY log-return + relative strength** |
| **Momentum features** | None | **5d, 10d, 20d** cumulative returns |
| **Architecture** | Plain BiLSTM | **BiLSTM + Additive Attention** |
| **XGBoost** | Fixed 1000 trees, no early stop | **Early stopping on val split** |
| **Final prediction** | LSTM only | **Ensemble (LSTM + XGBoost)** |

### Why SPY (S&P 500) Is The Biggest Win
~60-70% of AAPL's daily return is just the market moving.  
Giving the model past SPY returns lets it learn:
- "Market momentum has been positive for 5 days → likely continues"
- "AAPL is outperforming the market → relative strength divergence signal"
"""

CELL_INSTALL = """\
!pip install -q yfinance xgboost scikit-learn tensorflow pyarrow matplotlib numpy pandas
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

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    BatchNormalization, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
TICKER     = 'AAPL'
MARKET     = 'SPY'        # S&P 500 ETF as market benchmark
PERIOD     = '10y'        # 10 years of data (v3 used 5y)
LOOKBACK   = 90           # 90-day context window (v3 used 60)
EPOCHS     = 150
BATCH      = 32
TEST_RATIO = 0.15         # smaller holdout → more training data
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

    mae_naive = float(np.mean(np.abs(actual_ret)))   # naive = predict 0
    mase      = mae / (mae_naive + 1e-12)

    # Directional accuracy (% of days correct sign)
    da = float(np.mean(np.sign(pred_ret) == np.sign(actual_ret))) * 100

    # Profit factor (sum of correct-direction returns / sum of wrong)
    correct   = np.where(np.sign(pred_ret) == np.sign(actual_ret),
                         np.abs(actual_ret), 0)
    incorrect = np.where(np.sign(pred_ret) != np.sign(actual_ret),
                         np.abs(actual_ret), 0)
    pf = correct.sum() / (incorrect.sum() + 1e-9)

    return {
        'MASE':       mase,
        'MAE_%':      mae_pct,
        'RMSE_%':     rmse_pct,
        'Dir_Acc_%':  da,
        'Profit_Factor': pf,
    }


def reconstruct_prices(start_price, log_returns):
    return start_price * np.exp(np.cumsum(log_returns))


print('Metrics helper loaded.')
"""

CELL_ARCH = """\
# ── BiLSTM + Additive Attention Architecture ──────────────────────────────────
#
# Additive (Bahdanau) Attention:
#   score[t]  = tanh(Dense(hidden_state[t]))      shape: (batch, time, 1)
#   weight[t] = softmax(score)[t]                 focus on relevant days
#   context   = sum(weight[t] * hidden_state[t])  weighted history
#
# This lets the model learn WHICH of the past 90 days most influenced
# the next return — e.g. "the earnings day 45 steps ago matters now".

def build_lstm_attention(lookback, n_features, units_1=128, units_2=64):
    inputs = Input(shape=(lookback, n_features), name='price_sequence')

    # ── Layer 1: Bidirectional LSTM ───────────────────────────────────────
    x = Bidirectional(LSTM(units_1, return_sequences=True,
                           recurrent_dropout=0.05),
                      name='bilstm_1')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # ── Layer 2: LSTM (return all hidden states for attention) ────────────
    x = Bidirectional(LSTM(units_2, return_sequences=True,
                           recurrent_dropout=0.05),
                      name='bilstm_2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)

    # ── Additive Attention ────────────────────────────────────────────────
    # score shape: (batch, lookback, 1)
    score   = Dense(1, activation='tanh', name='attn_score')(x)
    # softmax over time axis
    weights = tf.nn.softmax(score, axis=1)           # (batch, lookback, 1)
    # context: weighted sum of hidden states
    context = tf.reduce_sum(x * weights, axis=1)     # (batch, 2*units_2=128)

    # ── Prediction Head ────────────────────────────────────────────────────
    x = Dense(128, activation='gelu', name='dense_1')(context)
    x = Dropout(0.15)(x)
    x = Dense(64,  activation='gelu', name='dense_2')(x)
    x = Dropout(0.10)(x)
    x = Dense(32,  activation='relu', name='dense_3')(x)
    output = Dense(1, name='return_pred')(x)

    model = Model(inputs, output)
    model.compile(
        optimizer=Adam(learning_rate=2e-4, clipnorm=1.0),
        loss='huber',
    )
    return model


lstm_model = build_lstm_attention(LOOKBACK, len(FEATURE_COLS))
lstm_model.summary()
"""

CELL_CV = """\
# ── Walk-Forward Cross-Validation ─────────────────────────────────────────────
def walk_forward_cv(features_all, target_all, lookback, epochs, batch, n_splits):
    tscv         = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    print(f'\\n[3/7] Walk-Forward CV ({n_splits} folds) — BiLSTM+Attention')
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

        m_fold = build_lstm_attention(lookback, features_all.shape[1])
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
              f'PF={fm["Profit_Factor"]:.3f}')

    return fold_metrics


cv_metrics = walk_forward_cv(features, target, LOOKBACK, EPOCHS, BATCH, N_SPLITS)
if cv_metrics:
    mean_mase = np.mean([m['MASE'] for m in cv_metrics])
    mean_da   = np.mean([m['Dir_Acc_%'] for m in cv_metrics])
    mean_pf   = np.mean([m['Profit_Factor'] for m in cv_metrics])
    print(f'\\n  CV  MASE={mean_mase:.4f}  Dir_Acc={mean_da:.1f}%  PF={mean_pf:.3f}')
"""

CELL_TRAIN = """\
# ── Train Final LSTM+Attention on Full Train Set ──────────────────────────────
print('[4/7] Training final BiLSTM+Attention...')

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
# ── XGBoost with Early Stopping ───────────────────────────────────────────────
#
# Key improvements over v3:
#   - Lag-1..30 AAPL returns + lag-1..10 SPY returns (market context)
#   - Multi-timeframe momentum: ret_5d, ret_10d, ret_20d
#   - Early stopping on last 15% of train (no more fixed 1000 trees)
#   - Shallower trees (depth=3) for noisy return data

print('[5/7] Training XGBoost with early stopping...')

XGB_STOCK_LAGS = 30
XGB_MARKET_LAGS = 10

def build_xgb_df(df, n_stock_lags=30, n_mkt_lags=10):
    out = pd.DataFrame(index=df.index)

    # Lag features of AAPL log returns
    for lag in range(1, n_stock_lags + 1):
        out[f'ret_lag_{lag}'] = df['log_return'].shift(lag)

    # Lag features of SPY log returns
    for lag in range(1, n_mkt_lags + 1):
        out[f'spy_lag_{lag}'] = df['spy_return'].shift(lag)

    # Yesterday's technical indicators (shift 1 to avoid lookahead)
    for col in ['ret_5d', 'ret_10d', 'ret_20d', 'RSI', 'macd_norm',
                'close_vs_ma20', 'close_vs_ma50', 'BB_pct',
                'atr_norm', 'Vol_ratio', 'rel_strength']:
        out[col] = df[col].shift(1)

    out['target'] = df['log_return']
    out.dropna(inplace=True)
    return out


xgb_df    = build_xgb_df(df, XGB_STOCK_LAGS, XGB_MARKET_LAGS)
feat_cols = [c for c in xgb_df.columns if c != 'target']

xgb_train = xgb_df[xgb_df.index <  split_date]
xgb_test  = xgb_df[xgb_df.index >= split_date]

print(f'  XGB features: {len(feat_cols)}  '
      f'Train: {len(xgb_train)}  Test: {len(xgb_test)}')

xgb_sc = StandardScaler()
X_xtr  = xgb_sc.fit_transform(xgb_train[feat_cols].values)
y_xtr  = xgb_train['target'].values
X_xte  = xgb_sc.transform(xgb_test[feat_cols].values)
y_xte  = xgb_test['target'].values

# Carve out last 15% of TRAIN as validation for early stopping
val_size = max(int(len(X_xtr) * 0.15), 60)
X_xtr_fit, X_xval = X_xtr[:-val_size], X_xtr[-val_size:]
y_xtr_fit, y_xval = y_xtr[:-val_size], y_xtr[-val_size:]

xgb_model = XGBRegressor(
    n_estimators     = 3000,       # upper bound; early stopping finds optimal
    learning_rate    = 0.01,
    max_depth        = 3,          # shallow = less overfit on noisy returns
    subsample        = 0.7,
    colsample_bytree = 0.5,
    min_child_weight = 10,
    reg_lambda       = 3.0,
    reg_alpha        = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    early_stopping_rounds = 50,    # stops when val doesn't improve for 50 rounds
)

xgb_model.fit(
    X_xtr_fit, y_xtr_fit,
    eval_set  = [(X_xval, y_xval)],
    verbose   = 200,
)

xgb_ret_pred = xgb_model.predict(X_xte)
print(f'  Best iteration: {xgb_model.best_iteration}')
print(f'  XGBoost trained | test days: {len(xgb_ret_pred)}')
"""

CELL_EVAL = """\
# ── Evaluate: LSTM, XGBoost, Ensemble ────────────────────────────────────────
print('[6/7] Evaluating on holdout test set...')

# LSTM predictions
lstm_pred_sc  = lstm_model.predict(X_test, verbose=0)
lstm_ret_pred = targ_sc.inverse_transform(lstm_pred_sc).flatten()
actual_ret    = targ_sc.inverse_transform(y_test.reshape(-1,1)).flatten()

# Align to common dates
lstm_dates   = df.index[split_idx : split_idx + len(actual_ret)]
xgb_dates    = xgb_test.index
common_dates = lstm_dates.intersection(xgb_dates)

actual_s = pd.Series(actual_ret,   index=lstm_dates)
lstm_s   = pd.Series(lstm_ret_pred, index=lstm_dates)
xgb_s    = pd.Series(xgb_ret_pred,  index=xgb_dates)

actual_r = actual_s.loc[common_dates].values
lstm_r   = lstm_s.loc[common_dates].values
xgb_r    = xgb_s.loc[common_dates].values
naive_r  = np.zeros(len(actual_r))

# ── Ensemble: weighted average ────────────────────────────────────────────────
# We use a simple 50/50 mix; could be optimised on validation set.
LSTM_W, XGB_W = 0.5, 0.5
ens_r = LSTM_W * lstm_r + XGB_W * xgb_r

# ── Compute metrics ───────────────────────────────────────────────────────────
lstm_met  = compute_return_metrics(actual_r, lstm_r)
xgb_met   = compute_return_metrics(actual_r, xgb_r)
ens_met   = compute_return_metrics(actual_r, ens_r)
naive_met = compute_return_metrics(actual_r, naive_r)

print()
print('  ── Model Comparison (Return Space) ─────────────────────────────────')
print(f"  {'Metric':16s}  {'LSTM':>8s}  {'XGBoost':>8s}  {'Ensemble':>8s}  {'Naive':>8s}")
print('  ' + '-' * 60)
for k in ['MASE', 'MAE_%', 'RMSE_%', 'Dir_Acc_%', 'Profit_Factor']:
    lv = lstm_met.get(k, 0.0)
    xv = xgb_met.get(k, 0.0)
    ev = ens_met.get(k, 0.0)
    nv = naive_met.get(k, 0.0)
    print(f'  {k:16s}  {lv:8.4f}  {xv:8.4f}  {ev:8.4f}  {nv:8.4f}')

print()
print('  MASE < 1.0 = beats naive  |  Dir_Acc > 50% = better than random')
print('  Profit_Factor > 1.0 = correct-direction returns > wrong-direction losses')
print()
for name, m in [('LSTM', lstm_met), ('XGBoost', xgb_met), ('Ensemble', ens_met)]:
    mase = m['MASE']
    da   = m['Dir_Acc_%']
    pf   = m['Profit_Factor']
    beat = mase < 1.0
    print(f'  {"OK" if beat else "!!"} {name}: '
          f'MASE={mase:.4f}  DA={da:.1f}%  PF={pf:.3f}  '
          f'-> {"BEATS" if beat else "loses to"} naive')

# ── Reconstruct prices from predicted returns ─────────────────────────────────
start_price = df['Close'].values[split_idx - 1]
actual_prices   = df['Close'].values[split_idx : split_idx + len(actual_r)]
lstm_prices     = reconstruct_prices(start_price, lstm_r)
xgb_prices      = reconstruct_prices(start_price, xgb_r)
ens_prices      = reconstruct_prices(start_price, ens_r)
"""

CELL_SAVE = """\
# ── Save Outputs & Plot ───────────────────────────────────────────────────────
print('[7/7] Saving outputs...')

comparison = {
    'ticker': TICKER, 'period': PERIOD, 'lookback': LOOKBACK,
    'lstm':     {k: float(v) for k, v in lstm_met.items()},
    'xgb':      {k: float(v) for k, v in xgb_met.items()},
    'ensemble': {k: float(v) for k, v in ens_met.items()},
    'naive':    {k: float(v) for k, v in naive_met.items()},
}
cmp_path = f'models/{TICKER}_comparison.json'
with open(cmp_path, 'w') as fh:
    json.dump(comparison, fh, indent=2)

n_save = min(len(common_dates), len(lstm_r), len(xgb_r))
pd.DataFrame({
    'date':          [str(d.date()) for d in common_dates[:n_save]],
    'actual_return': actual_r[:n_save].tolist(),
    'lstm_return':   lstm_r[:n_save].tolist(),
    'xgb_return':    xgb_r[:n_save].tolist(),
    'ens_return':    ens_r[:n_save].tolist(),
    'actual_price':  actual_prices[:n_save].tolist(),
    'lstm_price':    lstm_prices[:n_save].tolist(),
    'xgb_price':     xgb_prices[:n_save].tolist(),
    'ens_price':     ens_prices[:n_save].tolist(),
}).to_json(f'models/{TICKER}_predictions.json', orient='records', indent=2)

# ── 4-Panel Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(20, 10))
fig.patch.set_facecolor('#0d1117')
axes = axes.flatten()
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
axes[0].set_title('BiLSTM+Attention Training Loss (Huber)')
axes[0].set_xlabel('Epoch')
axes[0].legend(facecolor='#21262d', labelcolor='#c9d1d9')

# 2. Actual vs predicted returns (first 120 test days)
pn = min(120, n_save)
axes[1].plot(common_dates[:pn], actual_r[:pn]*100,
             color='#8b949e', label='Actual return %', lw=1.2, alpha=0.85)
axes[1].plot(common_dates[:pn], lstm_r[:pn]*100,
             color='#58a6ff', label='LSTM %', lw=1.0, alpha=0.8)
axes[1].plot(common_dates[:pn], xgb_r[:pn]*100,
             color='#f0883e', label='XGBoost %', lw=1.0, alpha=0.8, ls='--')
axes[1].axhline(0, color='#444', lw=0.8, ls=':')
axes[1].set_title('Daily Log Returns — Actual vs Predicted')
axes[1].set_ylabel('Return (%)')
axes[1].legend(facecolor='#21262d', labelcolor='#c9d1d9', fontsize=8)

# 3. Cumulative price reconstruction (full test period)
pn2 = n_save
lstm_lbl = 'LSTM  MASE={:.3f} DA={:.1f}%'.format(lstm_met['MASE'], lstm_met['Dir_Acc_%'])
xgb_lbl  = 'XGB   MASE={:.3f} DA={:.1f}%'.format(xgb_met['MASE'],  xgb_met['Dir_Acc_%'])
ens_lbl  = 'Ens   MASE={:.3f} DA={:.1f}%'.format(ens_met['MASE'],   ens_met['Dir_Acc_%'])
axes[2].plot(common_dates[:pn2], actual_prices[:pn2], color='#8b949e', label='Actual', lw=2)
axes[2].plot(common_dates[:pn2], lstm_prices[:pn2],   color='#58a6ff', label=lstm_lbl, lw=1.5)
axes[2].plot(common_dates[:pn2], xgb_prices[:pn2],    color='#f0883e', label=xgb_lbl,  lw=1.5, ls='--')
axes[2].plot(common_dates[:pn2], ens_prices[:pn2],    color='#3fb950', label=ens_lbl,  lw=2.0, ls='-.')
axes[2].set_title(f'{TICKER} — Cumulative Price from Predicted Returns')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Price (USD)')
axes[2].legend(facecolor='#21262d', labelcolor='#c9d1d9', fontsize=8)

# 4. Directional accuracy bar chart
models  = ['Naive', 'LSTM', 'XGBoost', 'Ensemble']
da_vals = [50.0, lstm_met['Dir_Acc_%'], xgb_met['Dir_Acc_%'], ens_met['Dir_Acc_%']]
colors  = ['#444', '#58a6ff', '#f0883e', '#3fb950']
bars    = axes[3].bar(models, da_vals, color=colors, alpha=0.85, width=0.5)
axes[3].axhline(50, color='#f78166', lw=1.5, ls='--', label='Random (50%)')
axes[3].set_title('Directional Accuracy (% days correct sign)')
axes[3].set_ylabel('Accuracy (%)')
axes[3].set_ylim(45, 65)
axes[3].legend(facecolor='#21262d', labelcolor='#c9d1d9', fontsize=9)
for bar, val in zip(bars, da_vals):
    axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}%', ha='center', va='bottom',
                 color='#c9d1d9', fontsize=9)

fig.suptitle(f'{TICKER} v4 Evaluation  ({PERIOD} data, lookback={LOOKBACK})',
             color='#e6edf3', fontsize=13, y=1.01)
fig.tight_layout()
plot_path = f'models/{TICKER}_evaluation.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()

print(f'  Done!')
print(f'  Model  -> models/{TICKER}_lstm.keras')
print(f'  Plot   -> {plot_path}')
print()
print('  ── Final Metrics (Return Space) ─────────────────────────────────────')
print(f"  {'Metric':16s}  {'LSTM':>8s}  {'XGBoost':>8s}  {'Ensemble':>8s}  {'Naive':>8s}")
print('  ' + '-' * 60)
for k in ['MASE', 'MAE_%', 'RMSE_%', 'Dir_Acc_%', 'Profit_Factor']:
    lv = lstm_met.get(k, 0.0)
    xv = xgb_met.get(k, 0.0)
    ev = ens_met.get(k, 0.0)
    nv = naive_met.get(k, 0.0)
    print(f'  {k:16s}  {lv:8.4f}  {xv:8.4f}  {ev:8.4f}  {nv:8.4f}')
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
