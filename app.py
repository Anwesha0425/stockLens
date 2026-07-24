"""
app.py — Enhanced Stock Price Prediction Dashboard
====================================================
Features:
  • Live data via yfinance (any ticker)
  • Bidirectional LSTM predictions loaded from disk
  • 30-day future forecast
  • Candlestick chart with volume
  • Moving averages overlay (MA20, MA50, MA200)
  • Multi-stock comparison (High/Low + Volume)
  • Model performance metrics panel
  • Dark glassmorphism UI

Run:
    python app.py
Then open: http://127.0.0.1:8050
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from utils import (
    fetch_stock_data,
    add_moving_averages,
    create_sequences,
    compute_metrics,
    forecast_future,
)

# ── Try loading TF / Keras ────────────────────────────────────────────────────
try:
    from tensorflow.keras.models import load_model as keras_load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── App Init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="📈 Stock Price Prediction",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server  # Expose Flask server for deployment

# ── Constants ─────────────────────────────────────────────────────────────────
POPULAR_TICKERS = [
    {"label": "🍎 Apple (AAPL)",       "value": "AAPL"},
    {"label": "🪟 Microsoft (MSFT)",    "value": "MSFT"},
    {"label": "🔍 Alphabet (GOOGL)",   "value": "GOOGL"},
    {"label": "📦 Amazon (AMZN)",       "value": "AMZN"},
    {"label": "⚡ Tesla (TSLA)",        "value": "TSLA"},
    {"label": "🤖 NVIDIA (NVDA)",       "value": "NVDA"},
    {"label": "📘 Meta (META)",          "value": "META"},
    {"label": "🏦 JPMorgan (JPM)",      "value": "JPM"},
    {"label": "🛢 ExxonMobil (XOM)",    "value": "XOM"},
    {"label": "🏪 Walmart (WMT)",       "value": "WMT"},
]

COMPARISON_TICKERS = [
    {"label": "AAPL", "value": "AAPL"},
    {"label": "MSFT", "value": "MSFT"},
    {"label": "GOOGL","value": "GOOGL"},
    {"label": "AMZN", "value": "AMZN"},
    {"label": "TSLA", "value": "TSLA"},
    {"label": "NVDA", "value": "NVDA"},
    {"label": "META", "value": "META"},
    {"label": "JPM",  "value": "JPM"},
]

PERIODS = [
    {"label": "1 Month",  "value": "1mo"},
    {"label": "3 Months", "value": "3mo"},
    {"label": "6 Months", "value": "6mo"},
    {"label": "1 Year",   "value": "1y"},
    {"label": "2 Years",  "value": "2y"},
    {"label": "5 Years",  "value": "5y"},
]

# ── Plotly layout defaults ────────────────────────────────────────────────────
LAYOUT_DEFAULTS = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="Inter, sans-serif", color="#c9d1d9", size=12),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", showgrid=True),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", showgrid=True),
    legend=dict(
        bgcolor="rgba(22,27,34,0.9)",
        bordercolor="#30363d",
        borderwidth=1,
        font=dict(color="#c9d1d9"),
    ),
    margin=dict(l=50, r=30, t=60, b=50),
    hovermode="x unified",
)

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # ── Header ──────────────────────────────────────────────────────────────
    html.Div([
        html.Span("📈", className="header-icon"),
        html.H1("Stock Price Prediction Dashboard"),
        html.Span(
            "Powered by Bidirectional LSTM · yfinance · Plotly Dash",
            style={"fontSize": "0.75rem", "color": "#8b949e",
                   "marginLeft": "auto", "fontStyle": "italic"},
        ),
    ], className="header-band"),

    # ── Metric Cards (filled by callback) ─────────────────────────────────
    html.Div(id="metric-strip", className="metric-strip"),

    # ── Controls ────────────────────────────────────────────────────────────
    html.Div([
        html.Span("Ticker", className="control-label"),
        dcc.Dropdown(
            id="ticker-dropdown",
            options=POPULAR_TICKERS,
            value="AAPL",
            clearable=False,
            style={"width": "220px", "background": "#161b22",
                   "border": "1px solid #30363d", "color": "#e6edf3"},
        ),
        html.Span("Period", className="control-label"),
        dcc.Dropdown(
            id="period-dropdown",
            options=PERIODS,
            value="2y",
            clearable=False,
            style={"width": "140px", "background": "#161b22",
                   "border": "1px solid #30363d"},
        ),
        html.Span("MA Overlays", className="control-label"),
        dcc.Checklist(
            id="ma-checklist",
            options=[
                {"label": " MA20",  "value": "MA20"},
                {"label": " MA50",  "value": "MA50"},
                {"label": " MA200", "value": "MA200"},
            ],
            value=["MA20", "MA50"],
            inline=True,
            style={"color": "#c9d1d9", "fontSize": "0.85rem", "gap": "1rem"},
            inputStyle={"marginRight": "4px"},
        ),
        html.Span("Show Forecast", className="control-label"),
        dcc.Checklist(
            id="forecast-checklist",
            options=[{"label": " 30-day", "value": "show"}],
            value=["show"],
            inline=True,
            style={"color": "#58a6ff", "fontSize": "0.85rem"},
            inputStyle={"marginRight": "4px"},
        ),
    ], className="controls-bar"),

    # ── Main Tabs ────────────────────────────────────────────────────────────
    html.Div([
        dcc.Tabs(
            id="main-tabs",
            value="tab-lstm",
            className="custom-tabs",
            children=[

                # ── Tab 1: LSTM Prediction ─────────────────────────────────
                dcc.Tab(
                    label="🤖 LSTM Prediction",
                    value="tab-lstm",
                    className="tab",
                    selected_className="tab--selected",
                    children=[
                        html.Div([
                            html.Div(
                                "Train first with: python train_model.py --ticker AAPL  "
                                "(predictions load automatically when model file exists)",
                                className="info-box",
                            ),
                            html.Div("Price + LSTM Prediction + Forecast",
                                     className="section-heading"),
                            dcc.Loading(
                                dcc.Graph(id="lstm-graph", style={"height": "480px"}),
                                type="circle", color="#58a6ff",
                            ),
                        ], className="tab-content"),
                    ],
                ),

                # ── Tab 2: Candlestick & Volume ────────────────────────────
                dcc.Tab(
                    label="🕯 Candlestick",
                    value="tab-candle",
                    className="tab",
                    selected_className="tab--selected",
                    children=[
                        html.Div([
                            html.Div("OHLC Candlestick + Volume",
                                     className="section-heading"),
                            dcc.Loading(
                                dcc.Graph(id="candle-graph", style={"height": "560px"}),
                                type="circle", color="#58a6ff",
                            ),
                        ], className="tab-content"),
                    ],
                ),

                # ── Tab 3: Moving Averages ─────────────────────────────────
                dcc.Tab(
                    label="📊 Moving Averages",
                    value="tab-ma",
                    className="tab",
                    selected_className="tab--selected",
                    children=[
                        html.Div([
                            html.Div("Close Price & Moving Averages",
                                     className="section-heading"),
                            dcc.Loading(
                                dcc.Graph(id="ma-graph", style={"height": "480px"}),
                                type="circle", color="#58a6ff",
                            ),
                        ], className="tab-content"),
                    ],
                ),

                # ── Tab 4: Multi-Stock Comparison ──────────────────────────
                dcc.Tab(
                    label="⚖ Compare Stocks",
                    value="tab-compare",
                    className="tab",
                    selected_className="tab--selected",
                    children=[
                        html.Div([
                            html.Div("Select stocks to compare", className="section-heading"),
                            dcc.Dropdown(
                                id="compare-dropdown",
                                options=COMPARISON_TICKERS,
                                value=["AAPL", "MSFT", "NVDA"],
                                multi=True,
                                style={"background": "#161b22",
                                       "border": "1px solid #30363d",
                                       "marginBottom": "1rem"},
                            ),
                            dcc.Loading(
                                dcc.Graph(id="compare-price-graph",
                                          style={"height": "360px"}),
                                type="circle", color="#58a6ff",
                            ),
                            html.Div("Market Volume Comparison",
                                     className="section-heading",
                                     style={"marginTop": "1.5rem"}),
                            dcc.Loading(
                                dcc.Graph(id="compare-volume-graph",
                                          style={"height": "300px"}),
                                type="circle", color="#58a6ff",
                            ),
                        ], className="tab-content"),
                    ],
                ),

                # ── Tab 5: Returns Distribution ────────────────────────────
                dcc.Tab(
                    label="📉 Returns",
                    value="tab-returns",
                    className="tab",
                    selected_className="tab--selected",
                    children=[
                        html.Div([
                            html.Div("Daily Returns & Rolling Volatility",
                                     className="section-heading"),
                            dcc.Loading(
                                dcc.Graph(id="returns-graph", style={"height": "540px"}),
                                type="circle", color="#58a6ff",
                            ),
                        ], className="tab-content"),
                    ],
                ),
            ],
        ),
    ], style={"padding": "0 2rem 2rem"}),

    # ── Hidden store for raw data ─────────────────────────────────────────
    dcc.Store(id="stock-store"),

], style={"minHeight": "100vh", "background": "#050a0f"})


# ── Helper: load model + make predictions ────────────────────────────────────

def load_predictions(ticker: str, df: pd.DataFrame, lookback: int = 60,
                     test_ratio: float = 0.2):
    """
    If a saved model exists for the ticker, run predictions on the test set
    and load the 30-day forecast. Returns (train_df, valid_df, future_dates, future_preds)
    or None if no model found.
    """
    model_path = f"models/{ticker}_lstm.keras"
    if not TF_AVAILABLE or not os.path.exists(model_path):
        return None

    try:
        model = keras_load_model(model_path)
        close = df[["Close"]].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close)

        split_idx   = int(len(scaled) * (1 - test_ratio))
        test_scaled = scaled[split_idx - lookback:]
        X_test, _   = create_sequences(test_scaled, lookback)

        preds_scaled = model.predict(X_test, verbose=0)
        preds        = scaler.inverse_transform(preds_scaled).flatten()

        train_df = df.iloc[:split_idx].copy()
        valid_df = df.iloc[split_idx:].copy()
        valid_df = valid_df.iloc[:len(preds)].copy()
        valid_df["Predicted"] = preds

        # 30-day forecast
        last_seq    = scaled[-lookback:]
        future_pred = forecast_future(model, last_seq, scaler, n_days=30)
        last_date   = df.index[-1]
        future_dates = pd.bdate_range(start=last_date, periods=31)[1:]  # business days

        return train_df, valid_df, future_dates, future_pred

    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        return None


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("stock-store", "data"),
    Input("ticker-dropdown", "value"),
    Input("period-dropdown", "value"),
)
def update_store(ticker, period):
    """Fetch stock data and cache in dcc.Store."""
    df = fetch_stock_data(ticker, period=period)
    df = add_moving_averages(df)
    # Flatten MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.reset_index().to_json(date_format="iso", orient="split")


@app.callback(
    Output("metric-strip", "children"),
    Input("stock-store", "data"),
    State("ticker-dropdown", "value"),
)
def update_metrics(store_json, ticker):
    if not store_json:
        return []
    df = pd.read_json(store_json, orient="split")
    df.set_index(df.columns[0], inplace=True)

    close = df["Close"]
    current = close.iloc[-1]
    prev    = close.iloc[-2]
    chg     = current - prev
    chg_pct = chg / prev * 100
    high52  = close.max()
    low52   = close.min()
    vol_avg = df["Volume"].mean() if "Volume" in df.columns else 0
    volatility = close.pct_change().std() * (252 ** 0.5) * 100

    color = "#3fb950" if chg >= 0 else "#f85149"
    arrow = "▲" if chg >= 0 else "▼"

    def card(label, value, sub="", color_val=None):
        return html.Div([
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value",
                     style={"color": color_val or "#e6edf3"}),
            html.Div(sub,   className="metric-sub"),
        ], className="metric-card")

    return [
        card("Current Price",  f"${current:.2f}",
             f"{arrow} {abs(chg):.2f} ({chg_pct:+.2f}%)", color),
        card("52W High",       f"${high52:.2f}",  "Rolling high", "#58a6ff"),
        card("52W Low",        f"${low52:.2f}",   "Rolling low",  "#f0883e"),
        card("Avg Volume",     f"{vol_avg/1e6:.1f}M", "shares/day"),
        card("Ann. Volatility",f"{volatility:.1f}%", "std of returns", "#bc8cff"),
        card("Ticker",         ticker, "Selected stock", "#58a6ff"),
    ]


@app.callback(
    Output("lstm-graph", "figure"),
    Input("stock-store", "data"),
    State("ticker-dropdown", "value"),
    Input("forecast-checklist", "value"),
)
def update_lstm_graph(store_json, ticker, show_forecast):
    if not store_json:
        return go.Figure()

    df = pd.read_json(store_json, orient="split")
    df.set_index(df.columns[0], inplace=True)

    fig = go.Figure()

    result = load_predictions(ticker, df)

    if result:
        train_df, valid_df, future_dates, future_preds = result

        # Actual close
        fig.add_trace(go.Scatter(
            x=train_df.index, y=train_df["Close"],
            name="Train (Actual)", line=dict(color="#58a6ff", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=valid_df.index, y=valid_df["Close"],
            name="Test (Actual)", line=dict(color="#8b949e", width=1.5),
        ))
        # Predicted
        fig.add_trace(go.Scatter(
            x=valid_df.index, y=valid_df["Predicted"],
            name="LSTM Predicted", line=dict(color="#f0883e", width=2, dash="dot"),
        ))
        # Forecast
        if "show" in (show_forecast or []):
            fig.add_trace(go.Scatter(
                x=list(future_dates), y=future_preds,
                name="30-Day Forecast",
                line=dict(color="#bc8cff", width=2, dash="dash"),
                fill="tozeroy",
                fillcolor="rgba(188,140,255,0.05)",
            ))
            # Add vertical line at today
            last = df.index[-1]
            fig.add_vline(x=last, line=dict(color="#30363d", dash="dot", width=1))
    else:
        # No model: just show close price with hint
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            name="Close Price", line=dict(color="#58a6ff", width=1.5),
        ))
        fig.add_annotation(
            text="⚠ No model found — run: python train_model.py --ticker " + ticker,
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#f0883e"),
            bgcolor="rgba(240,136,62,0.1)",
            bordercolor="#f0883e", borderwidth=1,
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"<b>{ticker}</b> — LSTM Price Prediction",
            font=dict(color="#e6edf3", size=16),
        ),
        xaxis_title="Date", yaxis_title="Price (USD)",
    )
    return fig


@app.callback(
    Output("candle-graph", "figure"),
    Input("stock-store", "data"),
    State("ticker-dropdown", "value"),
    Input("ma-checklist", "value"),
)
def update_candle_graph(store_json, ticker, ma_list):
    if not store_json:
        return go.Figure()

    df = pd.read_json(store_json, orient="split")
    df.set_index(df.columns[0], inplace=True)
    # Keep only last 252 rows for clarity
    df = df.tail(252)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.02,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        name="OHLC",
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149",
    ), row=1, col=1)

    # Moving averages
    ma_colors = {"MA20": "#58a6ff", "MA50": "#f0883e", "MA200": "#bc8cff"}
    for ma in (ma_list or []):
        if ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[ma], name=ma,
                line=dict(color=ma_colors[ma], width=1.5),
            ), row=1, col=1)

    # Volume bars
    colors = ["#3fb950" if c >= o else "#f85149"
               for c, o in zip(df["Close"], df["Open"])]
    if "Volume" in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=colors, opacity=0.7,
        ), row=2, col=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"<b>{ticker}</b> — Candlestick Chart (Last Year)",
            font=dict(color="#e6edf3", size=16),
        ),
        xaxis_rangeslider_visible=False,
        yaxis_title="Price (USD)",
        yaxis2_title="Volume",
    )
    return fig


@app.callback(
    Output("ma-graph", "figure"),
    Input("stock-store", "data"),
    State("ticker-dropdown", "value"),
    Input("ma-checklist", "value"),
)
def update_ma_graph(store_json, ticker, ma_list):
    if not store_json:
        return go.Figure()

    df = pd.read_json(store_json, orient="split")
    df.set_index(df.columns[0], inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Close",
        line=dict(color="#58a6ff", width=1.5),
    ))

    ma_colors = {"MA20": "#f0883e", "MA50": "#3fb950", "MA200": "#bc8cff"}
    for ma in (ma_list or []):
        if ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[ma], name=ma,
                line=dict(color=ma_colors[ma], width=1.8, dash="dot"),
            ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"<b>{ticker}</b> — Close Price & Moving Averages",
            font=dict(color="#e6edf3", size=16),
        ),
        xaxis_title="Date", yaxis_title="Price (USD)",
    )
    return fig


@app.callback(
    Output("compare-price-graph", "figure"),
    Output("compare-volume-graph", "figure"),
    Input("compare-dropdown", "value"),
    Input("period-dropdown", "value"),
)
def update_compare(selected, period):
    if not selected:
        empty = go.Figure().update_layout(**LAYOUT_DEFAULTS)
        return empty, empty

    COLORS = ["#58a6ff", "#3fb950", "#f0883e", "#bc8cff",
              "#f85149", "#d29922", "#79c0ff", "#56d364"]

    price_fig  = go.Figure()
    volume_fig = go.Figure()

    for i, t in enumerate(selected[:8]):
        df = fetch_stock_data(t, period=period)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c = COLORS[i % len(COLORS)]

        price_fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name=t,
            line=dict(color=c, width=1.8),
        ))
        if "Volume" in df.columns:
            volume_fig.add_trace(go.Bar(
                x=df.index, y=df["Volume"], name=t,
                marker_color=c, opacity=0.75,
            ))

    price_fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="<b>Close Price Comparison</b>",
                   font=dict(color="#e6edf3", size=15)),
        yaxis_title="Price (USD)",
    )
    volume_fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="<b>Trading Volume Comparison</b>",
                   font=dict(color="#e6edf3", size=15)),
        yaxis_title="Volume",
        barmode="group",
    )
    return price_fig, volume_fig


@app.callback(
    Output("returns-graph", "figure"),
    Input("stock-store", "data"),
    State("ticker-dropdown", "value"),
)
def update_returns_graph(store_json, ticker):
    if not store_json:
        return go.Figure()

    df = pd.read_json(store_json, orient="split")
    df.set_index(df.columns[0], inplace=True)

    returns     = df["Close"].pct_change().dropna() * 100
    rolling_vol = returns.rolling(21).std() * (252 ** 0.5)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["Daily Returns (%)", "Rolling 21-Day Volatility (Ann.)", "Returns Distribution"],
        row_heights=[0.35, 0.30, 0.35],
        vertical_spacing=0.08,
    )

    # Daily returns
    colors = ["#3fb950" if r >= 0 else "#f85149" for r in returns]
    fig.add_trace(go.Bar(
        x=returns.index, y=returns, name="Daily Return %",
        marker_color=colors, opacity=0.8,
    ), row=1, col=1)

    # Rolling volatility
    fig.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol,
        name="Ann. Volatility", line=dict(color="#bc8cff", width=2),
        fill="tozeroy", fillcolor="rgba(188,140,255,0.08)",
    ), row=2, col=1)

    # Histogram distribution
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=80, name="Returns Dist.",
        marker_color="#58a6ff", opacity=0.75,
    ), row=3, col=1)

    # Add 0 line for returns
    fig.add_hline(y=0, line=dict(color="#30363d", width=1), row=1, col=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"<b>{ticker}</b> — Daily Returns Analysis",
            font=dict(color="#e6edf3", size=16),
        ),
        showlegend=False,
        height=540,
    )
    return fig


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  📈 Stock Price Prediction Dashboard")
    print("="*55)
    print("  → Open http://127.0.0.1:8050 in your browser")
    print("  → To add LSTM predictions, first run:")
    print("      python train_model.py --ticker AAPL")
    print("="*55 + "\n")
    app.run(debug=True, port=8050)
