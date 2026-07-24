"use client";
import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Header from "@/components/Header";
import MetricCard from "@/components/MetricCard";
import PredictionChart from "@/components/PredictionChart";
import MAChart from "@/components/MAChart";
import CompareChart from "@/components/CompareChart";
import ReturnsChart from "@/components/ReturnsChart";
import ModelComparisonTable from "@/components/ModelComparisonTable";
import { api, OHLCVRecord } from "@/lib/api";

// ── Tab configuration ────────────────────────────────────────────────────────
const TABS = [
  { id: "lstm",    label: "🤖 LSTM Prediction"  },
  { id: "ma",      label: "📊 Moving Averages"   },
  { id: "compare", label: "⚖ Compare"            },
  { id: "returns", label: "📉 Returns"            },
  { id: "models",  label: "🏆 Model Comparison"  },
];

const COMPARE_DEFAULT = ["AAPL", "MSFT", "NVDA", "TSLA"];

// ── Skeleton loader ──────────────────────────────────────────────────────────
function Skeleton({ h = "h-[400px]" }: { h?: string }) {
  return <div className={`skeleton rounded-xl w-full ${h}`} />;
}

// ── Main Page ────────────────────────────────────────────────────────────────
export default function HomePage() {
  const [ticker,  setTicker]  = useState("AAPL");
  const [period,  setPeriod]  = useState("2y");
  const [tab,     setTab]     = useState("lstm");
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState<string | null>(null);

  // Data states
  const [ohlcv,       setOhlcv]       = useState<OHLCVRecord[]>([]);
  const [predData,    setPredData]    = useState<any>(null);
  const [compareData, setCompareData] = useState<any>({});
  const [returnsData, setReturnsData] = useState<any[]>([]);
  const [modelData,   setModelData]   = useState<any>(null);

  // ── Fetch stock data ───────────────────────────────────────────────────────
  const fetchStock = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [stock, returns] = await Promise.all([
        api.stock(ticker, period),
        api.returns(ticker, period),
      ]);
      setOhlcv(stock.data);
      setReturnsData(returns.data);

      // Attempt predictions (may not exist yet)
      try {
        const pred = await api.predict(ticker);
        setPredData(pred);
      } catch { setPredData(null); }

      // Attempt model metrics
      try {
        const m = await api.metrics(ticker);
        setModelData(m.metrics);
      } catch { setModelData(null); }

    } catch (e: any) {
      setError(e.message ?? "Failed to fetch data. Is the API running?");
    } finally { setLoading(false); }
  }, [ticker, period]);

  const fetchCompare = useCallback(async () => {
    try {
      const data = await api.compare(COMPARE_DEFAULT, period);
      setCompareData(data);
    } catch {}
  }, [period]);

  useEffect(() => { fetchStock(); }, [fetchStock]);
  useEffect(() => { fetchCompare(); }, [fetchCompare]);

  // ── Derived data for charts ────────────────────────────────────────────────
  const chartData = (() => {
    const map: Record<string, any> = {};
    ohlcv.forEach(r => { map[r.date] = { ...r }; });

    if (predData) {
      const minDate = ohlcv.length > 0 ? ohlcv[0].date : "0000-00-00";

      predData.predictions?.forEach((p: any) => {
        if (p.date >= minDate) {
          if (map[p.date]) { map[p.date].lstm = p.lstm; map[p.date].actual = p.actual; }
          else map[p.date] = { date: p.date, actual: p.actual, lstm: p.lstm };
        }
      });
      predData.forecast?.forEach((f: any) => {
        map[f.date] = {
          date: f.date,
          forecast_mean:  f.mean,
          forecast_lower: f.lower,
          forecast_upper: f.upper,
        };
      });
    }
    return Object.values(map).sort((a: any, b: any) =>
      a.date.localeCompare(b.date));
  })();

  // ── Metric cards ──────────────────────────────────────────────────────────
  const latest  = ohlcv[ohlcv.length - 1];
  const prev    = ohlcv[ohlcv.length - 2];
  const chg     = latest && prev ? latest.close - prev.close : 0;
  const chgPct  = prev ? (chg / prev.close) * 100 : 0;
  const high52  = Math.max(...ohlcv.map(r => r.high));
  const low52   = Math.min(...ohlcv.map(r => r.low));
  const avgVol  = ohlcv.reduce((s, r) => s + r.volume, 0) / (ohlcv.length || 1);
  const returns = ohlcv.map((r, i) => i > 0 ? (r.close - ohlcv[i-1].close) / ohlcv[i-1].close : 0);
  const annVol  = Math.sqrt(252) * Math.sqrt(returns.reduce((s, r) => s + r*r, 0) / (returns.length || 1)) * 100;

  return (
    <div className="min-h-screen bg-[var(--bg)]">
      <Header ticker={ticker} period={period} onTicker={setTicker} onPeriod={setPeriod} />

      <main className="max-w-screen-2xl mx-auto px-4 py-6 space-y-6">

        {/* Error banner */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="rounded-xl border border-[var(--danger)] bg-[var(--danger)]/10
                         p-4 text-sm text-[var(--danger)]"
            >
              ⚠ {error}
              <span className="ml-2 text-[var(--muted)]">
                Start the API: <code className="font-mono bg-[var(--surface)] px-1 rounded">
                  cd backend &amp;&amp; uvicorn api:app --reload
                </code>
              </span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Metric cards */}
        <div className="flex gap-3 flex-wrap">
          {loading ? (
            Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="skeleton h-20 flex-1 min-w-[140px] rounded-xl" />
            ))
          ) : latest ? (
            <>
              <MetricCard label="Price"    value={`$${latest.close.toFixed(2)}`}
                sub={`${chg >= 0 ? "+" : ""}${chg.toFixed(2)} (${chgPct.toFixed(2)}%)`}
                trend={chg >= 0 ? "up" : "down"} delay={0} />
              <MetricCard label="52W High" value={`$${high52.toFixed(2)}`}
                trend="neutral" delay={0.05} />
              <MetricCard label="52W Low"  value={`$${low52.toFixed(2)}`}
                trend="neutral" delay={0.1} />
              <MetricCard label="Avg Vol"  value={`${(avgVol/1e6).toFixed(1)}M`}
                sub="shares/day" delay={0.15} />
              <MetricCard label="Ann. Vol" value={`${annVol.toFixed(1)}%`}
                sub="stddev of returns" delay={0.2} />
              <MetricCard label="Ticker"   value={ticker} delay={0.25} />
            </>
          ) : null}
        </div>

        {/* Tab bar */}
        <div className="flex gap-1 overflow-x-auto pb-1">
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`tab-btn ${tab === t.id ? "active" : ""}`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab panels */}
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
            className="card p-5"
          >

            {/* LSTM Prediction */}
            {tab === "lstm" && (
              <>
                <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
                  <h2 className="font-semibold text-[var(--text)]">
                    {ticker} — LSTM Price Prediction + 30-Day Forecast
                  </h2>
                  {!predData && !loading && (
                    <span className="badge badge-down text-xs">
                      No model — run: python train_model.py --ticker {ticker}
                    </span>
                  )}
                  {predData && (
                    <span className="badge" style={{
                      background: "rgba(107,124,58,0.15)", color: "var(--primary)"
                    }}>
                      ✓ Model loaded · 90% MC-Dropout CI
                    </span>
                  )}
                </div>
                {loading ? <Skeleton /> : <PredictionChart data={chartData} ticker={ticker} />}
              </>
            )}

            {/* Moving Averages */}
            {tab === "ma" && (
              <>
                <h2 className="font-semibold text-[var(--text)] mb-4">
                  {ticker} — Price & Moving Averages (MA20 / MA50 / MA200)
                </h2>
                {loading ? <Skeleton /> : (
                  <MAChart
                    data={ohlcv.map(r => ({
                      date:  r.date,
                      close: r.close,
                      ma20:  r.ma20,
                      ma50:  r.ma50,
                      ma200: r.ma200,
                    }))}
                    showMA={{ ma20: true, ma50: true, ma200: true }}
                  />
                )}
              </>
            )}

            {/* Compare */}
            {tab === "compare" && (
              <>
                <h2 className="font-semibold text-[var(--text)] mb-1">
                  Multi-Stock Comparison (Normalised, Base = 100)
                </h2>
                <p className="text-xs text-[var(--muted)] mb-4">
                  Comparing: {COMPARE_DEFAULT.join(", ")} · {period.toUpperCase()} window
                </p>
                {Object.keys(compareData).length === 0
                  ? <Skeleton />
                  : <CompareChart data={compareData} />
                }
              </>
            )}

            {/* Returns */}
            {tab === "returns" && (
              <>
                <h2 className="font-semibold text-[var(--text)] mb-4">
                  {ticker} — Daily Returns & Rolling Volatility
                </h2>
                {loading ? <Skeleton /> : <ReturnsChart data={returnsData} />}
              </>
            )}

            {/* Model Comparison */}
            {tab === "models" && (
              <>
                <h2 className="font-semibold text-[var(--text)] mb-4">
                  {ticker} — Model Comparison: LSTM vs XGBoost vs Naive Baseline
                </h2>
                {modelData ? (
                  <ModelComparisonTable
                    metrics={modelData}
                    cv_rmse_mean={modelData.cv_rmse_mean}
                    cv_rmse_std={modelData.cv_rmse_std}
                  />
                ) : (
                  <div className="text-sm text-[var(--muted)] p-6 text-center rounded-xl
                                  border border-dashed border-[var(--border)]">
                    No comparison data — run{" "}
                    <code className="font-mono bg-[var(--surface)] px-1 rounded">
                      python train_model.py --ticker {ticker}
                    </code>{" "}
                    first.
                  </div>
                )}
              </>
            )}

          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
