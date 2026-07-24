"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import { TrendingUp, BarChart2 } from "lucide-react";
import ThemeToggle from "./ThemeToggle";

const TICKERS = ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","JPM","WMT","XOM"];
const PERIODS  = [
  { label: "1M",  value: "1mo" },
  { label: "3M",  value: "3mo" },
  { label: "6M",  value: "6mo" },
  { label: "1Y",  value: "1y"  },
  { label: "2Y",  value: "2y"  },
  { label: "5Y",  value: "5y"  },
];

interface Props {
  ticker:    string;
  period:    string;
  onTicker:  (t: string) => void;
  onPeriod:  (p: string) => void;
}

export default function Header({ ticker, period, onTicker, onPeriod }: Props) {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0  }}
      transition={{ duration: 0.5 }}
      className="sticky top-0 z-50 border-b border-[var(--border)]
                 bg-[var(--bg)]/90 backdrop-blur-md"
    >
      <div className="max-w-screen-2xl mx-auto px-4 py-3 flex items-center gap-4 flex-wrap">

        {/* Logo */}
        <div className="flex items-center gap-2 mr-2">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center
                          bg-[var(--primary)] text-white">
            <TrendingUp size={16} />
          </div>
          <span className="font-bold text-[var(--text)] text-sm hidden sm:block">
            StockLens
          </span>
        </div>

        {/* Ticker selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-[var(--muted)] font-medium">Ticker</span>
          <select
            value={ticker}
            onChange={e => onTicker(e.target.value)}
            className="select-field"
          >
            {TICKERS.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        {/* Period selector */}
        <div className="flex gap-1 p-1 rounded-lg bg-[var(--surface)] border border-[var(--border)]">
          {PERIODS.map(p => (
            <button
              key={p.value}
              onClick={() => onPeriod(p.value)}
              className={`px-3 py-1 rounded-md text-xs font-semibold transition-all duration-150 ${
                period === p.value
                  ? "bg-[var(--primary)] text-white shadow-sm"
                  : "text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--surface2)]"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-3">
          <span className="text-xs text-[var(--muted)] hidden md:flex items-center gap-1">
            <BarChart2 size={12} /> Powered by Bidirectional LSTM · XGBoost · Walk-Forward CV
          </span>
          <ThemeToggle />
        </div>
      </div>
    </motion.header>
  );
}
