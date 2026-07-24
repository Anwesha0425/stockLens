"use client";
import { motion } from "framer-motion";
import { TrendingUp, TrendingDown } from "lucide-react";

interface Props {
  label: string;
  value: string;
  sub?: string;
  trend?: "up" | "down" | "neutral";
  delay?: number;
}

export default function MetricCard({ label, value, sub, trend, delay = 0 }: Props) {
  const trendColor =
    trend === "up"   ? "text-[var(--success)]" :
    trend === "down" ? "text-[var(--danger)]"  : "text-[var(--accent)]";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0  }}
      transition={{ duration: 0.45, delay, ease: "easeOut" }}
      className="card metric-bar relative p-4 overflow-hidden flex-1 min-w-[150px]"
    >
      {/* Subtle background glow */}
      <div className="absolute inset-0 opacity-[0.03] bg-[radial-gradient(ellipse_at_top_left,var(--primary),transparent_70%)] pointer-events-none" />

      <p className="text-[0.7rem] font-semibold uppercase tracking-widest text-[var(--muted)] mb-1">
        {label}
      </p>

      <div className="flex items-end gap-2">
        <span className={`text-[1.6rem] font-bold font-mono leading-none ${trendColor}`}>
          {value}
        </span>
        {trend === "up"   && <TrendingUp  size={16} className="text-[var(--success)] mb-1" />}
        {trend === "down" && <TrendingDown size={16} className="text-[var(--danger)]  mb-1" />}
      </div>

      {sub && (
        <p className="text-[0.72rem] text-[var(--muted)] mt-1">{sub}</p>
      )}
    </motion.div>
  );
}
