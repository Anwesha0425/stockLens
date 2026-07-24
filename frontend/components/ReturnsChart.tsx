"use client";
import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { motion } from "framer-motion";

interface ReturnPoint { date: string; return_pct: number; vol21: number; }
interface Props { data: ReturnPoint[]; }

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="card p-3 text-xs shadow-lg min-w-[160px]">
      <p className="font-semibold mb-2 text-[var(--text)]">{label}</p>
      {payload.map((p: any) => (
        <div key={p.name} className="flex justify-between gap-4 mb-1">
          <span style={{ color: p.color }}>{p.name}</span>
          <span className="font-mono font-semibold text-[var(--accent)]">
            {p.value?.toFixed(3)}{p.name.includes("Vol") ? "%" : "%"}
          </span>
        </div>
      ))}
    </div>
  );
};

export default function ReturnsChart({ data }: Props) {
  // Colour bars by positive/negative
  const barColor = (v: number) =>
    v >= 0 ? "var(--success)" : "var(--danger)";

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      <ResponsiveContainer width="100%" height={420}>
        <ComposedChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="date" tick={{ fill: "var(--muted)", fontSize: 11 }}
            tickLine={false} tickFormatter={(v) => v?.slice(5)}
            interval="preserveStartEnd"
          />
          <YAxis
            yAxisId="left"
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            tickLine={false} tickFormatter={(v) => `${v}%`} width={55}
          />
          <YAxis
            yAxisId="right" orientation="right"
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            tickLine={false} tickFormatter={(v) => `${v}%`} width={55}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: "12px", color: "var(--muted)", paddingTop: "10px" }} />

          <Bar
            yAxisId="left"
            dataKey="return_pct"
            name="Daily Return %"
            fill="var(--primary)"
            opacity={0.75}
            radius={[2, 2, 0, 0]}
            // cell-level colour based on value
            label={false}
          />
          <Line
            yAxisId="right"
            dataKey="vol21"
            name="21d Volatility %"
            stroke="var(--secondary)"
            strokeWidth={2}
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </motion.div>
  );
}
