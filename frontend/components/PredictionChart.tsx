"use client";
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { motion } from "framer-motion";

interface DataPoint {
  date: string;
  actual?: number;
  lstm?: number;
  forecast_mean?: number;
  forecast_lower?: number;
  forecast_upper?: number;
}

interface Props { data: DataPoint[]; ticker: string; }

const fmt = (v: number) => `$${v?.toFixed(2)}`;

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="card p-3 text-xs min-w-[160px] shadow-lg">
      <p className="font-semibold text-[var(--text)] mb-2">{label}</p>
      {payload.map((p: any) => (
        <div key={p.name} className="flex justify-between gap-4 mb-1">
          <span style={{ color: p.color }}>{p.name}</span>
          <span className="font-mono text-[var(--accent)] font-semibold">
            {fmt(p.value)}
          </span>
        </div>
      ))}
    </div>
  );
};

export default function PredictionChart({ data, ticker }: Props) {
  const lastActual = data.findLast(d => d.actual !== undefined);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="w-full h-full"
    >
      <ResponsiveContainer width="100%" height={440}>
        <ComposedChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="date"
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            tickLine={false}
            tickFormatter={(v) => v?.slice(5)}   // MM-DD
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            tickLine={false}
            tickFormatter={(v) => `$${v}`}
            width={65}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: "12px", color: "var(--muted)", paddingTop: "12px" }}
          />

          {/* MC-Dropout confidence band */}
          <Area
            dataKey="forecast_upper"
            name="90% CI Upper"
            stroke="none"
            fill="var(--primary)"
            fillOpacity={0.12}
            legendType="none"
          />
          <Area
            dataKey="forecast_lower"
            name="90% CI Lower"
            stroke="none"
            fill="var(--bg)"
            fillOpacity={1}
            legendType="none"
          />

          {/* Actual price */}
          <Line
            dataKey="actual"
            name="Actual"
            stroke="var(--muted)"
            strokeWidth={1.8}
            dot={false}
            activeDot={{ r: 4, fill: "var(--muted)" }}
          />

          {/* LSTM prediction */}
          <Line
            dataKey="lstm"
            name="LSTM Predicted"
            stroke="var(--primary)"
            strokeWidth={2}
            strokeDasharray="5 3"
            dot={false}
            activeDot={{ r: 4, fill: "var(--primary)" }}
          />

          {/* MC Forecast mean */}
          <Line
            dataKey="forecast_mean"
            name="30-Day Forecast"
            stroke="var(--secondary)"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "var(--secondary)" }}
          />

          {/* Divider between historical and forecast */}
          {lastActual && (
            <ReferenceLine
              x={lastActual.date}
              stroke="var(--accent)"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{
                value: "Today",
                position: "top",
                fill: "var(--accent)",
                fontSize: 10,
              }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </motion.div>
  );
}
