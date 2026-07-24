"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { motion } from "framer-motion";

const COLORS = [
  "var(--chart1)", "var(--chart2)", "var(--chart3)",
  "var(--chart4)", "var(--chart5)",
];

interface Props {
  data: Record<string, { dates: string[]; values: number[] }>;
}

export default function CompareChart({ data }: Props) {
  const tickers = Object.keys(data);
  if (!tickers.length) return null;

  // Merge into unified date array
  const dateSet = new Set<string>();
  tickers.forEach(t => data[t].dates?.forEach(d => dateSet.add(d)));
  const dates = Array.from(dateSet).sort();

  const merged = dates.map(date => {
    const row: Record<string, number | string> = { date };
    tickers.forEach((t, i) => {
      const idx = data[t].dates?.indexOf(date) ?? -1;
      if (idx !== -1) row[t] = data[t].values[idx];
    });
    return row;
  });

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={merged} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="date" tick={{ fill: "var(--muted)", fontSize: 11 }}
            tickLine={false} tickFormatter={(v) => v?.slice(5)}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            tickLine={false} tickFormatter={(v) => `${v}`} width={55}
            label={{ value: "Normalised (100=start)", angle: -90, position: "insideLeft",
                     fill: "var(--muted)", fontSize: 10, dx: -5 }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--surface)", border: "1px solid var(--border)",
              borderRadius: "8px", fontSize: "12px", color: "var(--text)",
            }}
            formatter={(v: any) => typeof v === "number" ? v.toFixed(2) : v}
          />
          <Legend wrapperStyle={{ fontSize: "12px", color: "var(--muted)", paddingTop: "10px" }} />
          {tickers.map((t, i) => (
            <Line
              key={t} dataKey={t} name={t}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2} dot={false}
              activeDot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  );
}
