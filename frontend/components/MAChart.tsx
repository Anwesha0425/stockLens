"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { motion } from "framer-motion";

interface DataPoint {
  date: string;
  close: number;
  ma20?: number | null;
  ma50?: number | null;
  ma200?: number | null;
}

interface Props {
  data: DataPoint[];
  showMA?: { ma20: boolean; ma50: boolean; ma200: boolean };
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="card p-3 text-xs min-w-[150px] shadow-lg">
      <p className="font-semibold mb-2 text-[var(--text)]">{label}</p>
      {payload.map((p: any) => p.value != null && (
        <div key={p.name} className="flex justify-between gap-4 mb-1">
          <span style={{ color: p.color }}>{p.name}</span>
          <span className="font-mono font-semibold text-[var(--accent)]">
            ${p.value?.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
};

export default function MAChart({ data, showMA }: Props) {
  const show = showMA ?? { ma20: true, ma50: true, ma200: false };
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="date" tick={{ fill: "var(--muted)", fontSize: 11 }}
            tickLine={false} tickFormatter={(v) => v?.slice(5)}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            tickLine={false} tickFormatter={(v) => `$${v}`} width={65}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: "12px", color: "var(--muted)", paddingTop: "10px" }} />
          <Line dataKey="close" name="Close" stroke="var(--muted)"
            strokeWidth={1.5} dot={false} />
          {show.ma20 && (
            <Line dataKey="ma20" name="MA 20" stroke="var(--primary)"
              strokeWidth={1.8} dot={false} strokeDasharray="4 2" />
          )}
          {show.ma50 && (
            <Line dataKey="ma50" name="MA 50" stroke="var(--secondary)"
              strokeWidth={1.8} dot={false} strokeDasharray="6 3" />
          )}
          {show.ma200 && (
            <Line dataKey="ma200" name="MA 200" stroke="var(--accent)"
              strokeWidth={2} dot={false} strokeDasharray="8 4" />
          )}
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  );
}
