"use client";
import { motion } from "framer-motion";

interface ModelMetrics {
  RMSE: number; MAE: number; "R²": number; MASE?: number;
}
interface Props {
  metrics: { lstm: ModelMetrics; xgb: ModelMetrics; naive: ModelMetrics };
  cv_rmse_mean?: number; cv_rmse_std?: number;
}

const fmt = (v: number | undefined) =>
  v !== undefined ? v.toFixed(4) : "—";

const winner = (vals: number[], lowerIsBetter = true) => {
  const best = lowerIsBetter ? Math.min(...vals) : Math.max(...vals);
  return vals.map(v => v === best);
};

export default function ModelComparisonTable({ metrics, cv_rmse_mean, cv_rmse_std }: Props) {
  const rows = [
    { label: "RMSE",    values: [metrics.lstm.RMSE, metrics.xgb.RMSE, metrics.naive.RMSE],       lowerBetter: true  },
    { label: "MAE",     values: [metrics.lstm.MAE,  metrics.xgb.MAE,  metrics.naive.MAE],        lowerBetter: true  },
    { label: "R²",      values: [metrics.lstm["R²"],metrics.xgb["R²"],metrics.naive["R²"]],      lowerBetter: false },
    { label: "MASE",    values: [metrics.lstm.MASE ?? 0, metrics.xgb.MASE ?? 0, metrics.naive.MASE ?? 0], lowerBetter: true },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {cv_rmse_mean !== undefined && (
        <p className="text-xs text-[var(--muted)] mb-3 flex gap-2 items-center">
          <span className="badge" style={{ background: "rgba(107,124,58,0.15)", color: "var(--primary)" }}>
            Walk-Forward CV
          </span>
          RMSE {cv_rmse_mean.toFixed(4)} ± {cv_rmse_std?.toFixed(4)} across 5 folds
        </p>
      )}

      <div className="overflow-x-auto rounded-xl border border-[var(--border)]">
        <table className="data-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th className="text-right">LSTM</th>
              <th className="text-right">XGBoost</th>
              <th className="text-right">Naive</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ label, values, lowerBetter }) => {
              const wins = winner(values.filter(v => v !== 0), lowerBetter);
              return (
                <tr key={label}>
                  <td className="font-semibold text-[var(--muted)] uppercase text-[0.7rem] tracking-wider">
                    {label}
                  </td>
                  {values.map((v, i) => (
                    <td
                      key={i}
                      className={`text-right ${wins[i]
                        ? "text-[var(--success)] font-bold"
                        : "text-[var(--text)]"
                      }`}
                    >
                      {fmt(v)}
                      {wins[i] && (
                        <span className="ml-1 text-[0.65rem] text-[var(--success)]">★</span>
                      )}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-[0.68rem] text-[var(--muted)] mt-2">
        ★ = best value · MASE &lt;1 means model beats naive baseline
      </p>
    </motion.div>
  );
}
