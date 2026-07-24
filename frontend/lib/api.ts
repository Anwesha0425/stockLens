// lib/api.ts — Typed API client for the FastAPI backend

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface OHLCVRecord {
  date: string;
  open: number; high: number; low: number; close: number; volume: number;
  ma20: number | null; ma50: number | null; ma200: number | null;
}

export interface StockResponse {
  ticker: string;
  data: OHLCVRecord[];
}

export interface PredRecord {
  date: string; actual: number; lstm: number;
}

export interface ForecastRecord {
  date: string; mean: number; lower: number; upper: number;
}

export interface PredictResponse {
  ticker: string;
  predictions: PredRecord[];
  forecast: ForecastRecord[];
}

export interface ModelMetrics {
  RMSE: number; MAE: number; "R²": number; MASE?: number;
}

export interface ComparisonMetrics {
  lstm: ModelMetrics; xgb: ModelMetrics; naive: ModelMetrics;
  cv_rmse_mean: number; cv_rmse_std: number;
}

export interface CompareEntry {
  dates: string[]; values: number[]; raw: number[];
}

export interface ReturnRecord {
  date: string; return_pct: number; vol21: number;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { next: { revalidate: 60 } });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  stock:   (ticker: string, period = "2y") =>
    get<StockResponse>(`/api/stock/${ticker}?period=${period}`),
  predict: (ticker: string) =>
    get<PredictResponse>(`/api/predict/${ticker}`),
  compare: (tickers: string[], period = "1y") =>
    get<Record<string, CompareEntry>>(`/api/compare?tickers=${tickers.join(",")}&period=${period}`),
  metrics: (ticker: string) =>
    get<{ ticker: string; metrics: ComparisonMetrics }>(`/api/models/${ticker}`),
  returns: (ticker: string, period = "1y") =>
    get<{ ticker: string; data: ReturnRecord[] }>(`/api/returns/${ticker}?period=${period}`),
};
