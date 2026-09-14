export type ObservationType = "actual" | "nowcast" | "forecast" | "nowcast_eval";

export type RiskSignal = "LOW" | "MODERATE" | "ELEVATED" | "HIGH" | "CRITICAL";

export interface DriverContribution {
  feature: string;
  contribution_pp: number;
  direction: "up" | "down" | "neutral";
  note: string;
}

export interface NowcastResult {
  target: string;
  reference_period: string;
  nowcast_value: number;
  actual_last_release: number | null;
  actual_last_period: string | null;
  change_pp: number | null;
  confidence_low: number;
  confidence_high: number;
  risk_signal: RiskSignal;
  observation_type: ObservationType;
  model_used: string;
  model_confidence: "low" | "moderate" | "high";
  drivers: DriverContribution[];
  news_signals: string[];
  is_synthetic: boolean;
  disclaimer: string;
}

export interface LeaderboardRow {
  model: string;
  display_name: string;
  rmse: number;
  mae: number;
  mape: number;
  r2: number;
  bias?: number;
  directional_accuracy?: number;
  rank: number;
  n_observations: number;
}

export interface ScenarioInput {
  gdp_growth: number;
  credit_growth: number;
  policy_rate: number;
  inflation: number;
  exchange_rate: number;
  news_stress: number;
}

export interface WorkflowConfig {
  target: string;
  frequency: string;
  model_selection: "manual" | "auto" | "ensemble";
  selected_models: string[];
  sentiment_model: string;
  sentiment_auto: boolean;
  variable_selection: "manual" | "automatic" | "hybrid";
  selected_indicators: string[];
  mandatory_indicators: string[];
  backtest_mode: "expanding" | "rolling";
  rolling_window: number;
  include_news: boolean;
}

export interface WorkflowResponse {
  selection: {
    leaderboard: LeaderboardRow[];
    best_model: string;
    ensemble_weights: Record<string, number>;
    features: string[];
    failed: string[];
  };
  nowcast: NowcastResult;
  timeseries: Array<{
    period: string;
    value: number | null;
    observation_type: string;
    low?: number | null;
    high?: number | null;
    actual?: number;
  }>;
  news: {
    current: number;
    previous: number;
    change: number;
    article_count: number;
    negative_ratio: number;
    top_topics: Array<{ topic: string; count: number }>;
    series: Array<{ period: string; news_stress: number; count: number }>;
    sector_heatmap: Array<{ sector: string; stress: number }>;
    recent_articles: Array<{
      id: string;
      headline: string;
      published_at: string;
      source: string;
      url?: string;
      sentiment?: string;
      sector?: string;
      npl_relevance?: number;
      is_synthetic: boolean;
    }>;
  };
  analyst: {
    summary: string;
    bullets: string[];
    risks: string[];
    confidence_note: string;
    is_synthetic: boolean;
  };
}

export interface ApiEnvelope<T> {
  data: T;
  mode: string;
  is_synthetic: boolean;
  message?: string | null;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function apiGet<T>(path: string): Promise<ApiEnvelope<T>> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${path} failed: ${res.status}`);
  return res.json();
}

export async function apiPost<T>(path: string, body: unknown): Promise<ApiEnvelope<T>> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API ${path} failed: ${res.status}`);
  return res.json();
}

export const DEFAULT_CONFIG: WorkflowConfig = {
  target: "gross_npl_ratio",
  frequency: "monthly",
  model_selection: "auto",
  selected_models: [],
  sentiment_model: "tfidf_logistic",
  sentiment_auto: false,
  variable_selection: "automatic",
  selected_indicators: [],
  mandatory_indicators: ["gdp_growth", "credit_growth", "lending_rate", "news_stress"],
  backtest_mode: "expanding",
  rolling_window: 36,
  include_news: true,
};
