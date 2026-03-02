// TypeScript interfaces matching the Phase 5 Pydantic schemas exactly.

export interface StrategyInfo {
  id: string;
  name: string;
  family: "hedging" | "allocation" | "signal";
  description: string;
  params: Record<string, ParamSchema>;
}

export interface ParamSchema {
  type: "integer" | "number" | "boolean" | "string";
  default: number | boolean | string;
  description?: string;
  min?: number;
  max?: number;
  enum?: string[];
}

// ── Backtest ──────────────────────────────────────────────────────────────────

export interface BacktestRequest {
  strategy_id: string;
  symbols: string[];
  start_date: string;
  end_date: string;
  initial_value: number;
  params: Record<string, unknown>;
  data_source: "simulated" | "yahoo";
}

export interface BacktestResponse {
  portfolio_values: Record<string, number>;
  benchmark_values: Record<string, number> | null;
  weights_history: Record<string, Record<string, number>>;
  risk_metrics: RiskMetrics;
  trades_log: TradeRecord[];
  computation_time_ms: number;
  strategy_name: string;
}

export type RiskMetrics = Record<string, number | null>;

export interface TradeRecord {
  date: string;
  symbol: string;
  direction: string;
  shares: number;
  price: number;
  value: number;
}

// ── Compare ───────────────────────────────────────────────────────────────────

export interface StrategySpec {
  strategy_id: string;
  params: Record<string, unknown>;
}

export interface CompareRequest {
  strategies: StrategySpec[];
  symbols: string[];
  start_date: string;
  end_date: string;
  initial_value: number;
  data_source: "simulated" | "yahoo";
}

// ── Hedging ───────────────────────────────────────────────────────────────────

export interface HedgingRequest {
  option_type: "call" | "put";
  weights: number[];
  symbols: string[];
  strike: number;
  maturity_years: number;
  risk_free_rate: number;
  volatilities: number[];
  correlation_matrix: number[][];
  initial_spots: number[];
  n_simulations: number;
  rebalancing_frequency: "daily" | "weekly" | "monthly";
  data_source: "simulated" | "yahoo";
  n_paths: number;
}

export interface HedgingPath {
  portfolio_values: Record<string, number>;
  tracking_error: number;
}

export interface HedgingResponse {
  paths: HedgingPath[];
  average_tracking_error: number;
  initial_option_price: number;
  initial_option_price_ci: [number, number];
}

// ── Option Pricing ────────────────────────────────────────────────────────────

export interface OptionPricingRequest {
  option_type: "call" | "put";
  S: number;
  K: number;
  T: number;
  r: number;
  sigma: number;
  method: "bs" | "mc";
  n_simulations?: number;
}

export interface OptionPricingResponse {
  price: number;
  std_error: number | null;
  confidence_interval: [number, number] | null;
  deltas: number[] | null;
  greeks: {
    delta: number | number[];
    gamma?: number;
    vega?: number;
    theta?: number;
    rho?: number;
  };
}

// ── Risk Analyze ──────────────────────────────────────────────────────────────

export interface RiskAnalyzeRequest {
  portfolio_values: Record<string, number>;
  benchmark_values?: Record<string, number>;
  risk_free_rate: number;
}
