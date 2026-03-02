import axios from "axios";
import type {
  StrategyInfo,
  BacktestRequest,
  BacktestResponse,
  CompareRequest,
  HedgingRequest,
  HedgingResponse,
  OptionPricingRequest,
  OptionPricingResponse,
  RiskAnalyzeRequest,
  RiskMetrics,
} from "./types";

const api = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
});

// Normalise all error shapes into a plain Error
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg =
      err.response?.data?.detail ||
      err.response?.data?.error ||
      err.message ||
      "Unknown error";
    return Promise.reject(new Error(String(msg)));
  }
);

// ── Strategies ────────────────────────────────────────────────────────────────

export async function fetchStrategies(): Promise<StrategyInfo[]> {
  const res = await api.get("/strategies");
  return res.data.strategies;
}

// ── Backtest ──────────────────────────────────────────────────────────────────

export async function runBacktest(req: BacktestRequest): Promise<BacktestResponse> {
  const res = await api.post("/backtest", req);
  return res.data;
}

export async function compareStrategies(req: CompareRequest): Promise<BacktestResponse[]> {
  const res = await api.post("/backtest/compare", req);
  return res.data;
}

// ── Hedging ───────────────────────────────────────────────────────────────────

export async function simulateHedging(req: HedgingRequest): Promise<HedgingResponse> {
  const res = await api.post("/hedging/simulate", req);
  return res.data;
}

// ── Risk ──────────────────────────────────────────────────────────────────────

export async function analyzeRisk(req: RiskAnalyzeRequest): Promise<RiskMetrics> {
  const res = await api.post("/risk/analyze", req);
  return res.data;
}

// ── Data ──────────────────────────────────────────────────────────────────────

export async function fetchAssets(): Promise<string[]> {
  const res = await api.get("/data/assets");
  return (res.data.assets as { symbol: string }[]).map((a) => a.symbol);
}

// ── Pricing ───────────────────────────────────────────────────────────────────

export async function priceOption(req: OptionPricingRequest): Promise<OptionPricingResponse> {
  const res = await api.post("/pricing/option", req);
  return res.data;
}
