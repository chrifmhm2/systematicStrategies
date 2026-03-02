import { useState } from "react";
import { analyzeRisk } from "../api/client";
import type { RiskAnalyzeRequest, RiskMetrics } from "../api/types";

export function useRiskMetrics() {
  const [metrics, setMetrics] = useState<RiskMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function analyze(req: RiskAnalyzeRequest) {
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeRisk(req);
      setMetrics(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return { metrics, loading, error, analyze };
}
