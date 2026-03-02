import { useState } from "react";
import { runBacktest } from "../api/client";
import type { BacktestRequest, BacktestResponse } from "../api/types";

export function useBacktest() {
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(req: BacktestRequest) {
    setLoading(true);
    setError(null);
    try {
      const res = await runBacktest(req);
      setResult(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return { result, loading, error, submit };
}
