import { useEffect, useState } from "react";
import { fetchStrategies } from "../api/client";
import type { StrategyInfo } from "../api/types";

export function useStrategies() {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStrategies()
      .then(setStrategies)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return { strategies, loading, error };
}
