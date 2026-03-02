import { useState } from "react";
import { simulateHedging } from "../api/client";
import type { HedgingResponse } from "../api/types";
import MetricCard from "../components/common/MetricCard";
import LoadingSpinner from "../components/common/LoadingSpinner";
import ErrorBanner from "../components/common/ErrorBanner";
import EquityCurve from "../components/charts/EquityCurve";
import { CHART_COLORS, THEME } from "../utils/colors";
import { formatNumber } from "../utils/formatters";

const DEFAULT_SYMBOLS = ["AAPL", "MSFT"];

export default function HedgingPage() {
  const [symbols, setSymbols] = useState(DEFAULT_SYMBOLS.join(", "));
  const [weights, setWeights] = useState("0.5, 0.5");
  const [spots, setSpots] = useState("150, 300");
  const [vols, setVols] = useState("0.25, 0.22");
  const [strike, setStrike] = useState(200);
  const [maturity, setMaturity] = useState(1.0);
  const [riskFreeRate, setRiskFreeRate] = useState(0.05);
  const [nPaths, setNPaths] = useState(5);
  const [nSim, setNSim] = useState(5_000);
  const [freq, setFreq] = useState<"daily" | "weekly" | "monthly">("weekly");

  const [result, setResult] = useState<HedgingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function parseFloats(s: string): number[] {
    return s.split(",").map((x) => parseFloat(x.trim())).filter((x) => !isNaN(x));
  }
  function parseStrings(s: string): string[] {
    return s.split(",").map((x) => x.trim()).filter(Boolean);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const syms = parseStrings(symbols);
    const ws = parseFloats(weights);
    const sp = parseFloats(spots);
    const vs = parseFloats(vols);
    const n = syms.length;

    if (syms.length === 0 || ws.length !== n || sp.length !== n || vs.length !== n) {
      setError("Symbols, weights, spots, and volatilities must all have the same count.");
      return;
    }
    const sumW = ws.reduce((a, b) => a + b, 0);
    if (Math.abs(sumW - 1) > 0.01) {
      setError(`Weights must sum to 1 (currently ${sumW.toFixed(3)}).`);
      return;
    }

    // Identity correlation matrix
    const corr = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1.0 : 0.5))
    );

    setLoading(true);
    try {
      const res = await simulateHedging({
        option_type: "call",
        weights: ws,
        symbols: syms,
        strike,
        maturity_years: maturity,
        risk_free_rate: riskFreeRate,
        volatilities: vs,
        correlation_matrix: corr,
        initial_spots: sp,
        n_simulations: nSim,
        rebalancing_frequency: freq,
        data_source: "simulated",
        n_paths: nPaths,
      });
      setResult(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  // Build chart data: one line per hedging path
  let chartData: { date: string; [key: string]: string | number }[] = [];
  const chartSeries: { key: string; name: string; color?: string }[] = [];

  if (result?.paths?.length) {
    const dates = Object.keys(result.paths[0].portfolio_values ?? {});
    chartData = dates.map((date) => {
      const row: { date: string; [key: string]: string | number } = { date };
      result.paths.forEach((path, i) => {
        row[`Path ${i + 1}`] = (path.portfolio_values as Record<string, number>)[date] ?? 0;
      });
      return row;
    });
    result.paths.forEach((_, i) => {
      chartSeries.push({
        key: `Path ${i + 1}`,
        name: `Path ${i + 1}`,
        color: CHART_COLORS[i % CHART_COLORS.length],
      });
    });
  }

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="card space-y-5">
        <h2 className="text-prose font-semibold text-lg">Delta Hedging Simulator</h2>
        <p className="text-muted text-sm">
          Simulates delta hedging of a basket call option across multiple GBM paths. Weights and
          assets define the basket; correlation is assumed at 0.5 between all pairs.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="label">Assets (comma-separated)</label>
            <input className="input w-full" value={symbols}
              onChange={(e) => setSymbols(e.target.value)} placeholder="AAPL, MSFT" />
          </div>
          <div>
            <label className="label">Basket Weights (must sum to 1)</label>
            <input className="input w-full" value={weights}
              onChange={(e) => setWeights(e.target.value)} placeholder="0.5, 0.5" />
          </div>
          <div>
            <label className="label">Initial Spot Prices</label>
            <input className="input w-full" value={spots}
              onChange={(e) => setSpots(e.target.value)} placeholder="150, 300" />
          </div>
          <div>
            <label className="label">Volatilities (annual)</label>
            <input className="input w-full" value={vols}
              onChange={(e) => setVols(e.target.value)} placeholder="0.25, 0.22" />
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div>
            <label className="label">Strike</label>
            <input type="number" className="input w-full" value={strike}
              step={1} onChange={(e) => setStrike(Number(e.target.value))} />
          </div>
          <div>
            <label className="label">Maturity (years)</label>
            <input type="number" className="input w-full" value={maturity}
              step={0.25} min={0.25} max={5} onChange={(e) => setMaturity(Number(e.target.value))} />
          </div>
          <div>
            <label className="label">Risk-Free Rate</label>
            <input type="number" className="input w-full" value={riskFreeRate}
              step={0.01} min={0} max={0.2} onChange={(e) => setRiskFreeRate(Number(e.target.value))} />
          </div>
          <div>
            <label className="label">Rebalancing</label>
            <select className="input w-full" value={freq}
              onChange={(e) => setFreq(e.target.value as "daily" | "weekly" | "monthly")}>
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Simulation Paths</label>
            <input type="number" className="input w-full" value={nPaths}
              step={1} min={1} max={10} onChange={(e) => setNPaths(Number(e.target.value))} />
          </div>
          <div>
            <label className="label">MC Simulations (option price)</label>
            <input type="number" className="input w-full" value={nSim}
              step={1000} min={1000} max={100000} onChange={(e) => setNSim(Number(e.target.value))} />
          </div>
        </div>

        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? "Simulating…" : "Run Hedging Simulation"}
        </button>
      </form>

      {loading && <LoadingSpinner label="Running delta hedging paths…" />}

      {result && !loading && (
        <div className="space-y-4">
          {/* Summary stats */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <MetricCard
              label="Initial Option Price"
              value={`$${result.initial_option_price.toFixed(2)}`}
            />
            <MetricCard
              label="95% CI"
              value={`[$${result.initial_option_price_ci[0].toFixed(2)}, $${result.initial_option_price_ci[1].toFixed(2)}]`}
              muted
            />
            <MetricCard
              label="Avg Tracking Error"
              value={formatNumber(result.average_tracking_error)}
              delta={result.average_tracking_error != null ? -Math.abs(result.average_tracking_error) : null}
            />
          </div>

          {/* Paths chart */}
          {chartData.length > 0 && (
            <EquityCurve
              title="Delta Hedging Paths — Portfolio Value"
              data={chartData}
              series={chartSeries}
              height={340}
            />
          )}

          {/* Per-path tracking errors */}
          <div className="card">
            <h3 className="text-prose font-semibold mb-3">Per-Path Summary</h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-muted text-xs uppercase border-b border-dim">
                  <th className="text-left py-2 pr-6">Path</th>
                  <th className="text-right py-2">Tracking Error</th>
                </tr>
              </thead>
              <tbody>
                {result.paths.map((path, i) => (
                  <tr key={i} className="border-b border-dim/40">
                    <td className="py-2 pr-6" style={{ color: THEME.text }}>
                      <span
                        className="inline-block w-2.5 h-2.5 rounded-full mr-2"
                        style={{ backgroundColor: CHART_COLORS[i % CHART_COLORS.length] }}
                      />
                      Path {i + 1}
                    </td>
                    <td className="py-2 text-right font-mono text-muted">
                      {formatNumber(path.tracking_error as number)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
