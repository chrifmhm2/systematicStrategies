import { useState } from "react";
import { useStrategies } from "../hooks/useStrategies";
import { runBacktest } from "../api/client";
import type { BacktestResponse } from "../api/types";
import MetricCard from "../components/common/MetricCard";
import LoadingSpinner from "../components/common/LoadingSpinner";
import ErrorBanner from "../components/common/ErrorBanner";
import ReturnHistogram from "../components/charts/ReturnHistogram";
import EquityCurve from "../components/charts/EquityCurve";
import { formatPercent, formatNumber, portfolioToChartData } from "../utils/formatters";
import type { ParamSchema } from "../api/types";

const POPULAR = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"];

const ALL_METRICS: { key: string; label: string; isPercent?: boolean; isMult?: boolean }[] = [
  { key: "total_return",          label: "Total Return",          isPercent: true },
  { key: "annualized_return",     label: "Annualised Return",     isPercent: true },
  { key: "annualized_volatility", label: "Annualised Volatility", isPercent: true },
  { key: "sharpe_ratio",          label: "Sharpe Ratio" },
  { key: "sortino_ratio",         label: "Sortino Ratio" },
  { key: "max_drawdown",          label: "Max Drawdown",          isPercent: true },
  { key: "calmar_ratio",          label: "Calmar Ratio" },
  { key: "win_rate",              label: "Win Rate",              isPercent: true },
  { key: "profit_factor",         label: "Profit Factor",         isMult: true },
  { key: "var_95",                label: "VaR 95% (daily)",       isPercent: true },
  { key: "cvar_95",               label: "CVaR 95% (daily)",      isPercent: true },
  { key: "tracking_error",        label: "Tracking Error",        isPercent: true },
  { key: "information_ratio",     label: "Information Ratio" },
  { key: "turnover",              label: "Avg Turnover",          isPercent: true },
];

export default function RiskPage() {
  const { strategies, loading: loadingStrats } = useStrategies();

  const [strategyId, setStrategyId] = useState("EqualWeightStrategy");
  const [symbols, setSymbols] = useState<string[]>(["AAPL", "MSFT", "GOOG"]);
  const [startDate, setStartDate] = useState("2022-01-03");
  const [endDate, setEndDate] = useState("2023-12-29");
  const [dataSource, setDataSource] = useState<"simulated" | "yahoo">("simulated");
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toggleSymbol(sym: string) {
    setSymbols((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]
    );
  }

  async function handleRun(e: React.FormEvent) {
    e.preventDefault();
    if (symbols.length === 0) { setError("Select at least one asset."); return; }
    setLoading(true);
    setError(null);
    try {
      const strat = strategies.find((s) => s.id === strategyId);
      const defaultParams = strat
        ? Object.fromEntries(Object.entries(strat.params as Record<string, ParamSchema>).map(([k, v]) => [k, v.default]))
        : {};
      const res = await runBacktest({
        strategy_id: strategyId,
        symbols,
        start_date: startDate,
        end_date: endDate,
        initial_value: 100_000,
        params: defaultParams,
        data_source: dataSource,
      });
      setResult(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const m = result?.risk_metrics ?? {};
  const chartData = result ? portfolioToChartData(result.portfolio_values, "value") : [];

  return (
    <div className="space-y-6">
      {/* Config */}
      <form onSubmit={handleRun} className="card space-y-4">
        <h2 className="text-prose font-semibold text-lg">Risk Analytics</h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="label">Strategy</label>
            <select
              className="input w-full"
              value={strategyId}
              disabled={loadingStrats}
              onChange={(e) => setStrategyId(e.target.value)}
            >
              {strategies
                .filter((s) => s.family !== "hedging")
                .map((s) => (
                  <option key={s.id} value={s.id}>{s.id}</option>
                ))}
            </select>
          </div>
          <div>
            <label className="label">Data Source</label>
            <div className="flex gap-2 mt-1">
              {(["simulated", "yahoo"] as const).map((src) => (
                <button key={src} type="button" onClick={() => setDataSource(src)}
                  className={`flex-1 py-2 rounded-lg text-sm font-medium border transition-colors ${
                    dataSource === src
                      ? "bg-accent/10 border-accent text-accent"
                      : "bg-elevated border-dim text-muted hover:text-prose"
                  }`}>
                  {src.charAt(0).toUpperCase() + src.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Start</label>
            <input type="date" className="input w-full" value={startDate}
              onChange={(e) => setStartDate(e.target.value)} />
          </div>
          <div>
            <label className="label">End</label>
            <input type="date" className="input w-full" value={endDate}
              onChange={(e) => setEndDate(e.target.value)} />
          </div>
        </div>

        <div>
          <label className="label">Assets</label>
          <div className="flex flex-wrap gap-2 mt-1">
            {POPULAR.map((sym) => (
              <button key={sym} type="button" onClick={() => toggleSymbol(sym)}
                className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                  symbols.includes(sym)
                    ? "bg-accent/10 border-accent text-accent"
                    : "bg-elevated border-dim text-muted hover:text-prose"
                }`}>
                {sym}
              </button>
            ))}
          </div>
        </div>

        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? "Analysing…" : "Analyse Risk"}
        </button>
      </form>

      {loading && <LoadingSpinner label="Analysing portfolio risk…" />}

      {result && !loading && (
        <div className="space-y-4">
          {/* Key metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <MetricCard label="Sharpe Ratio" value={formatNumber(m.sharpe_ratio)} delta={m.sharpe_ratio} />
            <MetricCard label="Sortino Ratio" value={formatNumber(m.sortino_ratio)} delta={m.sortino_ratio} />
            <MetricCard label="Max Drawdown" value={formatPercent(m.max_drawdown)} delta={m.max_drawdown} />
            <MetricCard label="VaR 95% (daily)" value={formatPercent(m.var_95)} delta={m.var_95} />
          </div>

          {/* Return distribution + equity */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <ReturnHistogram
              portfolioValues={result.portfolio_values}
              var95={m.var_95}
              cvar95={m.cvar_95}
            />
            <EquityCurve
              title="Portfolio Value"
              data={chartData}
              series={[{ key: "value", name: result.strategy_name }]}
              height={260}
            />
          </div>

          {/* Full metrics table */}
          <div className="card">
            <h3 className="text-prose font-semibold mb-4">Full Risk Report</h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-8 gap-y-2">
              {ALL_METRICS.map(({ key, label, isPercent, isMult }) => {
                const v = m[key];
                const fmt = isPercent
                  ? formatPercent(v)
                  : isMult
                  ? v != null ? `${Number(v).toFixed(2)}×` : "—"
                  : formatNumber(v);
                const cls =
                  v == null ? "text-muted" : v >= 0 ? "text-gain" : "text-loss";
                return (
                  <div key={key} className="flex justify-between items-center py-1.5 border-b border-dim/30">
                    <span className="text-muted text-sm">{label}</span>
                    <span className={`font-mono text-sm font-medium ${cls}`}>{fmt}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
