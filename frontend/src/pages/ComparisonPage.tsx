import { useState } from "react";
import { useStrategies } from "../hooks/useStrategies";
import { compareStrategies } from "../api/client";
import type { BacktestResponse, StrategySpec } from "../api/types";
import ParamForm from "../components/common/ParamForm";
import LoadingSpinner from "../components/common/LoadingSpinner";
import ErrorBanner from "../components/common/ErrorBanner";
import EquityCurve from "../components/charts/EquityCurve";
import { CHART_COLORS } from "../utils/colors";
import { mergePortfolios, formatPercent, formatNumber } from "../utils/formatters";
import type { ParamSchema } from "../api/types";

const POPULAR = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"];

const METRIC_ROWS: { key: string; label: string; isPercent?: boolean }[] = [
  { key: "total_return",          label: "Total Return",       isPercent: true },
  { key: "annualized_return",     label: "Ann. Return",        isPercent: true },
  { key: "annualized_volatility", label: "Ann. Volatility",    isPercent: true },
  { key: "sharpe_ratio",          label: "Sharpe Ratio" },
  { key: "sortino_ratio",         label: "Sortino Ratio" },
  { key: "max_drawdown",          label: "Max Drawdown",       isPercent: true },
  { key: "calmar_ratio",          label: "Calmar Ratio" },
  { key: "win_rate",              label: "Win Rate",           isPercent: true },
  { key: "var_95",                label: "VaR 95%",            isPercent: true },
  { key: "cvar_95",               label: "CVaR 95%",           isPercent: true },
];

interface StrategySlot {
  id: string;
  params: Record<string, unknown>;
}

export default function ComparisonPage() {
  const { strategies, loading: loadingStrats } = useStrategies();

  const [slots, setSlots] = useState<StrategySlot[]>([
    { id: "", params: {} },
    { id: "", params: {} },
  ]);
  const [symbols, setSymbols] = useState<string[]>(["AAPL", "MSFT", "GOOG"]);
  const [startDate, setStartDate] = useState("2022-01-03");
  const [endDate, setEndDate] = useState("2023-12-29");
  const [initialValue, setInitialValue] = useState(100_000);
  const [dataSource, setDataSource] = useState<"simulated" | "yahoo">("simulated");
  const [results, setResults] = useState<BacktestResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const nonHedgingStrategies = strategies.filter((s) => s.family !== "hedging");

  // Initialise slot params from schema defaults
  function setSlotStrategy(idx: number, stratId: string) {
    const strat = strategies.find((s) => s.id === stratId);
    const defaultParams = strat
      ? Object.fromEntries(Object.entries(strat.params as Record<string, ParamSchema>).map(([k, v]) => [k, v.default]))
      : {};
    setSlots((prev) => {
      const next = [...prev];
      next[idx] = { id: stratId, params: defaultParams };
      return next;
    });
  }

  function setSlotParam(idx: number, key: string, value: unknown) {
    setSlots((prev) => {
      const next = [...prev];
      next[idx] = { ...next[idx], params: { ...next[idx].params, [key]: value } };
      return next;
    });
  }

  function toggleSymbol(sym: string) {
    setSymbols((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]
    );
  }

  async function handleCompare(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const validSlots = slots.filter((s) => s.id);
    if (validSlots.length < 2) {
      setError("Select at least 2 strategies.");
      return;
    }
    if (symbols.length === 0) {
      setError("Select at least one asset.");
      return;
    }
    setLoading(true);
    try {
      const specs: StrategySpec[] = validSlots.map((s) => ({
        strategy_id: s.id,
        params: s.params,
      }));
      const res = await compareStrategies({
        strategies: specs,
        symbols,
        start_date: startDate,
        end_date: endDate,
        initial_value: initialValue,
        data_source: dataSource,
      });
      setResults(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  // Build chart data merging all strategy portfolios
  const strategyNames = results.map((r) => r.strategy_name);
  const chartData = results.length
    ? mergePortfolios(
        results.map((r) => ({ name: r.strategy_name, values: r.portfolio_values }))
      )
    : [];
  const chartSeries = strategyNames.map((name, i) => ({
    key: name,
    name,
    color: CHART_COLORS[i % CHART_COLORS.length],
  }));

  return (
    <div className="space-y-6">
      <form onSubmit={handleCompare} className="card space-y-5">
        <h2 className="text-prose font-semibold text-lg">Strategy Comparison</h2>

        {/* Strategy slots */}
        <div className="space-y-4">
          <label className="label">Strategies (2–4)</label>
          {slots.map((slot, idx) => {
            const strat = strategies.find((s) => s.id === slot.id);
            return (
              <div key={idx} className="bg-elevated rounded-lg p-4 space-y-3">
                <div className="flex items-center gap-3">
                  <span
                    className="w-3 h-3 rounded-full flex-shrink-0"
                    style={{ backgroundColor: CHART_COLORS[idx % CHART_COLORS.length] }}
                  />
                  <select
                    className="input flex-1"
                    value={slot.id}
                    disabled={loadingStrats}
                    onChange={(e) => setSlotStrategy(idx, e.target.value)}
                  >
                    <option value="">— Select strategy —</option>
                    {nonHedgingStrategies.map((s) => (
                      <option key={s.id} value={s.id}>{s.id}</option>
                    ))}
                  </select>
                  {slots.length > 2 && (
                    <button
                      type="button"
                      onClick={() => setSlots((prev) => prev.filter((_, i) => i !== idx))}
                      className="text-muted hover:text-loss transition-colors text-lg"
                    >
                      ✕
                    </button>
                  )}
                </div>
                {strat && Object.keys(strat.params).length > 0 && (
                  <ParamForm
                    schema={strat.params}
                    values={slot.params}
                    onChange={(k, v) => setSlotParam(idx, k, v)}
                  />
                )}
              </div>
            );
          })}
          {slots.length < 4 && (
            <button
              type="button"
              onClick={() => setSlots((prev) => [...prev, { id: "", params: {} }])}
              className="btn-secondary text-sm"
            >
              + Add Strategy
            </button>
          )}
        </div>

        {/* Shared settings */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="label">Start Date</label>
            <input type="date" className="input w-full" value={startDate}
              onChange={(e) => setStartDate(e.target.value)} />
          </div>
          <div>
            <label className="label">End Date</label>
            <input type="date" className="input w-full" value={endDate}
              onChange={(e) => setEndDate(e.target.value)} />
          </div>
          <div>
            <label className="label">Initial Value ($)</label>
            <input type="number" className="input w-full" value={initialValue}
              step={1000} onChange={(e) => setInitialValue(Number(e.target.value))} />
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="label">Assets ({symbols.length} selected)</label>
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

        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

        <button type="submit" disabled={loading} className="btn-primary">
          {loading ? "Comparing…" : "Compare"}
        </button>
      </form>

      {loading && <LoadingSpinner label="Running strategies…" />}

      {results.length > 0 && !loading && (
        <div className="space-y-4">
          <EquityCurve
            title="Strategy Comparison — Equity Curves"
            data={chartData}
            series={chartSeries}
            initialValue={initialValue}
            height={340}
          />

          {/* Comparison table */}
          <div className="card overflow-x-auto">
            <h3 className="text-prose font-semibold mb-4">Risk Metrics Comparison</h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-muted text-xs uppercase border-b border-dim">
                  <th className="text-left py-2 pr-6">Metric</th>
                  {results.map((r, i) => (
                    <th
                      key={i}
                      className="text-right py-2 pr-4"
                      style={{ color: CHART_COLORS[i % CHART_COLORS.length] }}
                    >
                      {r.strategy_name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {METRIC_ROWS.map(({ key, label, isPercent }) => (
                  <tr key={key} className="border-b border-dim/40 hover:bg-elevated">
                    <td className="py-2 pr-6 text-muted">{label}</td>
                    {results.map((r, i) => {
                      const v = r.risk_metrics[key];
                      const fmt = isPercent ? formatPercent(v) : formatNumber(v);
                      const colorClass =
                        v == null
                          ? "text-muted"
                          : v >= 0
                          ? "text-gain"
                          : "text-loss";
                      return (
                        <td key={i} className={`py-2 pr-4 text-right font-mono ${colorClass}`}>
                          {fmt}
                        </td>
                      );
                    })}
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
