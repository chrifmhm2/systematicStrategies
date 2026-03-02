import { useEffect, useState } from "react";
import { useStrategies } from "../hooks/useStrategies";
import { useBacktest } from "../hooks/useBacktest";
import { fetchAssets } from "../api/client";
import MetricCard from "../components/common/MetricCard";
import LoadingSpinner from "../components/common/LoadingSpinner";
import ErrorBanner from "../components/common/ErrorBanner";
import ParamForm from "../components/common/ParamForm";
import EquityCurve from "../components/charts/EquityCurve";
import DrawdownChart from "../components/charts/DrawdownChart";
import WeightsChart from "../components/charts/WeightsChart";
import { formatPercent, formatNumber, mergePortfolios } from "../utils/formatters";
import type { ParamSchema } from "../api/types";

const POPULAR_ASSETS = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "JPM"];
const HEDGING_FAMILIES = new Set(["hedging"]);

export default function BacktestPage() {
  const { strategies, loading: loadingStrats } = useStrategies();
  const { result, loading: running, error, submit } = useBacktest();

  const [assets, setAssets] = useState<string[]>(POPULAR_ASSETS);
  const [selectedStrategy, setSelectedStrategy] = useState("");
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(["AAPL", "MSFT", "GOOG"]);
  const [startDate, setStartDate] = useState("2022-01-03");
  const [endDate, setEndDate] = useState("2023-12-29");
  const [initialValue, setInitialValue] = useState(100_000);
  const [dataSource, setDataSource] = useState<"simulated" | "yahoo">("simulated");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [localError, setLocalError] = useState<string | null>(null);

  // Load available assets from API
  useEffect(() => {
    fetchAssets()
      .then((a) => setAssets([...new Set([...POPULAR_ASSETS, ...a])].slice(0, 30)))
      .catch(() => {});
  }, []);

  // Set first non-hedging strategy as default
  useEffect(() => {
    if (strategies.length > 0 && !selectedStrategy) {
      const first = strategies.find((s) => !HEDGING_FAMILIES.has(s.family));
      if (first) {
        setSelectedStrategy(first.id);
        initParams(first.params);
      }
    }
  }, [strategies]);

  function initParams(schema: Record<string, ParamSchema>) {
    setParams(
      Object.fromEntries(Object.entries(schema).map(([k, v]) => [k, v.default]))
    );
  }

  function handleStrategyChange(id: string) {
    setSelectedStrategy(id);
    const strat = strategies.find((s) => s.id === id);
    if (strat) initParams(strat.params);
  }

  function toggleSymbol(sym: string) {
    setSelectedSymbols((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]
    );
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLocalError(null);
    if (selectedSymbols.length === 0) {
      setLocalError("Select at least one asset.");
      return;
    }
    if (!selectedStrategy) {
      setLocalError("Select a strategy.");
      return;
    }
    await submit({
      strategy_id: selectedStrategy,
      symbols: selectedSymbols,
      start_date: startDate,
      end_date: endDate,
      initial_value: initialValue,
      params,
      data_source: dataSource,
    });
  }

  const currentStrat = strategies.find((s) => s.id === selectedStrategy);
  const m = result?.risk_metrics ?? {};

  const chartData = result
    ? mergePortfolios([{ name: "Portfolio", values: result.portfolio_values }])
    : [];

  return (
    <div className="space-y-6">
      {/* Form */}
      <form onSubmit={handleSubmit} className="card space-y-5">
        <h2 className="text-prose font-semibold text-lg">Backtest Configuration</h2>

        {/* Strategy + Data Source */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="label">Strategy</label>
            {loadingStrats ? (
              <div className="input w-full text-muted">Loading…</div>
            ) : (
              <select
                className="input w-full"
                value={selectedStrategy}
                onChange={(e) => handleStrategyChange(e.target.value)}
              >
                {["allocation", "signal", "hedging"].map((fam) => (
                  <optgroup key={fam} label={fam.charAt(0).toUpperCase() + fam.slice(1)}>
                    {strategies
                      .filter((s) => s.family === fam)
                      .map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.id}
                        </option>
                      ))}
                  </optgroup>
                ))}
              </select>
            )}
            {currentStrat && (
              <p className="text-muted text-xs mt-1">{currentStrat.description}</p>
            )}
          </div>

          <div>
            <label className="label">Data Source</label>
            <div className="flex gap-2">
              {(["simulated", "yahoo"] as const).map((src) => (
                <button
                  key={src}
                  type="button"
                  onClick={() => setDataSource(src)}
                  className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors border ${
                    dataSource === src
                      ? "bg-accent/10 border-accent text-accent"
                      : "bg-elevated border-dim text-muted hover:text-prose"
                  }`}
                >
                  {src.charAt(0).toUpperCase() + src.slice(1)}
                </button>
              ))}
            </div>
            {dataSource === "yahoo" && (
              <p className="text-warn text-xs mt-1">Requires internet. May be slow.</p>
            )}
          </div>
        </div>

        {/* Dates + Initial Value */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="label">Start Date</label>
            <input
              type="date"
              className="input w-full"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          <div>
            <label className="label">End Date</label>
            <input
              type="date"
              className="input w-full"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
          <div>
            <label className="label">Initial Value ($)</label>
            <input
              type="number"
              className="input w-full"
              value={initialValue}
              min={1000}
              step={1000}
              onChange={(e) => setInitialValue(Number(e.target.value))}
            />
          </div>
        </div>

        {/* Asset selector */}
        <div>
          <label className="label">Assets ({selectedSymbols.length} selected)</label>
          <div className="flex flex-wrap gap-2 mt-1">
            {assets.map((sym) => (
              <button
                key={sym}
                type="button"
                onClick={() => toggleSymbol(sym)}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-colors border ${
                  selectedSymbols.includes(sym)
                    ? "bg-accent/10 border-accent text-accent"
                    : "bg-elevated border-dim text-muted hover:text-prose"
                }`}
              >
                {sym}
              </button>
            ))}
          </div>
        </div>

        {/* Strategy params */}
        {currentStrat && Object.keys(currentStrat.params).length > 0 && (
          <div>
            <label className="label">Strategy Parameters</label>
            <div className="bg-elevated rounded-lg p-4 mt-1">
              <ParamForm
                schema={currentStrat.params}
                values={params}
                onChange={(k, v) => setParams((prev) => ({ ...prev, [k]: v }))}
              />
            </div>
          </div>
        )}

        {(localError || error) && (
          <ErrorBanner
            message={localError ?? error ?? ""}
            onDismiss={() => setLocalError(null)}
          />
        )}

        <button type="submit" disabled={running} className="btn-primary w-full sm:w-auto">
          {running ? "Running…" : "Run Backtest"}
        </button>
      </form>

      {/* Loading */}
      {running && <LoadingSpinner label="Running backtest…" />}

      {/* Results */}
      {result && !running && (
        <div className="space-y-4">
          {/* Metric cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <MetricCard
              label="Total Return"
              value={formatPercent(m.total_return)}
              delta={m.total_return}
            />
            <MetricCard
              label="Annualised Return"
              value={formatPercent(m.annualized_return)}
              delta={m.annualized_return}
            />
            <MetricCard
              label="Sharpe Ratio"
              value={formatNumber(m.sharpe_ratio)}
              delta={m.sharpe_ratio}
            />
            <MetricCard
              label="Max Drawdown"
              value={formatPercent(m.max_drawdown)}
              delta={m.max_drawdown}
            />
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <MetricCard label="Volatility" value={formatPercent(m.annualized_volatility)} muted />
            <MetricCard label="Sortino" value={formatNumber(m.sortino_ratio)} delta={m.sortino_ratio} />
            <MetricCard label="Win Rate" value={formatPercent(m.win_rate)} muted />
            <MetricCard
              label="Compute Time"
              value={`${(result.computation_time_ms ?? 0).toFixed(0)}`}
              unit="ms"
              muted
            />
          </div>

          {/* Charts */}
          <EquityCurve
            title={`Equity Curve — ${result.strategy_name}`}
            data={chartData}
            series={[{ key: "Portfolio", name: "Portfolio" }]}
            initialValue={initialValue}
          />

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <DrawdownChart portfolioValues={result.portfolio_values} />
            <WeightsChart weightsHistory={result.weights_history} />
          </div>

          {/* Trades log */}
          {result.trades_log.length > 0 && (
            <div className="card">
              <h3 className="text-prose font-semibold mb-3">
                Trades Log ({result.trades_log.length} trades)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-muted text-xs uppercase border-b border-dim">
                      <th className="text-left py-2 pr-4">Date</th>
                      <th className="text-left py-2 pr-4">Symbol</th>
                      <th className="text-left py-2 pr-4">Direction</th>
                      <th className="text-right py-2 pr-4">Shares</th>
                      <th className="text-right py-2">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.trades_log.slice(0, 50).map((t, i) => (
                      <tr key={i} className="border-b border-dim/40 hover:bg-elevated">
                        <td className="py-1.5 pr-4 text-muted font-mono text-xs">{String(t.date)}</td>
                        <td className="py-1.5 pr-4 font-medium">{String(t.symbol)}</td>
                        <td
                          className={`py-1.5 pr-4 text-xs ${
                            String(t.direction) === "BUY" ? "text-gain" : "text-loss"
                          }`}
                        >
                          {String(t.direction)}
                        </td>
                        <td className="py-1.5 pr-4 text-right font-mono">
                          {Number(t.shares).toFixed(2)}
                        </td>
                        <td className="py-1.5 text-right font-mono">
                          ${Number(t.value).toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {result.trades_log.length > 50 && (
                  <p className="text-muted text-xs mt-2">
                    Showing 50 of {result.trades_log.length} trades.
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
