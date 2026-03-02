import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useStrategies } from "../hooks/useStrategies";
import { compareStrategies } from "../api/client";
import type { BacktestResponse } from "../api/types";
import EquityCurve from "../components/charts/EquityCurve";
import LoadingSpinner from "../components/common/LoadingSpinner";
import { mergePortfolios } from "../utils/formatters";
import { CHART_COLORS } from "../utils/colors";

const DEMO_STRATEGIES = [
  { strategy_id: "equal_weight", params: {} },
  { strategy_id: "max_sharpe", params: {} },
  { strategy_id: "momentum", params: { lookback_period: 60 } },
];
const DEMO_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN"];

const TECH_BADGES = [
  "Python 3.12",
  "FastAPI",
  "NumPy · SciPy · pandas",
  "React 18",
  "TypeScript",
  "Recharts",
];

interface Capability {
  icon: string;
  title: string;
  desc: string;
  route: string;
  textClass: string;
  badgeClass: string;
}

const CAPABILITIES: Capability[] = [
  {
    icon: "◈",
    title: "Backtesting Engine",
    desc: "Event-driven loop with no look-ahead bias. Transaction costs, rebalancing oracles, and Yahoo Finance data in one pipeline.",
    route: "/backtest",
    textClass: "text-accent",
    badgeClass: "bg-accent/10 border-accent/20",
  },
  {
    icon: "⇌",
    title: "Strategy Comparison",
    desc: "Run 2–4 strategies head-to-head on the same universe. Overlaid equity curves and a side-by-side metric table.",
    route: "/compare",
    textClass: "text-gain",
    badgeClass: "bg-gain/10 border-gain/20",
  },
  {
    icon: "◉",
    title: "Risk Analytics",
    desc: "Sharpe, Sortino, Calmar, VaR 95%, CVaR, max drawdown, Beta, Alpha — computed on-the-fly from any backtest.",
    route: "/risk",
    textClass: "text-warn",
    badgeClass: "bg-warn/10 border-warn/20",
  },
  {
    icon: "△",
    title: "Delta Hedging Simulator",
    desc: "Replicate basket option payoffs with delta (or delta-gamma) hedging across multiple Monte Carlo paths.",
    route: "/hedging",
    textClass: "text-purple",
    badgeClass: "bg-purple/10 border-purple/20",
  },
  {
    icon: "◇",
    title: "Option Pricing",
    desc: "Price vanilla and basket options via Black-Scholes or Monte Carlo. Full Greeks grid + strike sweep charts.",
    route: "/pricing",
    textClass: "text-cyan",
    badgeClass: "bg-cyan/10 border-cyan/20",
  },
  {
    icon: "⊕",
    title: "Plugin Architecture",
    desc: "Register new strategies with a single decorator. The UI auto-discovers params and renders dynamic forms.",
    route: "/backtest",
    textClass: "text-prose",
    badgeClass: "bg-elevated border-dim",
  },
];

export default function HomePage() {
  const { strategies, loading: strategiesLoading } = useStrategies();
  const navigate = useNavigate();
  const [preview, setPreview] = useState<BacktestResponse[] | null>(null);
  const [previewLoading, setPreviewLoading] = useState(true);
  const [previewError, setPreviewError] = useState<string | null>(null);

  useEffect(() => {
    const now = new Date();
    const end = now.toISOString().slice(0, 10);
    const startDate = new Date(now);
    startDate.setFullYear(startDate.getFullYear() - 2);
    const start = startDate.toISOString().slice(0, 10);

    compareStrategies({
      strategies: DEMO_STRATEGIES,
      symbols: DEMO_SYMBOLS,
      start_date: start,
      end_date: end,
      initial_value: 100_000,
      data_source: "yahoo",
    })
      .then(setPreview)
      .catch((e: Error) => setPreviewError(e.message ?? "Backend not available"))
      .finally(() => setPreviewLoading(false));
  }, []);

  const byFamily = strategies.reduce<Record<string, typeof strategies>>((acc, s) => {
    (acc[s.family] ??= []).push(s);
    return acc;
  }, {});

  const previewChartData =
    preview && preview.length > 0
      ? mergePortfolios(
          preview.map((r) => ({ name: r.strategy_name, values: r.portfolio_values }))
        )
      : [];

  const previewSeries =
    preview?.map((r, i) => ({
      key: r.strategy_name,
      name: r.strategy_name,
      color: CHART_COLORS[i % CHART_COLORS.length],
    })) ?? [];

  return (
    <div className="max-w-5xl space-y-14">
      {/* ── Hero ──────────────────────────────────────────────────────── */}
      <div className="space-y-5 pt-4">
        <div className="flex items-center gap-2 text-accent font-semibold text-xs uppercase tracking-widest">
          <span className="w-2 h-2 rounded-full bg-gain animate-pulse" />
          QuantForge · Quantitative Research Platform
        </div>

        <h1 className="text-5xl font-bold text-prose leading-tight">
          Backtest. Analyse.
          <br />
          <span className="text-accent">Price. Hedge.</span>
        </h1>

        <p className="text-muted text-base max-w-2xl leading-relaxed">
          QuantForge is a full-stack quantitative finance platform built from scratch in{" "}
          <span className="text-prose font-medium">Python + React</span>. It covers the full quant
          research workflow — from strategy backtesting on live Yahoo Finance data to risk
          attribution, option pricing, and dynamic hedging simulation — all in one self-contained
          environment.
        </p>

        {/* Tech badges */}
        <div className="flex flex-wrap gap-2 pt-1">
          {TECH_BADGES.map((b) => (
            <span
              key={b}
              className="text-xs px-2.5 py-1 rounded-full bg-elevated border border-dim text-muted font-mono"
            >
              {b}
            </span>
          ))}
        </div>

        <div className="flex flex-wrap gap-3 pt-2">
          <button className="btn-primary" onClick={() => navigate("/backtest")}>
            Run a Backtest →
          </button>
          <button className="btn-secondary" onClick={() => navigate("/compare")}>
            Compare Strategies
          </button>
          <button className="btn-secondary" onClick={() => navigate("/pricing")}>
            Price an Option
          </button>
        </div>
      </div>

      {/* ── Live Strategy Preview ──────────────────────────────────────── */}
      <div className="space-y-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-prose font-semibold text-lg">Live Strategy Preview</h2>
            <p className="text-muted text-sm mt-0.5">
              Equal Weight · Max Sharpe · Momentum on{" "}
              <span className="font-mono text-prose">AAPL MSFT GOOGL AMZN</span> — past 2 years
            </p>
          </div>
          {!previewLoading && preview && (
            <button
              className="text-xs text-accent hover:underline flex-shrink-0 mt-1"
              onClick={() => navigate("/compare")}
            >
              Open in Comparison →
            </button>
          )}
        </div>

        {previewLoading ? (
          <div className="card flex items-center justify-center h-60">
            <LoadingSpinner label="Fetching live market data…" />
          </div>
        ) : previewError ? (
          <div className="card flex flex-col items-center justify-center h-44 gap-3 text-center">
            <p className="text-warn text-sm font-medium">Backend not reachable</p>
            <p className="text-muted text-xs leading-relaxed max-w-sm">
              Start the FastAPI server to see live equity curves here.
            </p>
            <code className="font-mono text-xs text-prose bg-elevated border border-dim px-3 py-1.5 rounded">
              uvicorn main:app --reload --port 8000
            </code>
          </div>
        ) : preview && preview.length > 0 ? (
          <div className="space-y-3">
            <EquityCurve
              data={previewChartData}
              series={previewSeries}
              height={290}
              initialValue={100_000}
            />
            {/* Per-strategy mini metrics */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {preview.map((r, i) => (
                <div key={r.strategy_name} className="card space-y-3">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ background: CHART_COLORS[i % CHART_COLORS.length] }}
                    />
                    <span className="text-prose text-sm font-semibold font-mono truncate">
                      {r.strategy_name}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-1">
                    <MiniMetric
                      label="Return"
                      value={
                        r.risk_metrics.total_return != null
                          ? `${(r.risk_metrics.total_return * 100).toFixed(1)}%`
                          : "—"
                      }
                      positive={(r.risk_metrics.total_return ?? 0) >= 0}
                    />
                    <MiniMetric
                      label="Sharpe"
                      value={r.risk_metrics.sharpe_ratio?.toFixed(2) ?? "—"}
                      positive={(r.risk_metrics.sharpe_ratio ?? 0) >= 1}
                    />
                    <MiniMetric
                      label="Max DD"
                      value={
                        r.risk_metrics.max_drawdown != null
                          ? `${(r.risk_metrics.max_drawdown * 100).toFixed(1)}%`
                          : "—"
                      }
                      positive={false}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>

      {/* ── Platform Capabilities ──────────────────────────────────────── */}
      <div className="space-y-4">
        <h2 className="text-prose font-semibold text-lg">Platform Capabilities</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {CAPABILITIES.map((c) => (
            <div
              key={c.title}
              className={`card border cursor-pointer hover:brightness-110 transition-all duration-150 ${c.badgeClass}`}
              onClick={() => navigate(c.route)}
            >
              <div className="flex items-start gap-3">
                <span className={`text-xl mt-0.5 flex-shrink-0 ${c.textClass}`}>{c.icon}</span>
                <div className="space-y-1.5">
                  <h3 className={`font-semibold text-sm ${c.textClass}`}>{c.title}</h3>
                  <p className="text-muted text-xs leading-relaxed">{c.desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Strategy Directory ────────────────────────────────────────── */}
      {!strategiesLoading && Object.keys(byFamily).length > 0 && (
        <div className="space-y-4">
          <h2 className="text-prose font-semibold text-lg">
            Strategy Directory
            <span className="ml-2 text-xs text-muted font-normal font-mono">
              {strategies.length} registered
            </span>
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {(["allocation", "signal", "hedging"] as const).map((family) => {
              const strats = byFamily[family];
              if (!strats?.length) return null;
              return (
                <div key={family} className="card space-y-4">
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs px-2.5 py-0.5 rounded-full font-medium ${
                        family === "allocation"
                          ? "bg-accent/20 text-accent"
                          : family === "signal"
                          ? "bg-gain/20 text-gain"
                          : "bg-purple/20 text-purple"
                      }`}
                    >
                      {family}
                    </span>
                    <span className="text-muted text-xs">{strats.length} strategies</span>
                  </div>
                  <ul className="space-y-3">
                    {strats.map((s) => (
                      <li
                        key={s.id}
                        className="cursor-pointer group"
                        onClick={() => navigate("/backtest")}
                      >
                        <span className="text-xs font-medium font-mono text-prose group-hover:text-accent transition-colors">
                          {s.id}
                        </span>
                        <p className="text-muted text-xs mt-0.5 leading-relaxed">
                          {s.description}
                        </p>
                      </li>
                    ))}
                  </ul>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Architecture ──────────────────────────────────────────────── */}
      <div className="card border border-dim space-y-3">
        <p className="text-prose text-sm font-semibold">Architecture</p>
        <p className="text-muted text-xs leading-relaxed">
          The quant engine (<code className="font-mono text-prose">backend/core/</code>) is a pure
          Python library with zero web dependencies — usable standalone in a Jupyter notebook or
          CLI. A <span className="text-prose">FastAPI</span> layer exposes it over HTTP, and this
          React app consumes it via a typed Axios client. A Vite dev proxy routes{" "}
          <code className="font-mono text-prose">/api</code> to{" "}
          <code className="font-mono text-prose">:8000</code>, so there is no CORS configuration.
          Strategies are plugins: adding one requires only a decorated Python class;{" "}
          <code className="font-mono text-prose">param_schema</code> is returned by the API and
          rendered dynamically by the UI.
        </p>
        <div className="flex flex-wrap gap-x-6 gap-y-1.5 font-mono text-xs text-muted pt-1">
          <span>
            <span className="text-accent">→</span> backend/core/ — pure Python quant engine
          </span>
          <span>
            <span className="text-accent">→</span> FastAPI REST API on :8000
          </span>
          <span>
            <span className="text-accent">→</span> React / Vite SPA on :5173
          </span>
          <span>
            <span className="text-accent">→</span> /api proxy — no CORS needed
          </span>
        </div>
      </div>
    </div>
  );
}

function MiniMetric({
  label,
  value,
  positive,
}: {
  label: string;
  value: string;
  positive: boolean;
}) {
  return (
    <div className="text-center">
      <p className={`font-mono font-bold text-sm ${positive ? "text-gain" : "text-loss"}`}>
        {value}
      </p>
      <p className="text-muted text-xs mt-0.5">{label}</p>
    </div>
  );
}
