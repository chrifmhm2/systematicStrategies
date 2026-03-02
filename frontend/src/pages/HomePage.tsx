import { useNavigate } from "react-router-dom";
import { useStrategies } from "../hooks/useStrategies";
import LoadingSpinner from "../components/common/LoadingSpinner";

const FEATURES = [
  {
    icon: "◎",
    title: "8 Strategies",
    body: "Allocation, signal, and hedging strategies — from Equal Weight to Delta-Gamma hedging.",
  },
  {
    icon: "△",
    title: "Risk Analytics",
    body: "Sharpe, Sortino, VaR 95%, CVaR, rolling metrics — all computed on-the-fly.",
  },
  {
    icon: "◇",
    title: "Option Pricing",
    body: "Black-Scholes and Monte Carlo basket pricing with full Greeks (Δ, Γ, ν, θ, ρ).",
  },
];

export default function HomePage() {
  const { strategies, loading } = useStrategies();
  const navigate = useNavigate();

  const byFamily = strategies.reduce<Record<string, number>>((acc, s) => {
    acc[s.family] = (acc[s.family] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="max-w-4xl space-y-10">
      {/* Hero */}
      <div className="space-y-4 pt-4">
        <div className="flex items-center gap-2 text-accent font-semibold text-sm uppercase tracking-widest">
          <span>◆</span> QuantForge
        </div>
        <h2 className="text-4xl font-bold text-prose leading-tight">
          Quantitative Strategies
          <br />
          <span className="text-accent">Dashboard</span>
        </h2>
        <p className="text-muted text-lg max-w-2xl">
          A full-stack backtesting platform built in Python + React. Run strategies on real or
          simulated market data, analyze risk, price options, and simulate delta hedging — all
          from one interface.
        </p>
        <div className="flex gap-3 pt-2">
          <button className="btn-primary" onClick={() => navigate("/backtest")}>
            Run a Backtest →
          </button>
          <button className="btn-secondary" onClick={() => navigate("/compare")}>
            Compare Strategies
          </button>
        </div>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {loading ? (
          <LoadingSpinner label="Loading stats…" />
        ) : (
          <>
            <StatBox value={String(strategies.length)} label="Strategies" />
            <StatBox value={String(byFamily["allocation"] ?? 0)} label="Allocation" />
            <StatBox value={String(byFamily["signal"] ?? 0)} label="Signal" />
            <StatBox value={String(byFamily["hedging"] ?? 0)} label="Hedging" />
          </>
        )}
      </div>

      {/* Feature cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {FEATURES.map((f) => (
          <div key={f.title} className="card space-y-2">
            <span className="text-accent text-2xl">{f.icon}</span>
            <h3 className="text-prose font-semibold">{f.title}</h3>
            <p className="text-muted text-sm">{f.body}</p>
          </div>
        ))}
      </div>

      {/* Registered strategies list */}
      {!loading && strategies.length > 0 && (
        <div className="card space-y-3">
          <h3 className="text-prose font-semibold">Registered Strategies</h3>
          <div className="space-y-2">
            {strategies.map((s) => (
              <div
                key={s.id}
                className="flex items-start gap-3 p-3 bg-elevated rounded-lg cursor-pointer hover:bg-dim transition-colors"
                onClick={() => navigate("/backtest")}
              >
                <span
                  className={`text-xs px-2 py-0.5 rounded-full font-medium mt-0.5 ${
                    s.family === "allocation"
                      ? "bg-accent/20 text-accent"
                      : s.family === "signal"
                      ? "bg-gain/20 text-gain"
                      : "bg-purple/20 text-purple"
                  }`}
                >
                  {s.family}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-prose text-sm font-medium">{s.id}</p>
                  <p className="text-muted text-xs truncate">{s.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StatBox({ value, label }: { value: string; label: string }) {
  return (
    <div className="card text-center">
      <p className="text-3xl font-bold font-mono text-accent">{value}</p>
      <p className="text-muted text-sm mt-1">{label}</p>
    </div>
  );
}
