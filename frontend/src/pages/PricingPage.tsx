import { useState } from "react";
import { priceOption } from "../api/client";
import type { OptionPricingResponse } from "../api/types";
import MetricCard from "../components/common/MetricCard";
import LoadingSpinner from "../components/common/LoadingSpinner";
import ErrorBanner from "../components/common/ErrorBanner";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { THEME, CHART_COLORS } from "../utils/colors";

export default function PricingPage() {
  // Single option inputs
  const [optionType, setOptionType] = useState<"call" | "put">("call");
  const [S, setS] = useState(100);
  const [K, setK] = useState(100);
  const [T, setT] = useState(1.0);
  const [r, setR] = useState(0.05);
  const [sigma, setSigma] = useState(0.2);
  const [method, setMethod] = useState<"bs" | "mc">("bs");
  const [nSim, setNSim] = useState(50_000);

  const [result, setResult] = useState<OptionPricingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Strike sweep state
  const [sweepData, setSweepData] = useState<
    { K: number; call: number; put: number; callDelta: number; putDelta: number }[]
  >([]);
  const [sweeping, setSweeping] = useState(false);

  async function handlePrice(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await priceOption({ option_type: optionType, S, K, T, r, sigma, method, n_simulations: nSim });
      setResult(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function runStrikeSweep() {
    setSweeping(true);
    setError(null);
    const strikes = Array.from({ length: 21 }, (_, i) => Math.round(S * (0.7 + i * 0.03)));
    const rows: typeof sweepData = [];
    try {
      for (const k of strikes) {
        const [callRes, putRes] = await Promise.all([
          priceOption({ option_type: "call", S, K: k, T, r, sigma, method: "bs" }),
          priceOption({ option_type: "put",  S, K: k, T, r, sigma, method: "bs" }),
        ]);
        rows.push({
          K: k,
          call: Math.round(callRes.price * 100) / 100,
          put:  Math.round(putRes.price * 100) / 100,
          callDelta: Math.round((callRes.greeks.delta as number) * 1000) / 1000,
          putDelta:  Math.round((putRes.greeks.delta as number) * 1000) / 1000,
        });
      }
      setSweepData(rows);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSweeping(false);
    }
  }

  const greeks: OptionPricingResponse["greeks"] | null = result?.greeks ?? null;

  return (
    <div className="space-y-6">
      {/* Form */}
      <form onSubmit={handlePrice} className="card space-y-5">
        <h2 className="text-prose font-semibold text-lg">Option Pricing</h2>

        {/* Option type */}
        <div className="flex gap-2">
          {(["call", "put"] as const).map((t) => (
            <button key={t} type="button" onClick={() => setOptionType(t)}
              className={`flex-1 py-2 rounded-lg text-sm font-semibold border transition-colors ${
                optionType === t
                  ? t === "call"
                    ? "bg-gain/10 border-gain text-gain"
                    : "bg-loss/10 border-loss text-loss"
                  : "bg-elevated border-dim text-muted hover:text-prose"
              }`}>
              {t.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Inputs */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {[
            { label: "Spot (S)", value: S, setter: setS, step: 1 },
            { label: "Strike (K)", value: K, setter: setK, step: 1 },
            { label: "Maturity T (years)", value: T, setter: setT, step: 0.25, min: 0.01 },
            { label: "Risk-Free Rate (r)", value: r, setter: setR, step: 0.01 },
            { label: "Volatility (σ)", value: sigma, setter: setSigma, step: 0.01, min: 0.01 },
          ].map(({ label, value, setter, step, min }) => (
            <div key={label}>
              <label className="label">{label}</label>
              <input type="number" className="input w-full" value={value}
                step={step} min={min ?? undefined}
                onChange={(e) => setter(Number(e.target.value))} />
            </div>
          ))}

          <div>
            <label className="label">Method</label>
            <div className="flex gap-2 mt-1">
              {(["bs", "mc"] as const).map((m) => (
                <button key={m} type="button" onClick={() => setMethod(m)}
                  className={`flex-1 py-2 rounded-lg text-sm font-medium border transition-colors ${
                    method === m
                      ? "bg-accent/10 border-accent text-accent"
                      : "bg-elevated border-dim text-muted hover:text-prose"
                  }`}>
                  {m === "bs" ? "Black-Scholes" : "Monte Carlo"}
                </button>
              ))}
            </div>
          </div>
        </div>

        {method === "mc" && (
          <div>
            <label className="label">MC Simulations</label>
            <input type="number" className="input w-40" value={nSim}
              step={10000} min={1000} onChange={(e) => setNSim(Number(e.target.value))} />
          </div>
        )}

        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

        <div className="flex gap-3 flex-wrap">
          <button type="submit" disabled={loading} className="btn-primary">
            {loading ? "Pricing…" : "Price Option"}
          </button>
          <button type="button" onClick={runStrikeSweep} disabled={sweeping}
            className="btn-secondary">
            {sweeping ? "Sweeping…" : "Strike Sweep (BS)"}
          </button>
        </div>
      </form>

      {loading && <LoadingSpinner label="Computing option price…" />}

      {/* Single result */}
      {result && !loading && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <MetricCard label="Price" value={`$${result.price.toFixed(4)}`} />
            {result.std_error != null && (
              <MetricCard label="Std Error" value={`$${result.std_error.toFixed(4)}`} muted />
            )}
            {result.confidence_interval != null && (
              <MetricCard
                label="95% CI"
                value={`[$${result.confidence_interval[0].toFixed(2)}, $${result.confidence_interval[1].toFixed(2)}]`}
                muted
              />
            )}
          </div>

          {/* Greeks */}
          {greeks != null && (
            <div className="card">
              <h3 className="text-prose font-semibold mb-3">Greeks</h3>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
                {(["delta", "gamma", "vega", "theta", "rho"] as const).map((g) => {
                  const v = greeks[g];
                  if (v == null) return null;
                  const num = typeof v === "number" ? v : (v as number[])[0];
                  return (
                    <div key={g} className="bg-elevated rounded-lg p-3 text-center">
                      <p className="text-muted text-xs uppercase tracking-wider mb-1">{g}</p>
                      <p className="text-prose font-mono text-lg font-semibold">
                        {num.toFixed(4)}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Strike sweep charts */}
      {sweeping && <LoadingSpinner label="Sweeping strikes…" />}

      {sweepData.length > 0 && !sweeping && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="card">
            <h3 className="text-prose font-semibold mb-4">
              Call / Put Price vs Strike (S={S}, σ={sigma}, T={T})
            </h3>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={sweepData} margin={{ top: 4, right: 12, left: 8, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={THEME.border} />
                <XAxis dataKey="K" tick={{ fill: THEME.muted, fontSize: 11 }}
                  label={{ value: "Strike (K)", position: "insideBottom", offset: -2, fill: THEME.muted, fontSize: 11 }} />
                <YAxis tick={{ fill: THEME.muted, fontSize: 11 }} width={44}
                  tickFormatter={(v) => `$${v}`} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: THEME.card, border: `1px solid ${THEME.border}`, borderRadius: 8, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="call" stroke={CHART_COLORS[1]} dot={false} strokeWidth={2} name="Call" />
                <Line type="monotone" dataKey="put"  stroke={CHART_COLORS[3]} dot={false} strokeWidth={2} name="Put" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h3 className="text-prose font-semibold mb-4">Delta vs Strike</h3>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={sweepData} margin={{ top: 4, right: 12, left: 8, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={THEME.border} />
                <XAxis dataKey="K" tick={{ fill: THEME.muted, fontSize: 11 }}
                  label={{ value: "Strike (K)", position: "insideBottom", offset: -2, fill: THEME.muted, fontSize: 11 }} />
                <YAxis tick={{ fill: THEME.muted, fontSize: 11 }} width={44}
                  domain={[-1, 1]} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: THEME.card, border: `1px solid ${THEME.border}`, borderRadius: 8, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="callDelta" stroke={CHART_COLORS[1]} dot={false} strokeWidth={2} name="Call Δ" />
                <Line type="monotone" dataKey="putDelta"  stroke={CHART_COLORS[3]} dot={false} strokeWidth={2} name="Put Δ" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
