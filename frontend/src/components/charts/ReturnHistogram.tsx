import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { THEME } from "../../utils/colors";

interface Props {
  portfolioValues: Record<string, number>;
  var95?: number | null;
  cvar95?: number | null;
  title?: string;
  height?: number;
}

function buildHistogram(
  portfolioValues: Record<string, number>,
  bins = 40
): { bin: string; count: number; midpoint: number }[] {
  const vals = Object.values(portfolioValues);
  const returns: number[] = [];
  for (let i = 1; i < vals.length; i++) {
    if (vals[i - 1] !== 0) {
      returns.push(((vals[i] - vals[i - 1]) / vals[i - 1]) * 100);
    }
  }
  if (returns.length === 0) return [];

  const min = Math.min(...returns);
  const max = Math.max(...returns);
  const binWidth = (max - min) / bins;

  const counts = new Array(bins).fill(0);
  for (const r of returns) {
    const idx = Math.min(Math.floor((r - min) / binWidth), bins - 1);
    counts[idx]++;
  }

  return counts.map((count, i) => {
    const mid = min + (i + 0.5) * binWidth;
    return { bin: `${mid.toFixed(1)}%`, count, midpoint: mid };
  });
}

export default function ReturnHistogram({
  portfolioValues,
  var95,
  cvar95,
  title = "Daily Return Distribution",
  height = 260,
}: Props) {
  const data = buildHistogram(portfolioValues);

  return (
    <div className="card">
      {title && <h3 className="text-prose font-semibold mb-4">{title}</h3>}
      {var95 != null && (
        <div className="flex gap-4 mb-3 text-xs">
          <span className="text-loss">
            VaR 95%: <span className="font-mono">{(var95 * 100).toFixed(2)}%</span>
          </span>
          {cvar95 != null && (
            <span className="text-warn">
              CVaR 95%: <span className="font-mono">{(cvar95 * 100).toFixed(2)}%</span>
            </span>
          )}
        </div>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 4, right: 12, left: 8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={THEME.border} vertical={false} />
          <XAxis
            dataKey="bin"
            tick={{ fill: THEME.muted, fontSize: 10 }}
            tickLine={false}
            interval={Math.floor(data.length / 6)}
          />
          <YAxis
            tick={{ fill: THEME.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={36}
          />
          <Tooltip
            contentStyle={{
              background: THEME.card,
              border: `1px solid ${THEME.border}`,
              borderRadius: 8,
              fontSize: 12,
            }}
            cursor={{ fill: THEME.border }}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={(v: any) => [+v || 0, "Count"]}
          />
          <Bar dataKey="count" fill={THEME.accent} fillOpacity={0.7} radius={[2, 2, 0, 0]} />
          {var95 != null && (
            <ReferenceLine
              x={`${(var95 * 100).toFixed(1)}%`}
              stroke={THEME.loss}
              strokeDasharray="4 4"
              label={{ value: "VaR 95%", fill: THEME.loss, fontSize: 10, position: "top" }}
            />
          )}
          {cvar95 != null && (
            <ReferenceLine
              x={`${(cvar95 * 100).toFixed(1)}%`}
              stroke={THEME.warn}
              strokeDasharray="4 4"
              label={{ value: "CVaR", fill: THEME.warn, fontSize: 10, position: "top" }}
            />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
