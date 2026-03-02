import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { THEME, CHART_COLORS } from "../../utils/colors";
import { shortDate, weightsToChartData } from "../../utils/formatters";

interface Props {
  weightsHistory: Record<string, Record<string, number>>;
  title?: string;
  height?: number;
}

export default function WeightsChart({ weightsHistory, title = "Portfolio Allocation", height = 220 }: Props) {
  if (Object.keys(weightsHistory).length === 0) return null;

  const data = weightsToChartData(weightsHistory);
  const symbols = Object.keys(Object.values(weightsHistory)[0] ?? {});

  return (
    <div className="card">
      {title && <h3 className="text-prose font-semibold mb-4">{title}</h3>}
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 4, right: 12, left: 8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={THEME.border} />
          <XAxis
            dataKey="date"
            tick={{ fill: THEME.muted, fontSize: 11 }}
            tickFormatter={shortDate}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: THEME.muted, fontSize: 11 }}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tickLine={false}
            axisLine={false}
            domain={[0, 1]}
            width={44}
          />
          <Tooltip
            contentStyle={{
              background: THEME.card,
              border: `1px solid ${THEME.border}`,
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: THEME.muted }}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={(v: any) => [`${((+v || 0) * 100).toFixed(1)}%`]}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
            formatter={(value) => <span style={{ color: THEME.text }}>{value}</span>}
          />
          {symbols.map((sym, i) => (
            <Area
              key={sym}
              type="monotone"
              dataKey={sym}
              stackId="1"
              stroke={CHART_COLORS[i % CHART_COLORS.length]}
              fill={CHART_COLORS[i % CHART_COLORS.length]}
              fillOpacity={0.6}
              dot={false}
              strokeWidth={1}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
