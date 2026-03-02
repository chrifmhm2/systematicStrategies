import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { THEME } from "../../utils/colors";
import { shortDate, computeDrawdown } from "../../utils/formatters";

interface Props {
  portfolioValues: Record<string, number>;
  title?: string;
  height?: number;
}

export default function DrawdownChart({ portfolioValues, title = "Drawdown", height = 220 }: Props) {
  const data = computeDrawdown(portfolioValues);

  return (
    <div className="card">
      {title && <h3 className="text-prose font-semibold mb-4">{title}</h3>}
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 4, right: 12, left: 8, bottom: 0 }}>
          <defs>
            <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={THEME.loss} stopOpacity={0.4} />
              <stop offset="100%" stopColor={THEME.loss} stopOpacity={0.02} />
            </linearGradient>
          </defs>
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
            tickFormatter={(v) => `${v.toFixed(1)}%`}
            tickLine={false}
            axisLine={false}
            width={52}
            domain={["dataMin", 0]}
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
            formatter={(v: any) => [`${(+v || 0).toFixed(2)}%`, "Drawdown"]}
          />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke={THEME.loss}
            fill="url(#ddGrad)"
            strokeWidth={1.5}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
