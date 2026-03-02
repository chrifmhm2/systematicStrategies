import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { THEME, CHART_COLORS } from "../../utils/colors";
import { shortDate } from "../../utils/formatters";

interface Series {
  key: string;
  name: string;
  color?: string;
}

interface Props {
  data: { date: string; [key: string]: string | number }[];
  series: Series[];
  title?: string;
  height?: number;
  initialValue?: number;
}

const ChartTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: { name: string; value: number; color: string }[];
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: THEME.card,
        border: `1px solid ${THEME.border}`,
        borderRadius: 8,
        padding: "10px 14px",
      }}
    >
      <p style={{ color: THEME.muted, marginBottom: 6, fontSize: 12 }}>{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color, margin: "2px 0", fontSize: 13 }}>
          {p.name}: <span style={{ fontFamily: "monospace" }}>${p.value.toLocaleString()}</span>
        </p>
      ))}
    </div>
  );
};

export default function EquityCurve({
  data, series, title, height = 300, initialValue,
}: Props) {
  return (
    <div className="card">
      {title && <h3 className="text-prose font-semibold mb-4">{title}</h3>}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: 12, left: 8, bottom: 0 }}>
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
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            tickLine={false}
            axisLine={false}
            width={56}
          />
          <Tooltip content={<ChartTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
            formatter={(value) => <span style={{ color: THEME.text }}>{value}</span>}
          />
          {initialValue != null && (
            <ReferenceLine y={initialValue} stroke={THEME.muted} strokeDasharray="4 4" />
          )}
          {series.map((s, i) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.name}
              stroke={s.color ?? CHART_COLORS[i % CHART_COLORS.length]}
              dot={false}
              strokeWidth={2}
              activeDot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
