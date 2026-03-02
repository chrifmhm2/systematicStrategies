interface Props {
  label: string;
  value: string;
  delta?: number | null;  // if set, colours the value (positive = gain, negative = loss)
  unit?: string;
  muted?: boolean;
}

export default function MetricCard({ label, value, delta, unit, muted }: Props) {
  const sign = delta == null ? null : delta >= 0 ? "gain" : "loss";
  const valueClass =
    sign === "gain" ? "text-gain" : sign === "loss" ? "text-loss" : muted ? "text-muted" : "text-prose";

  return (
    <div className="card flex flex-col gap-1">
      <span className="label">{label}</span>
      <span className={`font-mono text-2xl font-semibold ${valueClass}`}>
        {value}
        {unit && <span className="text-sm ml-1 text-muted">{unit}</span>}
      </span>
    </div>
  );
}
