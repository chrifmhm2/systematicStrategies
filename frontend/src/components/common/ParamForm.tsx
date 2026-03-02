import type { ParamSchema } from "../../api/types";

interface Props {
  schema: Record<string, ParamSchema>;
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}

/** Renders a dynamic form from a strategy's param_schema. */
export default function ParamForm({ schema, values, onChange }: Props) {
  const entries = Object.entries(schema);
  if (entries.length === 0) {
    return <p className="text-muted text-sm italic">No parameters for this strategy.</p>;
  }

  return (
    <div className="space-y-3">
      {entries.map(([key, spec]) => {
        const val = values[key] ?? spec.default;

        if (spec.type === "boolean") {
          return (
            <div key={key} className="flex items-center justify-between">
              <div>
                <p className="text-prose text-sm font-medium">{key}</p>
                {spec.description && (
                  <p className="text-muted text-xs">{spec.description}</p>
                )}
              </div>
              <button
                type="button"
                onClick={() => onChange(key, !val)}
                className={`relative w-10 h-5 rounded-full transition-colors ${
                  val ? "bg-accent" : "bg-dim"
                }`}
              >
                <span
                  className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
                    val ? "translate-x-5" : "translate-x-0.5"
                  }`}
                />
              </button>
            </div>
          );
        }

        if (spec.type === "string" && spec.enum) {
          return (
            <div key={key}>
              <label className="label">{key}</label>
              {spec.description && (
                <p className="text-muted text-xs mb-1">{spec.description}</p>
              )}
              <select
                className="input w-full"
                value={String(val)}
                onChange={(e) => onChange(key, e.target.value)}
              >
                {spec.enum.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </div>
          );
        }

        // integer or number
        return (
          <div key={key}>
            <label className="label">{key}</label>
            {spec.description && (
              <p className="text-muted text-xs mb-1">{spec.description}</p>
            )}
            <input
              type="number"
              className="input w-full"
              value={String(val)}
              step={spec.type === "integer" ? 1 : "any"}
              min={spec.min}
              max={spec.max}
              onChange={(e) =>
                onChange(
                  key,
                  spec.type === "integer"
                    ? parseInt(e.target.value, 10)
                    : parseFloat(e.target.value)
                )
              }
            />
          </div>
        );
      })}
    </div>
  );
}
