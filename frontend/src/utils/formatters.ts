export function formatPercent(value: number | null | undefined, decimals = 2): string {
  if (value == null) return "—";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(decimals)}%`;
}

export function formatCurrency(value: number | null | undefined): string {
  if (value == null) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatNumber(value: number | null | undefined, decimals = 2): string {
  if (value == null) return "—";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(decimals)}`;
}

export function formatDate(date: string | Date): string {
  const d = typeof date === "string" ? new Date(date) : date;
  return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
}

// Convert portfolio_values dict → Recharts-compatible array
export function portfolioToChartData(
  portfolio: Record<string, number>,
  key = "portfolio"
): { date: string; [key: string]: string | number }[] {
  return Object.entries(portfolio).map(([date, value]) => ({
    date,
    [key]: Math.round(value * 100) / 100,
  }));
}

// Merge multiple portfolio_values dicts into one array keyed by date
export function mergePortfolios(
  series: { name: string; values: Record<string, number> }[]
): { date: string; [key: string]: string | number }[] {
  if (series.length === 0) return [];
  const allDates = Object.keys(series[0].values);
  return allDates.map((date) => {
    const row: { date: string; [key: string]: string | number } = { date };
    series.forEach(({ name, values }) => {
      row[name] = Math.round((values[date] ?? 0) * 100) / 100;
    });
    return row;
  });
}

// Compute drawdown series from portfolio values dict
export function computeDrawdown(
  values: Record<string, number>
): { date: string; drawdown: number }[] {
  let peak = -Infinity;
  return Object.entries(values).map(([date, v]) => {
    if (v > peak) peak = v;
    const dd = peak > 0 ? ((v - peak) / peak) * 100 : 0;
    return { date, drawdown: Math.round(dd * 100) / 100 };
  });
}

// Convert weights_history dict → Recharts stacked area data
export function weightsToChartData(
  weights_history: Record<string, Record<string, number>>
): { date: string; [key: string]: string | number }[] {
  return Object.entries(weights_history).map(([date, weights]) => ({
    date,
    ...Object.fromEntries(
      Object.entries(weights).map(([sym, w]) => [sym, Math.round(w * 1000) / 1000])
    ),
  }));
}

// Shorten tick labels for X axis: "2023-04-15" → "Apr '23"
export function shortDate(dateStr: string): string {
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
  } catch {
    return dateStr;
  }
}
