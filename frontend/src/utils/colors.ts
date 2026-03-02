// Consistent chart colors across all Recharts components.
// Same palette as the Python demo scripts.
export const CHART_COLORS = [
  "#4c9be8", // blue
  "#50fa7b", // green
  "#ffb86c", // orange
  "#ff5555", // red
  "#bd93f9", // purple
  "#8be9fd", // cyan
  "#f1fa8c", // yellow
  "#ff79c6", // pink
];

export const THEME = {
  bg:       "#0f1117",
  card:     "#1a1d27",
  elevated: "#252836",
  border:   "#2a2d3d",
  text:     "#c8ccd8",
  muted:    "#6272a4",
  accent:   "#4c9be8",
  gain:     "#50fa7b",
  loss:     "#ff5555",
  warn:     "#ffb86c",
};

export function metricColor(value: number | null | undefined): string {
  if (value == null) return THEME.muted;
  return value >= 0 ? THEME.gain : THEME.loss;
}
