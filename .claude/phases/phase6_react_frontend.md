# Phase 6 — React Frontend

> **Goal**: Build the interactive web dashboard. By the end you have a running React app that communicates with the Phase 5 API and lets users explore strategies, run backtests, and visualize results.

**Prerequisites**: Phase 5 API running at `localhost:8000`.

---

## Project Setup (`frontend/`)

- [ ] **[P6-01]** Scaffold the project: `npm create vite@latest frontend -- --template react-ts`
- [ ] **[P6-02]** Install dependencies:
  - `tailwindcss`, `@tailwindcss/forms`, `postcss`, `autoprefixer`
  - `recharts`
  - `react-router-dom`
  - `axios`
  - `katex` + `react-katex` (LaTeX rendering for strategy formulas)
  - `react-datepicker`
- [ ] **[P6-03]** Configure Tailwind: run `npx tailwindcss init -p`, update `tailwind.config.js` with content paths, add `@tailwind` directives to `src/index.css`
- [ ] **[P6-04]** Configure Vite proxy in `vite.config.ts`: forward `/api` to `http://localhost:8000/api` (avoids CORS issues in dev)
- [ ] **[P6-05]** Set up React Router in `src/App.tsx` with routes for all 6 pages

---

## Design Tokens

- [ ] **[P6-06]** Add custom colors to `tailwind.config.js`:
  - `navy: "#0f172a"` (background)
  - `accent: "#3b82f6"` (electric blue)
  - `gain: "#10b981"` (emerald green)
  - `loss: "#ef4444"` (red)
- [ ] **[P6-07]** Add custom fonts in `index.html`: Inter (headings) and JetBrains Mono (numbers)
- [ ] **[P6-08]** Create `src/utils/colors.ts` — export a `CHART_COLORS` array of 8 distinct hex colors used consistently across all Recharts charts

---

## API Client (`src/api/`)

- [ ] **[P6-09]** Create `src/api/client.ts` — Axios instance with `baseURL = "/api"`, JSON headers, and a response interceptor that throws a typed error on non-2xx responses
- [ ] **[P6-10]** Create `src/api/types.ts` — TypeScript interfaces matching all backend Pydantic schemas:
  - `StrategyInfo`, `BacktestRequest`, `BacktestResponse`, `HedgingRequest`, `HedgingResponse`, `RiskMetrics`, `OptionPricingRequest`, `OptionPricingResponse`
- [ ] **[P6-11]** Create typed API functions in `src/api/client.ts`:
  - `fetchStrategies() -> Promise<StrategyInfo[]>`
  - `runBacktest(req: BacktestRequest) -> Promise<BacktestResponse>`
  - `compareStrategies(req: CompareRequest) -> Promise<BacktestResponse[]>`
  - `simulateHedging(req: HedgingRequest) -> Promise<HedgingResponse>`
  - `analyzeRisk(req: RiskAnalyzeRequest) -> Promise<RiskMetrics>`
  - `fetchPrices(symbols, start, end) -> Promise<Record<string, Record<string, number>>>`
  - `priceOption(req: OptionPricingRequest) -> Promise<OptionPricingResponse>`

---

## Utilities

- [ ] **[P6-12]** Create `src/utils/formatters.ts`:
  - `formatPercent(value: number, decimals = 2) -> string` — e.g. `0.134 → "+13.40%"`
  - `formatCurrency(value: number) -> string` — e.g. `100234.5 → "$100,234.50"`
  - `formatDate(date: string | Date) -> string` — e.g. `"2020-01-02"`
  - `formatNumber(value: number, decimals = 2) -> string`

---

## Layout Components (`src/components/layout/`)

- [ ] **[P6-13]** Create `Navbar.tsx` — top bar with "QuantForge" title and a GitHub icon link
- [ ] **[P6-14]** Create `Sidebar.tsx` — fixed left sidebar with nav links to all 6 pages, collapsible on mobile; active link highlighted in accent blue
- [ ] **[P6-15]** Create `Footer.tsx` — minimal footer with "Built with QuantForge" text

---

## Common Components (`src/components/common/`)

- [ ] **[P6-16]** Create `MetricCard.tsx` — displays a label, a large number value, and an optional delta badge (green if positive, red if negative)
- [ ] **[P6-17]** Create `LoadingSpinner.tsx` — centered spinner with optional label text
- [ ] **[P6-18]** Create `ErrorBanner.tsx` — red banner with error message and a dismiss button
- [ ] **[P6-19]** Create `DataTable.tsx` — generic table with sortable columns, pagination (10 rows/page), and column formatting via a `columns` prop

---

## Chart Components (`src/components/charts/`)

- [ ] **[P6-20]** Create `EquityCurve.tsx` — Recharts `LineChart` with:
  - Multiple lines (one per strategy or portfolio/benchmark)
  - Tooltip showing date + all values
  - Log scale toggle button
  - Responsive container
- [ ] **[P6-21]** Create `DrawdownChart.tsx` — Recharts `AreaChart` filled red, showing drawdown % over time; Y-axis inverted (worst drawdown at bottom)
- [ ] **[P6-22]** Create `CompositionChart.tsx` — Recharts `AreaChart` stacked, showing asset weight over time; one color per asset
- [ ] **[P6-23]** Create `CorrelationHeatmap.tsx` — grid of colored cells (green = positive, red = negative correlation); use a custom SVG or a CSS grid approach; show values inside each cell
- [ ] **[P6-24]** Create `RollingMetrics.tsx` — dual-axis Recharts `LineChart` showing rolling 30d Sharpe (left axis) and rolling 30d volatility (right axis)
- [ ] **[P6-25]** Create `GreeksSurface.tsx` — 3D surface visualization using Recharts (approximated as a heatmap grid) for delta/gamma/vega/theta as a function of spot and vol; include a toggle for which Greek to show

---

## Form Components (`src/components/forms/`)

- [ ] **[P6-26]** Create `AssetSelector.tsx` — multi-select input with search; shows a list of assets from `GET /data/assets`; selected assets displayed as tags
- [ ] **[P6-27]** Create `StrategyConfigurator.tsx` — dynamically renders a form from a strategy's `params` schema (returned by `GET /strategies/{id}`):
  - `type: "number"` → `<input type="number">` with min/max
  - `type: "integer"` → same with step=1
  - `type: "select"` → `<select>` with the `options` array
  - `type: "boolean"` → toggle switch
- [ ] **[P6-28]** Create `BacktestForm.tsx` — combines `AssetSelector`, date range pickers, `StrategyConfigurator`, and a submit button; calls `runBacktest()` on submit
- [ ] **[P6-29]** Create `HedgingForm.tsx` — form with basket weights, strike, maturity, MC params, and data source toggle (Simulated / Historical)

---

## Custom Hooks (`src/hooks/`)

- [ ] **[P6-30]** Create `useStrategies.ts` — fetches strategy list on mount; returns `{ strategies, loading, error }`
- [ ] **[P6-31]** Create `useBacktest.ts` — manages backtest submission state; returns `{ result, loading, error, runBacktest }`
- [ ] **[P6-32]** Create `useRiskMetrics.ts` — fetches risk analysis; returns `{ metrics, loading, error }`

---

## Pages

- [ ] **[P6-33]** Create `src/pages/HomePage.tsx`:
  - Hero section: "QuantForge" title, one-line description, two CTA buttons ("Explore Strategies" → `/strategies`, "Run a Backtest" → `/backtest`)
  - Three feature cards: "8 Strategies", "Real Market Data", "Risk Analytics"
  - Quick stats row (hardcoded or fetched from API)

- [ ] **[P6-34]** Create `src/pages/StrategyExplorerPage.tsx`:
  - Left panel: strategy cards grouped by family (Hedging / Allocation / Signal) fetched from `useStrategies`
  - Right panel (on strategy select): description, LaTeX formula rendered by KaTeX, `StrategyConfigurator` form, `AssetSelector`, date range pickers, "Run Backtest" button
  - On submit: navigate to `BacktestResultsPage` and pass the result via React Router state or context

- [ ] **[P6-35]** Create `src/pages/BacktestResultsPage.tsx`:
  - Row of 4 `MetricCard`s: Total Return, Annualized Return, Sharpe Ratio, Max Drawdown
  - `EquityCurve` chart (portfolio vs benchmark)
  - `DrawdownChart`
  - `RollingMetrics` chart
  - `CompositionChart`
  - Full metrics `DataTable`
  - Trades log `DataTable` (paginated)

- [ ] **[P6-36]** Create `src/pages/StrategyComparisonPage.tsx`:
  - Strategy selector: add up to 5 strategy cards, each with their own `StrategyConfigurator`
  - Shared settings: `AssetSelector`, date range, initial capital
  - "Compare" button → calls `compareStrategies()`
  - Overlaid `EquityCurve` (all strategies on one chart)
  - Comparison `DataTable`: rows = metrics, columns = strategies
  - Side-by-side bar chart (Recharts `BarChart`) for Sharpe, max drawdown, return

- [ ] **[P6-37]** Create `src/pages/RiskAnalyticsPage.tsx`:
  - VaR/CVaR histogram: Recharts `BarChart` of return distribution with vertical reference lines for VaR95 and CVaR95
  - Rolling VaR `LineChart` (252-day window)
  - `CorrelationHeatmap` for selected assets
  - Return distribution with fitted normal overlay (compute from existing backtest result)

- [ ] **[P6-38]** Create `src/pages/HedgingSimulatorPage.tsx`:
  - `HedgingForm` on the left
  - On submit: show `EquityCurve` (portfolio vs option price, one line per path)
  - Tracking error chart (difference between portfolio and option price over time)
  - Delta evolution `LineChart` (one line per underlying)
  - Summary stats: initial option price ± CI, average tracking error, final P&L
  - `GreeksSurface` heatmap with toggle for delta / gamma / vega / theta

---

## How to Run the Frontend

```bash
cd frontend
npm install
npm run dev       # starts at http://localhost:5173, proxies /api to localhost:8000
npm run build     # production build to dist/
npm run lint      # ESLint check
```

---

## Definition of Done

- `npm run dev` starts without errors and the app loads in a browser
- All 6 pages render without JavaScript errors
- `StrategyExplorerPage` loads the strategy list from the live API and renders forms
- Running a backtest from the UI shows an equity curve and metric cards
- `npm run build` completes without errors
