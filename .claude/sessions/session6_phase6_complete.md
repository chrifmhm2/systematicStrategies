# Session 6 — Phase 6 Complete
**Date:** 2026-03-02
**Status:** Phase 6 fully implemented and building. Phase 7 (Docker + CI/CD) is next.

---

## What was built — Phase 6 (React Frontend)

### Project setup `[P6-01–P6-05]`

| File | What it does |
|------|-------------|
| `frontend/` | Scaffolded with `npm create vite@latest -- --template react-ts` (Node v24.13.0 via nvm) |
| `vite.config.ts` | Vite proxy: `/api` → `http://localhost:8000` (avoids CORS in dev) |
| `tailwind.config.js` | Dark-theme color tokens + Inter/JetBrains Mono fonts |
| `src/index.css` | Tailwind directives + global dark styles + `.btn-primary`, `.btn-secondary`, `.card`, `.input`, `.label` utility classes |
| `index.html` | Google Fonts: Inter 400/500/600/700, JetBrains Mono 400/500 |

### Design tokens (`tailwind.config.js`)

```
navy: #0f1117  — page background (matches Python demo charts)
card: #1a1d27  — card surface
elevated: #252836  — inputs, code blocks
dim: #2a2d3d  — borders, dividers
muted: #6272a4  — secondary text, labels
prose: #c8ccd8  — primary text
accent: #4c9be8  — electric blue
gain: #50fa7b  — green (positive values)
loss: #ff5555  — red (negative values)
warn: #ffb86c  — orange (warnings)
purple: #bd93f9
cyan: #8be9fd
```

### API Client `[P6-09–P6-11]` → `src/api/`

| File | What it does |
|------|-------------|
| `types.ts` | TypeScript interfaces matching all Phase 5 Pydantic schemas |
| `client.ts` | Axios instance (`baseURL="/api"`) + 8 typed API functions |

API functions:
- `fetchStrategies()` → `StrategyInfo[]`
- `runBacktest(req)` → `BacktestResponse`
- `compareStrategies(req)` → `BacktestResponse[]`
- `simulateHedging(req)` → `HedgingResponse`
- `analyzeRisk(req)` → `RiskMetrics`
- `fetchAssets()` → `string[]`
- `priceOption(req)` → `OptionPricingResponse`

### Utilities `[P6-12]` → `src/utils/`

| File | Exports |
|------|---------|
| `colors.ts` | `CHART_COLORS[8]`, `THEME` object, `metricColor(v)` |
| `formatters.ts` | `formatPercent`, `formatCurrency`, `formatNumber`, `formatDate`, `portfolioToChartData`, `mergePortfolios`, `computeDrawdown`, `weightsToChartData`, `shortDate` |

### Hooks `[P6-30–P6-32]` → `src/hooks/`

- `useStrategies` — fetches all strategies on mount
- `useBacktest` — manages backtest submission + loading state
- `useRiskMetrics` — manages risk analysis request

### Layout `[P6-13–P6-14]` → `src/components/layout/`

- `Sidebar.tsx` — fixed left nav (52px wide), 6 nav links with active highlight
- `Navbar.tsx` — page title + subtitle + GitHub link

### Common components `[P6-16–P6-19]` → `src/components/common/`

- `MetricCard.tsx` — label + large mono value + colour from `delta` sign
- `LoadingSpinner.tsx` — spin animation + label
- `ErrorBanner.tsx` — red banner, dismissible
- `ParamForm.tsx` — **dynamic form rendered from `param_schema`**: integer/number → `<input type=number>`, boolean → toggle switch, string+enum → `<select>`, string → text input

### Chart components → `src/components/charts/`

- `EquityCurve.tsx` — Recharts `LineChart`, multiple series, tooltip, initial-value reference line
- `DrawdownChart.tsx` — `AreaChart` red gradient, Y-axis domain `[dataMin, 0]`
- `WeightsChart.tsx` — stacked `AreaChart` of portfolio allocation over time
- `ReturnHistogram.tsx` — `BarChart` of daily return distribution with VaR/CVaR reference lines

### Pages `[P6-33–P6-38]`

| Page | Route | Key features |
|------|-------|-------------|
| `HomePage` | `/` | Hero, quick stats from API, strategy list |
| `BacktestPage` | `/backtest` | Strategy selector (grouped by family), asset tags, date range, dynamic ParamForm, results: 8 metric cards + equity curve + drawdown + weights + trades log |
| `ComparisonPage` | `/compare` | 2–4 strategy slots, each with own ParamForm, overlaid equity curves, metrics comparison table |
| `RiskPage` | `/risk` | Quick backtest form, return histogram, equity curve, full 14-metric report |
| `HedgingPage` | `/hedging` | Delta hedge simulator form, paths equity curve, per-path tracking errors |
| `PricingPage` | `/pricing` | BS/MC pricing, Greeks grid, **strike sweep** (21 strikes → call/put price + delta charts) |

### App.tsx

- `BrowserRouter` + `Routes` for all 6 pages
- Layout: fixed `Sidebar` (w-52) + scrollable `main` (ml-52)
- `Navbar` title/subtitle derived from `window.location.pathname`

---

## Key design decisions / improvements over the plan

1. **Merged StrategyExplorerPage + BacktestResultsPage** into one `BacktestPage` — results appear inline below the form. Better UX.
2. **Added PricingPage** (not in original plan) — exposes the `/api/pricing/option` endpoint with a strike sweep feature.
3. **Dynamic `ParamForm`** reads `param_schema` from the API (not hardcoded) — adding a new strategy with params auto-appears in the UI.
4. **Dark theme** exactly matches the Python demo chart palette (same hex codes).
5. **Native date inputs** instead of react-datepicker — simpler, no extra dependency.
6. **Strike sweep** in PricingPage — loops 21 strikes and plots call/put price + delta vs K.
7. **Skipped katex** — no LaTeX rendering, keeps deps minimal for a first touchable version.
8. **Recharts `Formatter` type fix** — used `any` annotation on tooltip formatters to bypass Recharts strict generic type (known TS interop issue with Recharts v2).

---

## Build & run

```bash
# Frontend dev server (requires backend on :8000)
cd /home/chrifmhm/systematicStrategies/frontend
export PATH="/home/chrifmhm/.nvm/versions/node/v24.13.0/bin:$PATH"
npm run dev       # http://localhost:5173 — proxies /api → localhost:8000
npm run build     # production build to dist/ — 0 errors

# Backend (must be running for API calls to work)
cd /home/chrifmhm/systematicStrategies/backend
.venv/bin/uvicorn main:app --reload --port 8000
```

---

## Files created in Phase 6

```
frontend/
├── index.html                          (updated)
├── vite.config.ts                      (updated — proxy)
├── tailwind.config.js                  (updated — dark theme tokens)
├── postcss.config.js                   (generated)
├── src/
│   ├── index.css                       (replaced — Tailwind + global styles)
│   ├── App.tsx                         (replaced — Router + Layout)
│   ├── App.css                         (cleared)
│   ├── api/
│   │   ├── types.ts                    (created)
│   │   └── client.ts                   (created)
│   ├── utils/
│   │   ├── colors.ts                   (created)
│   │   └── formatters.ts               (created)
│   ├── hooks/
│   │   ├── useStrategies.ts            (created)
│   │   ├── useBacktest.ts              (created)
│   │   └── useRiskMetrics.ts           (created)
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Navbar.tsx              (created)
│   │   │   └── Sidebar.tsx             (created)
│   │   ├── common/
│   │   │   ├── MetricCard.tsx          (created)
│   │   │   ├── LoadingSpinner.tsx      (created)
│   │   │   ├── ErrorBanner.tsx         (created)
│   │   │   └── ParamForm.tsx           (created)
│   │   └── charts/
│   │       ├── EquityCurve.tsx         (created)
│   │       ├── DrawdownChart.tsx       (created)
│   │       ├── WeightsChart.tsx        (created)
│   │       └── ReturnHistogram.tsx     (created)
│   └── pages/
│       ├── HomePage.tsx                (created)
│       ├── BacktestPage.tsx            (created)
│       ├── ComparisonPage.tsx          (created)
│       ├── RiskPage.tsx                (created)
│       ├── HedgingPage.tsx             (created)
│       └── PricingPage.tsx             (created)
```

---

## What's next — Phase 7 (Polish & Deploy)

Read `.claude/phases/phase7_polish_and_deploy.md`.

TODOs:
- Docker + docker-compose (backend + frontend containers)
- GitHub Actions CI/CD
- Railway (backend) + Vercel (frontend) deployment
- README with full setup instructions
