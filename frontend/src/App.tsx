import { BrowserRouter, Routes, Route } from "react-router-dom";
import Sidebar from "./components/layout/Sidebar";
import Navbar from "./components/layout/Navbar";
import HomePage from "./pages/HomePage";
import BacktestPage from "./pages/BacktestPage";
import ComparisonPage from "./pages/ComparisonPage";
import RiskPage from "./pages/RiskPage";
import HedgingPage from "./pages/HedgingPage";
import PricingPage from "./pages/PricingPage";

const PAGE_META: Record<string, { title: string; subtitle: string }> = {
  "/":          { title: "QuantForge",            subtitle: "Systematic quantitative strategies dashboard" },
  "/backtest":  { title: "Backtest",               subtitle: "Run a single-strategy backtest with full metrics" },
  "/compare":   { title: "Strategy Comparison",    subtitle: "Compare up to 4 strategies side-by-side" },
  "/risk":      { title: "Risk Analytics",         subtitle: "VaR, CVaR, drawdown, and return distribution" },
  "/hedging":   { title: "Hedging Simulator",      subtitle: "Delta hedging simulation across GBM paths" },
  "/pricing":   { title: "Option Pricing",         subtitle: "Black-Scholes and Monte Carlo with full Greeks" },
};

function Layout() {
  // Derive current path meta on each render — no hooks needed here
  const path = window.location.pathname;
  const meta = PAGE_META[path] ?? PAGE_META["/"];

  return (
    <div className="flex min-h-screen bg-navy">
      <Sidebar />
      <main className="flex-1 ml-52 px-8 py-6 min-h-screen overflow-y-auto">
        <Navbar title={meta.title} subtitle={meta.subtitle} />
        <Routes>
          <Route path="/"         element={<HomePage />} />
          <Route path="/backtest" element={<BacktestPage />} />
          <Route path="/compare"  element={<ComparisonPage />} />
          <Route path="/risk"     element={<RiskPage />} />
          <Route path="/hedging"  element={<HedgingPage />} />
          <Route path="/pricing"  element={<PricingPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout />
    </BrowserRouter>
  );
}
