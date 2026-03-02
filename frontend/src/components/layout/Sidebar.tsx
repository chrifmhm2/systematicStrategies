import { NavLink } from "react-router-dom";

const NAV = [
  { to: "/",         icon: "⬡", label: "Home" },
  { to: "/backtest", icon: "◎", label: "Backtest" },
  { to: "/compare",  icon: "≡", label: "Compare" },
  { to: "/risk",     icon: "△", label: "Risk" },
  { to: "/hedging",  icon: "◈", label: "Hedging" },
  { to: "/pricing",  icon: "◇", label: "Pricing" },
];

export default function Sidebar() {
  return (
    <nav className="fixed top-0 left-0 h-full w-52 bg-card border-r border-dim flex flex-col z-20">
      {/* Logo */}
      <div className="flex items-center gap-2 px-5 py-5 border-b border-dim">
        <span className="text-accent text-xl font-bold">◆</span>
        <span className="text-prose font-bold text-base tracking-tight">QuantForge</span>
      </div>

      {/* Links */}
      <div className="flex-1 py-4 space-y-1 px-3">
        {NAV.map(({ to, icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? "bg-accent/10 text-accent"
                  : "text-muted hover:text-prose hover:bg-elevated"
              }`
            }
          >
            <span className="text-base w-4 text-center">{icon}</span>
            {label}
          </NavLink>
        ))}
      </div>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-dim text-muted text-xs">
        Phase 6 — v1.0
      </div>
    </nav>
  );
}
