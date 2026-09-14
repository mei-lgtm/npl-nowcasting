"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "NPL Nowcast" },
  { href: "/news", label: "News Sentiment" },
  { href: "/indicators", label: "Indicators" },
  { href: "/models", label: "Models" },
  { href: "/backtesting", label: "Backtesting" },
  { href: "/scenarios", label: "Scenario Analysis" },
  { href: "/drivers", label: "Risk Drivers" },
  { href: "/data-quality", label: "Data Quality" },
  { href: "/run", label: "Run Workflow" },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          NPL <span>Signal</span>
        </div>
        <div className="brand-sub">Nowcasting Intelligence</div>
      </div>
      <nav className="nav">
        {LINKS.map((l) => (
          <Link
            key={l.href}
            href={l.href}
            className={pathname === l.href ? "active" : undefined}
          >
            {l.label}
          </Link>
        ))}
      </nav>
      <div style={{ marginTop: "auto", padding: "0.6rem", color: "var(--muted)", fontSize: "0.75rem" }}>
        Actual · Nowcast · Forecast are kept distinct. Demo mode uses synthetic data only.
      </div>
    </aside>
  );
}
