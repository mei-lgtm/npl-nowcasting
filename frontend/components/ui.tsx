import clsx from "clsx";
import type { RiskSignal } from "@/lib/api";

export function RiskChip({ signal }: { signal: RiskSignal }) {
  const cls =
    signal === "LOW"
      ? "chip-low"
      : signal === "MODERATE"
        ? "chip-moderate"
        : signal === "ELEVATED"
          ? "chip-elevated"
          : signal === "HIGH"
            ? "chip-high"
            : "chip-critical";
  return <span className={clsx("chip", cls)}>{signal}</span>;
}

export function DemoBanner({ show }: { show: boolean }) {
  if (!show) return null;
  return <div className="demo-banner">Demo / Synthetic Data</div>;
}

export function Panel({
  title,
  children,
  style,
}: {
  title?: string;
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  return (
    <section className="panel" style={style}>
      {title ? <h3>{title}</h3> : null}
      {children}
    </section>
  );
}
