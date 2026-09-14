"use client";

import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";
import { DemoBanner, Panel } from "@/components/ui";

export default function DataQualityPage() {
  const [rows, setRows] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<any[]>("/api/data-quality")
      .then((r) => setRows(r.data))
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <Panel title="Error">{error}</Panel>;
  if (!rows.length) return <div className="muted"><span className="spinner" /> Assessing data quality…</div>;

  return (
    <div className="stack" style={{ gap: "1rem" }}>
      <div className="topbar">
        <div>
          <h1 className="page-title">Data Quality</h1>
          <p className="page-sub">
            Missingness, publication lag, outliers, and source metadata for the current vintage.
          </p>
        </div>
        <DemoBanner show />
      </div>
      <Panel title="Indicator Health">
        <table className="table">
          <thead>
            <tr>
              <th>Indicator</th>
              <th>Status</th>
              <th>Frequency</th>
              <th>Last Update</th>
              <th style={{ textAlign: "right" }}>Pub. Lag (d)</th>
              <th style={{ textAlign: "right" }}>Missing %</th>
              <th style={{ textAlign: "right" }}>Outliers</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.indicator}>
                <td>{r.indicator}</td>
                <td>
                  <span className={`chip ${r.status === "ok" ? "chip-low" : "chip-elevated"}`}>
                    {r.status}
                  </span>
                </td>
                <td>{r.frequency}</td>
                <td>{r.last_update}</td>
                <td className="num">{r.publication_lag_days}</td>
                <td className="num">{r.missing_pct.toFixed(1)}</td>
                <td className="num">{r.outlier_count}</td>
                <td>{r.source}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
