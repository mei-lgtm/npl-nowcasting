"use client";

import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";
import { DemoBanner, Panel } from "@/components/ui";

export default function IndicatorsPage() {
  const [rows, setRows] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<any[]>("/api/indicators")
      .then((r) => setRows(r.data))
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <Panel title="Error">{error}</Panel>;
  if (!rows.length) return <div className="muted"><span className="spinner" /> Loading indicators…</div>;

  return (
    <div className="stack" style={{ gap: "1rem" }}>
      <div className="topbar">
        <div>
          <h1 className="page-title">Indicators</h1>
          <p className="page-sub">
            Latest vintage-available macroeconomic, banking, and news-derived features.
          </p>
        </div>
        <DemoBanner show />
      </div>
      <Panel title="Latest Available Values">
        <table className="table">
          <thead>
            <tr>
              <th>Indicator</th>
              <th>Period</th>
              <th style={{ textAlign: "right" }}>Value</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.indicator}>
                <td>{r.indicator}</td>
                <td>{r.period}</td>
                <td className="num">{Number(r.value).toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
