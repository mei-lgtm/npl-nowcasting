"use client";

import { useEffect, useState } from "react";
import { DEFAULT_CONFIG, apiGet, apiPost } from "@/lib/api";
import { DemoBanner, Panel } from "@/components/ui";

export default function ModelsPage() {
  const [catalog, setCatalog] = useState<any>(null);
  const [board, setBoard] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const c = await apiGet<any>("/api/catalog");
        setCatalog(c.data);
        const b = await apiPost<any>("/api/backtest", { config: DEFAULT_CONFIG });
        setBoard(b.data.leaderboard || []);
      } catch (e: any) {
        setError(e.message);
      }
    })();
  }, []);

  if (error) return <Panel title="Error">{error}</Panel>;
  if (!catalog) return <div className="muted"><span className="spinner" /> Loading models…</div>;

  const groups: Record<string, Array<[string, any]>> = {};
  for (const [k, v] of Object.entries(catalog.models || {})) {
    const fam = (v as any).family;
    groups[fam] = groups[fam] || [];
    groups[fam].push([k, v]);
  }

  return (
    <div className="stack" style={{ gap: "1rem" }}>
      <div className="topbar">
        <div>
          <h1 className="page-title">Models</h1>
          <p className="page-sub">
            Econometric, mixed-frequency, ML, deep learning, and hybrid nowcasters. Ranking uses
            out-of-sample RMSE/MAE — never in-sample R² alone.
          </p>
        </div>
        <DemoBanner show={catalog.is_synthetic} />
      </div>

      <Panel title="Leaderboard">
        <table className="table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Model</th>
              <th style={{ textAlign: "right" }}>RMSE</th>
              <th style={{ textAlign: "right" }}>MAE</th>
              <th style={{ textAlign: "right" }}>MAPE</th>
              <th style={{ textAlign: "right" }}>R²</th>
              <th style={{ textAlign: "right" }}>Dir. Acc.</th>
            </tr>
          </thead>
          <tbody>
            {board.map((r) => (
              <tr key={r.model}>
                <td>{r.rank}</td>
                <td>{r.display_name}</td>
                <td className="num">{r.rmse.toFixed(3)}</td>
                <td className="num">{r.mae.toFixed(3)}</td>
                <td className="num">{r.mape.toFixed(1)}%</td>
                <td className="num">{Number.isFinite(r.r2) ? r.r2.toFixed(2) : "—"}</td>
                <td className="num">
                  {r.directional_accuracy != null ? `${(r.directional_accuracy * 100).toFixed(0)}%` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <div className="grid grid-2">
        {Object.entries(groups).map(([fam, models]) => (
          <Panel key={fam} title={fam.replace("_", " ")}>
            <div className="stack">
              {models.map(([id, meta]) => (
                <div className="row" key={id}>
                  <span>{meta.name}</span>
                  <code className="muted" style={{ fontSize: "0.75rem" }}>
                    {id}
                  </code>
                </div>
              ))}
            </div>
          </Panel>
        ))}
        <Panel title="Sentiment Models">
          <div className="stack">
            {(catalog.sentiment_models || []).map((m: string) => (
              <div key={m} className="row">
                <span>{m}</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}
