"use client";

import { useEffect, useState } from "react";
import { DEFAULT_CONFIG, apiGet, apiPost } from "@/lib/api";
import { DemoBanner, Panel } from "@/components/ui";

export default function BacktestingPage() {
  const [track, setTrack] = useState<any>(null);
  const [board, setBoard] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const b = await apiPost<any>("/api/backtest", { config: DEFAULT_CONFIG });
        setBoard(b.data.leaderboard || []);
        const t = await apiGet<any>("/api/track-record");
        setTrack(t.data);
      } catch (e: any) {
        setError(e.message);
      }
    })();
  }, []);

  if (error) return <Panel title="Error">{error}</Panel>;
  if (!track) return <div className="muted"><span className="spinner" /> Running temporal backtest…</div>;

  return (
    <div className="stack" style={{ gap: "1rem" }}>
      <div className="topbar">
        <div>
          <h1 className="page-title">Backtesting</h1>
          <p className="page-sub">
            Expanding-window nowcast evaluation. Time order is preserved — series are never shuffled.
          </p>
        </div>
        <DemoBanner show={track.is_synthetic} />
      </div>

      <div className="grid grid-4">
        <Panel title="RMSE"><div className="metric">{track.metrics?.rmse?.toFixed(3)}</div></Panel>
        <Panel title="MAE"><div className="metric">{track.metrics?.mae?.toFixed(3)}</div></Panel>
        <Panel title="Bias"><div className="metric">{track.metrics?.bias?.toFixed(3)}</div></Panel>
        <Panel title="Directional Acc.">
          <div className="metric">
            {track.metrics?.directional_accuracy != null
              ? `${(track.metrics.directional_accuracy * 100).toFixed(0)}%`
              : "—"}
          </div>
        </Panel>
      </div>

      <Panel title="Nowcast Track Record">
        <table className="table">
          <thead>
            <tr>
              <th>Period</th>
              <th style={{ textAlign: "right" }}>Nowcast</th>
              <th style={{ textAlign: "right" }}>Actual</th>
              <th style={{ textAlign: "right" }}>Error</th>
            </tr>
          </thead>
          <tbody>
            {(track.records || []).map((r: any) => (
              <tr key={r.period}>
                <td>{r.period}</td>
                <td className="num">{r.nowcast.toFixed(3)}</td>
                <td className="num">{r.actual.toFixed(3)}</td>
                <td className="num">{r.error.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <Panel title="Model Ranking">
        <table className="table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Model</th>
              <th style={{ textAlign: "right" }}>RMSE</th>
              <th style={{ textAlign: "right" }}>MAE</th>
            </tr>
          </thead>
          <tbody>
            {board.map((r) => (
              <tr key={r.model}>
                <td>{r.rank}</td>
                <td>{r.display_name}</td>
                <td className="num">{r.rmse.toFixed(3)}</td>
                <td className="num">{r.mae.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
