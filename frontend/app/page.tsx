"use client";

import { useEffect, useState } from "react";
import {
  DEFAULT_CONFIG,
  apiPost,
  type WorkflowResponse,
} from "@/lib/api";
import { NowcastChart, StressChart } from "@/components/Charts";
import { DemoBanner, Panel, RiskChip } from "@/components/ui";

export default function DashboardPage() {
  const [data, setData] = useState<WorkflowResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        const res = await apiPost<WorkflowResponse>("/api/workflow/run", {
          config: DEFAULT_CONFIG,
        });
        if (!cancelled) setData(res.data);
      } catch (e: any) {
        if (!cancelled) setError(e.message || "Failed to load nowcast");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <div className="stack" style={{ paddingTop: "3rem", alignItems: "flex-start" }}>
        <div className="row">
          <span className="spinner" />
          <span className="muted">Running vintage-aligned nowcast pipeline…</span>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <Panel title="Backend unavailable">
        <p>{error || "No data"}</p>
        <p className="muted">
          Start the API: <code>cd backend && source .venv/bin/activate && uvicorn app:app --reload --port 8000</code>
        </p>
      </Panel>
    );
  }

  const nc = data.nowcast;
  const maxAbs = Math.max(...nc.drivers.map((d) => Math.abs(d.contribution_pp)), 0.01);

  return (
    <div className="stack" style={{ gap: "1rem" }}>
      <div className="topbar">
        <div>
          <h1 className="page-title">NPL Nowcast</h1>
          <p className="page-sub">
            Current-period estimate using information available before the official release.
            Actual, nowcast, and forecast series remain separate.
          </p>
        </div>
        <DemoBanner show={nc.is_synthetic} />
      </div>

      <div className="grid grid-4">
        <Panel title="Current Estimate">
          <div className="metric">{nc.nowcast_value.toFixed(2)}%</div>
          <div className="muted" style={{ marginTop: "0.5rem" }}>
            {nc.reference_period} · {nc.observation_type.toUpperCase()}
          </div>
        </Panel>
        <Panel title="Actual Last Release">
          <div className="metric">
            {nc.actual_last_release != null ? `${nc.actual_last_release.toFixed(2)}%` : "—"}
          </div>
          <div className="muted" style={{ marginTop: "0.5rem" }}>
            {nc.actual_last_period || "n/a"}
          </div>
        </Panel>
        <Panel title="Nowcast Change">
          <div className="metric" style={{ color: (nc.change_pp || 0) >= 0 ? "var(--warn)" : "var(--ok)" }}>
            {nc.change_pp != null ? `${nc.change_pp >= 0 ? "+" : ""}${nc.change_pp.toFixed(2)} pp` : "—"}
          </div>
          <div className="muted" style={{ marginTop: "0.5rem" }}>
            vs last actual
          </div>
        </Panel>
        <Panel title="Risk Signal">
          <div style={{ marginBottom: "0.75rem" }}>
            <RiskChip signal={nc.risk_signal} />
          </div>
          <div className="muted">
            CI {nc.confidence_low.toFixed(2)}% – {nc.confidence_high.toFixed(2)}%
          </div>
          <div className="muted" style={{ marginTop: "0.35rem" }}>
            Confidence: {nc.model_confidence} · {nc.model_used}
          </div>
        </Panel>
      </div>

      <div className="grid grid-2">
        <Panel title="Actual vs Nowcast vs Forecast">
          <NowcastChart data={data.timeseries} />
        </Panel>
        <Panel title="News Stress Index">
          <div className="row" style={{ marginBottom: "0.75rem" }}>
            <div>
              <div className="metric-sm">{data.news.current.toFixed(2)}</div>
              <div className="muted">Current (−1 soft / +1 stress)</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div className="metric-sm" style={{ color: data.news.change >= 0 ? "var(--warn)" : "var(--ok)" }}>
                {data.news.change >= 0 ? "+" : ""}
                {data.news.change.toFixed(2)}
              </div>
              <div className="muted">MoM change</div>
            </div>
          </div>
          <StressChart data={data.news.series} />
        </Panel>
      </div>

      <div className="grid grid-2">
        <Panel title="Top NPL Risk Drivers">
          <p className="muted" style={{ marginTop: 0, fontSize: "0.85rem" }}>
            Model contributions / associations — not causal claims.
          </p>
          <div className="stack">
            {nc.drivers.map((d) => (
              <div className="driver-bar" key={d.feature}>
                <div>
                  <div className="row">
                    <strong style={{ fontSize: "0.92rem" }}>{d.feature}</strong>
                    <span className="metric-sm">
                      {d.contribution_pp >= 0 ? "+" : ""}
                      {d.contribution_pp.toFixed(2)} pp
                    </span>
                  </div>
                  <div className="bar-track">
                    <div
                      className={`bar-fill ${d.contribution_pp < 0 ? "neg" : ""}`}
                      style={{ width: `${(Math.abs(d.contribution_pp) / maxAbs) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="AI Analyst">
          <p style={{ marginTop: 0, lineHeight: 1.55 }}>{data.analyst.summary}</p>
          <ul style={{ margin: "0.5rem 0", paddingLeft: "1.1rem", color: "var(--muted)" }}>
            {data.analyst.bullets.map((b) => (
              <li key={b} style={{ marginBottom: "0.35rem" }}>
                {b}
              </li>
            ))}
          </ul>
          <p className="muted" style={{ fontSize: "0.85rem" }}>
            {data.analyst.confidence_note}
          </p>
        </Panel>
      </div>

      <div className="grid grid-2">
        <Panel title="Model Leaderboard (OOS)">
          <table className="table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                <th style={{ textAlign: "right" }}>RMSE</th>
                <th style={{ textAlign: "right" }}>MAE</th>
                <th style={{ textAlign: "right" }}>MAPE</th>
                <th style={{ textAlign: "right" }}>R²</th>
              </tr>
            </thead>
            <tbody>
              {data.selection.leaderboard.slice(0, 8).map((r) => (
                <tr key={r.model}>
                  <td>{r.rank}</td>
                  <td>{r.display_name}</td>
                  <td className="num">{r.rmse.toFixed(3)}</td>
                  <td className="num">{r.mae.toFixed(3)}</td>
                  <td className="num">{r.mape.toFixed(1)}%</td>
                  <td className="num">{Number.isFinite(r.r2) ? r.r2.toFixed(2) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <Panel title="Why did the nowcast move?">
          <p style={{ marginTop: 0 }}>
            Nowcast moved by{" "}
            <strong>
              {nc.change_pp != null ? `${nc.change_pp >= 0 ? "+" : ""}${nc.change_pp.toFixed(2)} pp` : "n/a"}
            </strong>{" "}
            versus the last official release. Higher news stress is associated with a higher NPL
            nowcast and contributes materially to the model prediction.
          </p>
          <div className="stack">
            {nc.news_signals.length ? (
              nc.news_signals.map((s) => (
                <div className="article" key={s}>
                  <div style={{ fontSize: "0.92rem" }}>• {s}</div>
                </div>
              ))
            ) : (
              <div className="muted">No recent negative headlines in the demo window.</div>
            )}
          </div>
          <p className="muted" style={{ fontSize: "0.8rem", marginBottom: 0 }}>
            {nc.disclaimer}
          </p>
        </Panel>
      </div>
    </div>
  );
}
