"use client";

import { useEffect, useState } from "react";
import { DEFAULT_CONFIG, apiGet, apiPost, type NowcastResult } from "@/lib/api";
import { DemoBanner, Panel, RiskChip } from "@/components/ui";

export default function DriversPage() {
  const [nc, setNc] = useState<NowcastResult | null>(null);
  const [alert, setAlert] = useState<any>(null);

  useEffect(() => {
    (async () => {
      await apiPost("/api/backtest", { config: DEFAULT_CONFIG });
      const n = await apiPost<{ } & NowcastResult>("/api/nowcast", { config: DEFAULT_CONFIG });
      setNc(n.data as any);
      const a = await apiGet<any>("/api/alerts");
      setAlert(a.data);
    })().catch(console.error);
  }, []);

  if (!nc) return <div className="muted"><span className="spinner" /> Loading drivers…</div>;
  const maxAbs = Math.max(...nc.drivers.map((d) => Math.abs(d.contribution_pp)), 0.01);

  return (
    <div className="stack" style={{ gap: "1rem" }}>
      <div className="topbar">
        <div>
          <h1 className="page-title">Risk Drivers</h1>
          <p className="page-sub">
            Feature contributions explain the model prediction. Association is not causation.
          </p>
        </div>
        <DemoBanner show={nc.is_synthetic} />
      </div>

      {alert && (
        <Panel title="NPL Risk Alert">
          <div className="row">
            <div>
              <div className="metric">{alert.current_nowcast?.toFixed(2)}%</div>
              <div className="muted">
                Change {alert.change_pp >= 0 ? "+" : ""}
                {alert.change_pp?.toFixed(2)} pp
              </div>
            </div>
            <RiskChip signal={alert.signal} />
          </div>
        </Panel>
      )}

      <Panel title="Top NPL Risk Drivers">
        {nc.drivers.map((d, i) => (
          <div className="driver-bar" key={d.feature}>
            <div style={{ width: "100%" }}>
              <div className="row">
                <strong>
                  {i + 1}. {d.feature}
                </strong>
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
              <div className="muted" style={{ fontSize: "0.75rem", marginTop: "0.25rem" }}>
                {d.note}
              </div>
            </div>
          </div>
        ))}
      </Panel>

      <Panel title="News Signals">
        {nc.news_signals.length ? (
          nc.news_signals.map((s) => (
            <div className="article" key={s}>
              • {s}
            </div>
          ))
        ) : (
          <div className="muted">No news signals attached.</div>
        )}
      </Panel>
    </div>
  );
}
