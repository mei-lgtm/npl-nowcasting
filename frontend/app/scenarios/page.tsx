"use client";

import { useEffect, useState } from "react";
import { apiPost, type ScenarioInput } from "@/lib/api";
import { DemoBanner, Panel } from "@/components/ui";

const defaultBase: ScenarioInput = {
  gdp_growth: 5.0,
  credit_growth: 8.0,
  policy_rate: 5.0,
  inflation: 2.8,
  exchange_rate: 15800,
  news_stress: 0.2,
};

const defaultAdv: ScenarioInput = {
  gdp_growth: 3.0,
  credit_growth: 5.0,
  policy_rate: 6.0,
  inflation: 4.0,
  exchange_rate: 16500,
  news_stress: 0.75,
};

function Field({
  label,
  value,
  onChange,
  step = 0.1,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
}) {
  return (
    <div className="field">
      <label>{label}</label>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

export default function ScenariosPage() {
  const [baseline, setBaseline] = useState(defaultBase);
  const [adverse, setAdverse] = useState(defaultAdv);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    try {
      const res = await apiPost<any>("/api/scenario", { baseline, adverse });
      setResult(res.data);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="stack" style={{ gap: "1rem" }}>
      <div className="topbar">
        <div>
          <h1 className="page-title">Scenario Analysis</h1>
          <p className="page-sub">
            Simulate alternative macro and news-stress paths. Results are model simulations — not
            official forecasts.
          </p>
        </div>
        <DemoBanner show />
      </div>

      <div className="grid grid-2">
        <Panel title="Baseline Inputs">
          <div className="grid grid-2">
            <Field label="GDP Growth %" value={baseline.gdp_growth} onChange={(v) => setBaseline({ ...baseline, gdp_growth: v })} />
            <Field label="Credit Growth %" value={baseline.credit_growth} onChange={(v) => setBaseline({ ...baseline, credit_growth: v })} />
            <Field label="Policy Rate %" value={baseline.policy_rate} onChange={(v) => setBaseline({ ...baseline, policy_rate: v })} />
            <Field label="Inflation %" value={baseline.inflation} onChange={(v) => setBaseline({ ...baseline, inflation: v })} />
            <Field label="USD/IDR" value={baseline.exchange_rate} step={50} onChange={(v) => setBaseline({ ...baseline, exchange_rate: v })} />
            <Field label="News Stress" value={baseline.news_stress} step={0.05} onChange={(v) => setBaseline({ ...baseline, news_stress: v })} />
          </div>
        </Panel>
        <Panel title="Adverse Inputs">
          <div className="grid grid-2">
            <Field label="GDP Growth %" value={adverse.gdp_growth} onChange={(v) => setAdverse({ ...adverse, gdp_growth: v })} />
            <Field label="Credit Growth %" value={adverse.credit_growth} onChange={(v) => setAdverse({ ...adverse, credit_growth: v })} />
            <Field label="Policy Rate %" value={adverse.policy_rate} onChange={(v) => setAdverse({ ...adverse, policy_rate: v })} />
            <Field label="Inflation %" value={adverse.inflation} onChange={(v) => setAdverse({ ...adverse, inflation: v })} />
            <Field label="USD/IDR" value={adverse.exchange_rate} step={50} onChange={(v) => setAdverse({ ...adverse, exchange_rate: v })} />
            <Field label="News Stress" value={adverse.news_stress} step={0.05} onChange={(v) => setAdverse({ ...adverse, news_stress: v })} />
          </div>
        </Panel>
      </div>

      <button className="btn" onClick={run} disabled={loading}>
        {loading ? "Simulating…" : "Run Scenario Simulator"}
      </button>

      {result && (
        <div className="grid grid-2">
          {result.results.map((r: any) => (
            <Panel key={r.label} title={r.label}>
              <div className="metric">{r.estimated_npl.toFixed(2)}%</div>
              <div className="muted" style={{ marginTop: "0.5rem" }}>
                Δ vs baseline: {r.delta_from_baseline >= 0 ? "+" : ""}
                {r.delta_from_baseline.toFixed(2)} pp
              </div>
              <p className="muted" style={{ fontSize: "0.85rem" }}>
                {r.disclaimer}
              </p>
            </Panel>
          ))}
        </div>
      )}
    </div>
  );
}
