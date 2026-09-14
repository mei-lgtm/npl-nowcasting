"use client";

import { useState } from "react";
import {
  DEFAULT_CONFIG,
  apiPost,
  type WorkflowConfig,
  type WorkflowResponse,
} from "@/lib/api";
import { DemoBanner, Panel, RiskChip } from "@/components/ui";

const MODEL_OPTIONS = [
  "arima", "sarima", "arimax", "var", "bvar", "state_space", "dfm", "midas", "midas_news",
  "linear", "ridge", "lasso", "elastic_net", "random_forest", "xgboost", "lightgbm", "catboost", "svr",
  "lstm", "gru", "transformer_ts", "hybrid_arimax_xgb", "hybrid_dfm_xgb",
];

const SENTIMENT_OPTIONS = [
  "tfidf_logistic", "tfidf_svm", "indobert", "mbert", "xlm_roberta",
  "llm_zero_shot", "llm_structured", "ensemble",
];

export default function RunPage() {
  const [config, setConfig] = useState<WorkflowConfig>({ ...DEFAULT_CONFIG });
  const [result, setResult] = useState<WorkflowResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toggleModel(id: string) {
    setConfig((c) => {
      const has = c.selected_models.includes(id);
      return {
        ...c,
        selected_models: has ? c.selected_models.filter((m) => m !== id) : [...c.selected_models, id],
      };
    });
  }

  async function run() {
    setLoading(true);
    setError(null);
    try {
      const res = await apiPost<WorkflowResponse>("/api/workflow/run", { config });
      setResult(res.data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="stack" style={{ gap: "1rem" }}>
      <div className="topbar">
        <div>
          <h1 className="page-title">Run Workflow</h1>
          <p className="page-sub">
            Select target, sentiment model, variable mode, and nowcasting strategy — then run the
            full vintage-aligned pipeline.
          </p>
        </div>
        <DemoBanner show />
      </div>

      <div className="grid grid-3">
        <div className="field">
          <label>Target NPL</label>
          <select
            value={config.target}
            onChange={(e) => setConfig({ ...config, target: e.target.value })}
          >
            <option value="gross_npl_ratio">Gross NPL Ratio (%)</option>
            <option value="net_npl_ratio">Net NPL Ratio (%)</option>
            <option value="npl_by_bank_group">NPL by Bank Group</option>
            <option value="npl_by_credit_segment">NPL by Credit Segment</option>
            <option value="npl_by_sector">NPL by Economic Sector</option>
            <option value="npl_by_loan_type">NPL by Loan Type</option>
          </select>
        </div>
        <div className="field">
          <label>Model Selection</label>
          <select
            value={config.model_selection}
            onChange={(e) =>
              setConfig({ ...config, model_selection: e.target.value as WorkflowConfig["model_selection"] })
            }
          >
            <option value="auto">Auto Select Best Model</option>
            <option value="manual">Manual</option>
            <option value="ensemble">Ensemble</option>
          </select>
        </div>
        <div className="field">
          <label>Sentiment Model</label>
          <select
            value={config.sentiment_model}
            onChange={(e) => setConfig({ ...config, sentiment_model: e.target.value })}
          >
            {SENTIMENT_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Variable Selection</label>
          <select
            value={config.variable_selection}
            onChange={(e) =>
              setConfig({
                ...config,
                variable_selection: e.target.value as WorkflowConfig["variable_selection"],
              })
            }
          >
            <option value="automatic">Automatic</option>
            <option value="manual">Manual</option>
            <option value="hybrid">Hybrid</option>
          </select>
        </div>
        <div className="field">
          <label>Backtest Mode</label>
          <select
            value={config.backtest_mode}
            onChange={(e) =>
              setConfig({ ...config, backtest_mode: e.target.value as WorkflowConfig["backtest_mode"] })
            }
          >
            <option value="expanding">Expanding Window</option>
            <option value="rolling">Rolling Window</option>
          </select>
        </div>
        <div className="field">
          <label>Frequency</label>
          <select
            value={config.frequency}
            onChange={(e) => setConfig({ ...config, frequency: e.target.value })}
          >
            <option value="monthly">Monthly</option>
            <option value="quarterly">Quarterly</option>
          </select>
        </div>
      </div>

      {config.model_selection === "manual" && (
        <Panel title="Select Models">
          <div className="heatmap">
            {MODEL_OPTIONS.map((m) => {
              const on = config.selected_models.includes(m);
              return (
                <button
                  key={m}
                  className="btn"
                  style={{
                    opacity: on ? 1 : 0.55,
                    borderColor: on ? "rgba(46,196,182,0.7)" : undefined,
                  }}
                  onClick={() => toggleModel(m)}
                  type="button"
                >
                  {m}
                </button>
              );
            })}
          </div>
        </Panel>
      )}

      <button className="btn" onClick={run} disabled={loading}>
        {loading ? (
          <>
            <span className="spinner" /> Running pipeline…
          </>
        ) : (
          "Run Data → Sentiment → Features → Backtest → Nowcast"
        )}
      </button>

      {error && <Panel title="Error">{error}</Panel>}

      {result && (
        <div className="grid grid-2">
          <Panel title="Nowcast Result">
            <div className="metric">{result.nowcast.nowcast_value.toFixed(2)}%</div>
            <div style={{ marginTop: "0.75rem" }}>
              <RiskChip signal={result.nowcast.risk_signal} />
            </div>
            <p className="muted">
              Best / selected model: {result.selection.best_model} · used {result.nowcast.model_used}
            </p>
            <p>{result.analyst.summary}</p>
          </Panel>
          <Panel title="Selected Features">
            <div className="stack">
              {result.selection.features.slice(0, 20).map((f) => (
                <div key={f} className="row">
                  <code style={{ fontSize: "0.8rem" }}>{f}</code>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}
