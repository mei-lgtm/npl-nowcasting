"""Orchestration service for the NPL nowcasting workflow."""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.core.config import settings
from app.data.schemas.models import (
    AIAnalystResponse,
    DriverContribution,
    NowcastResult,
    ObservationType,
    RiskSignal,
    ScenarioInput,
    ScenarioResult,
    WorkflowConfig,
)
from app.data.synthetic import write_demo_files
from app.engines.evaluation.backtest import auto_select, expanding_backtest
from app.engines.features.engineering import (
    aggregate_news_to_monthly,
    build_feature_matrix,
    check_diagnostics,
    select_features_automatic,
)
from app.engines.nowcasting.models import MODEL_CATALOG, EnsembleModel, build_model
from app.engines.sentiment.engine import SentimentEngine
from app.engines.vintage.engine import VintageEngine


def _data_dir() -> Path:
    p = Path(settings.data_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    return p


class NowcastingService:
    def __init__(self):
        self.sentiment = SentimentEngine()
        self._cache: dict = {}
        self.reload()

    def reload(self):
        d = _data_dir()
        obs_path = d / "observations.parquet"
        news_path = d / "news.parquet"
        if not obs_path.exists():
            write_demo_files(d)
        self.observations = pd.read_parquet(obs_path)
        self.news = pd.read_parquet(news_path)
        meta_path = d / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"mode": "DEMO / SYNTHETIC DATA"}
        self.vintage = VintageEngine(self.observations, self.news)
        self._cache = {}

    @property
    def is_synthetic(self) -> bool:
        return settings.app_mode == "demo" or bool(self.meta.get("mode", "").startswith("DEMO"))

    def _as_of(self, cfg: WorkflowConfig | None = None) -> datetime:
        if cfg and cfg.as_of:
            return cfg.as_of
        # Demo "today"
        return datetime(2026, 9, 14, 12, 0, 0)

    def get_panel(self, as_of: datetime, target: str = "gross_npl_ratio") -> pd.DataFrame:
        panel = self.vintage.panel_as_of(as_of)
        news = self.vintage.available_news(as_of)
        news_m = aggregate_news_to_monthly(news, as_of=pd.Timestamp(as_of))
        if not news_m.empty:
            panel = panel.merge(news_m, on="reference_period", how="left")
        return panel

    def build_features(self, cfg: WorkflowConfig) -> tuple[pd.DataFrame, list[str], dict]:
        as_of = self._as_of(cfg)
        panel = self.get_panel(as_of, cfg.target)
        feat = build_feature_matrix(panel, target=cfg.target)
        diagnostics = check_diagnostics(feat, cfg.target)

        if cfg.variable_selection == "manual":
            features = [c for c in cfg.selected_indicators if c in feat.columns]
            if not features:
                features = select_features_automatic(feat, cfg.target, mandatory=cfg.mandatory_indicators)
        elif cfg.variable_selection == "hybrid":
            features = select_features_automatic(
                feat, cfg.target, mandatory=list(dict.fromkeys(cfg.mandatory_indicators + cfg.selected_indicators))
            )
        else:
            features = select_features_automatic(feat, cfg.target, mandatory=cfg.mandatory_indicators)

        return feat, features, diagnostics

    def run_backtest(self, cfg: WorkflowConfig) -> dict:
        feat, features, diagnostics = self.build_features(cfg)
        # If UI/API supplies a candidate list, respect it (even in auto mode).
        if cfg.selected_models:
            candidates = cfg.selected_models
        elif cfg.model_selection == "manual":
            candidates = cfg.selected_models or ["ridge", "var", "arimax"]
        else:
            candidates = None
        result = auto_select(
            feat,
            cfg.target,
            features,
            candidates=candidates,
            mode=cfg.backtest_mode,
            rolling_window=cfg.rolling_window,
        )
        result["features"] = features
        result["diagnostics"] = diagnostics
        result["is_synthetic"] = self.is_synthetic
        self._cache["last_selection"] = result
        self._cache["features"] = features
        self._cache["feat_df"] = feat
        self._cache["cfg"] = cfg
        return result

    def _risk_signal(self, nowcast: float, last_actual: float | None, change: float | None) -> RiskSignal:
        level = nowcast
        delta = abs(change or 0)
        if level >= 4.0 or delta >= 0.40:
            return RiskSignal.CRITICAL
        if level >= 3.5 or delta >= 0.28:
            return RiskSignal.HIGH
        if level >= 3.0 or delta >= 0.15:
            return RiskSignal.ELEVATED
        if level >= 2.7 or delta >= 0.08:
            return RiskSignal.MODERATE
        return RiskSignal.LOW

    def _confidence(self, rmse: float | None, missing_hf: bool) -> str:
        if rmse is None:
            return "low"
        if rmse < 0.12 and not missing_hf:
            return "high"
        if rmse < 0.20:
            return "moderate"
        return "low"

    def generate_nowcast(self, cfg: WorkflowConfig | None = None) -> NowcastResult:
        cfg = cfg or WorkflowConfig()
        as_of = self._as_of(cfg)

        if "last_selection" not in self._cache:
            self.run_backtest(cfg)

        selection = self._cache["last_selection"]
        feat: pd.DataFrame = self._cache["feat_df"]
        features: list[str] = self._cache["features"]

        latest_actual = self.vintage.latest_actual_npl(as_of, cfg.target)
        unreleased = self.vintage.unreleased_periods(as_of, cfg.target)

        if cfg.model_selection == "manual" and cfg.selected_models:
            model_name = cfg.selected_models[0]
        elif cfg.model_selection == "ensemble":
            model_name = "ensemble"
        else:
            model_name = selection.get("best_model") or "xgboost"

        # Train on all actuals
        known = feat.dropna(subset=[cfg.target]).copy()
        y = known.set_index("reference_period")[cfg.target]
        X = known.set_index("reference_period")[[c for c in features if c in known.columns]]

        # Future/nowcast rows: periods without actual
        nowcast_period = unreleased[0] if unreleased else as_of.strftime("%Y-%m")
        # Build feature row for nowcast period from latest available indicators
        future_rows = feat[feat["reference_period"] >= nowcast_period].copy()
        if future_rows.empty or nowcast_period not in set(feat["reference_period"]):
            # synthesize row from last known + latest indicators
            last = feat.iloc[[-1]].copy()
            last["reference_period"] = nowcast_period
            last[cfg.target] = np.nan
            future_rows = last

        X_future = future_rows.set_index("reference_period")[[c for c in features if c in future_rows.columns]]
        # forward-fill from history
        X_all = pd.concat([X, X_future]).sort_index()
        X_all = X_all.ffill().fillna(X.median())
        X_future = X_all.loc[[nowcast_period]] if nowcast_period in X_all.index else X_all.iloc[[-1]]

        if model_name == "ensemble":
            weights = selection.get("ensemble_weights") or None
            members = list((weights or {"dfm": 0.3, "xgboost": 0.3, "arimax": 0.2, "midas_news": 0.2}).keys())
            model = EnsembleModel("ensemble", members=members, weights=weights)
        else:
            model = build_model(model_name)

        if model_name in ("arima", "sarima", "state_space"):
            model.fit(y, None)
            result = model.predict(steps=1)
        else:
            try:
                model.fit(y, X)
                result = model.predict(steps=1, X_future=X_future)
            except Exception:
                # Fall back to next-best leaderboard model
                for alt in selection.get("leaderboard") or []:
                    alt_name = alt.get("model")
                    if not alt_name or alt_name == model_name:
                        continue
                    try:
                        model = build_model(alt_name)
                        model_name = alt_name
                        if alt_name in ("arima", "sarima", "state_space"):
                            model.fit(y, None)
                            result = model.predict(steps=1)
                        else:
                            model.fit(y, X)
                            result = model.predict(steps=1, X_future=X_future)
                        break
                    except Exception:
                        continue
                else:
                    model = build_model("ridge")
                    model_name = "ridge"
                    model.fit(y, X)
                    result = model.predict(steps=1, X_future=X_future)

        value = float(result.predictions[0])
        # Confidence interval from backtest RMSE
        best = next((r for r in selection.get("leaderboard", []) if r["model"] == model_name), None)
        if best is None and selection.get("leaderboard"):
            best = selection["leaderboard"][0]
        rmse = best["rmse"] if best else 0.15
        ci_low = value - 1.96 * rmse
        ci_high = value + 1.96 * rmse

        last_val = latest_actual["value"] if latest_actual else None
        change = (value - last_val) if last_val is not None else None

        # Driver contributions via prediction decomposition (approx)
        drivers = self._explain_drivers(model, X, X_future, features, result.feature_importance, change)

        # News signals
        news = self.vintage.available_news(as_of)
        recent = news[pd.to_datetime(news["published_at"]) >= pd.Timestamp(as_of) - pd.Timedelta(days=30)]
        news_signals = []
        if not recent.empty:
            neg = recent[recent["sentiment"] == "negative"].head(4)
            news_signals = [str(h) for h in neg["headline"].tolist()]

        missing_hf = float(X_future.isna().mean().mean()) > 0.2 if len(X_future.columns) else True

        out = NowcastResult(
            target=cfg.target,
            reference_period=nowcast_period,
            nowcast_value=round(value, 4),
            actual_last_release=round(last_val, 4) if last_val is not None else None,
            actual_last_period=latest_actual["period"] if latest_actual else None,
            change_pp=round(change, 4) if change is not None else None,
            confidence_low=round(ci_low, 4),
            confidence_high=round(ci_high, 4),
            risk_signal=self._risk_signal(value, last_val, change),
            observation_type=ObservationType.NOWCAST,
            model_used=model_name if model_name != "ensemble" else f"ensemble:{','.join(members) if model_name == 'ensemble' else model_name}",
            model_confidence=self._confidence(rmse, missing_hf),  # type: ignore
            drivers=drivers,
            news_signals=news_signals,
            is_synthetic=self.is_synthetic,
        )
        # Fix ensemble model_used string
        if cfg.model_selection == "ensemble" or model_name == "ensemble":
            out.model_used = "ensemble(" + ",".join((selection.get("ensemble_weights") or {}).keys()) + ")"
        self._cache["last_nowcast"] = out
        self._cache["X_future"] = X_future
        self._cache["model"] = model
        return out

    def _explain_drivers(self, model, X, X_future, features, importance, change) -> list[DriverContribution]:
        drivers: list[DriverContribution] = []
        if importance:
            # Normalize importance to approximate pp contributions
            items = sorted(importance.items(), key=lambda kv: -abs(kv[1]))[:8]
            total = sum(abs(v) for _, v in items) or 1.0
            delta = change if change is not None else 0.1
            for name, val in items[:5]:
                share = abs(val) / total
                contrib = float(np.sign(val) * share * abs(delta)) if delta else float(val)
                # Compare last vs current feature if available
                direction = "up" if contrib >= 0 else "down"
                drivers.append(
                    DriverContribution(
                        feature=name,
                        contribution_pp=round(contrib, 4),
                        direction=direction,
                        note="Model contribution / association — not a causal claim.",
                    )
                )
            return drivers

        # Fallback: largest feature moves
        if X is not None and X_future is not None and not X.empty:
            last = X.iloc[-1]
            cur = X_future.iloc[0]
            moves = (cur - last).abs().sort_values(ascending=False).head(5)
            for name, _ in moves.items():
                raw = float(cur[name] - last[name])
                drivers.append(
                    DriverContribution(
                        feature=str(name),
                        contribution_pp=round(raw * 0.02, 4),
                        direction="up" if raw >= 0 else "down",
                    )
                )
        return drivers

    def timeseries(self, cfg: WorkflowConfig | None = None) -> list[dict]:
        cfg = cfg or WorkflowConfig()
        as_of = self._as_of(cfg)
        if "last_nowcast" not in self._cache:
            self.generate_nowcast(cfg)
        nowcast = self._cache["last_nowcast"]
        selection = self._cache.get("last_selection", {})

        panel = self.get_panel(as_of, cfg.target)
        points = []
        npl = panel[["reference_period", cfg.target]].dropna()
        for _, row in npl.iterrows():
            points.append(
                {
                    "period": row["reference_period"],
                    "value": float(row[cfg.target]),
                    "observation_type": ObservationType.ACTUAL.value,
                    "low": None,
                    "high": None,
                }
            )

        # Historical nowcasts from backtest of best model
        board = selection.get("leaderboard") or []
        best = next((r for r in board if r["model"] == (selection.get("best_model"))), board[0] if board else None)
        if best:
            for p in best.get("predictions", [])[-18:]:
                # Don't overwrite actuals — add nowcast track as separate series points with type nowcast_eval
                points.append(
                    {
                        "period": p["period"],
                        "value": p["nowcast"],
                        "observation_type": "nowcast_eval",
                        "low": None,
                        "high": None,
                        "actual": p["actual"],
                    }
                )

        points.append(
            {
                "period": nowcast.reference_period,
                "value": nowcast.nowcast_value,
                "observation_type": ObservationType.NOWCAST.value,
                "low": nowcast.confidence_low,
                "high": nowcast.confidence_high,
            }
        )

        # One-step forecast beyond nowcast (explicitly labeled)
        try:
            from dateutil.relativedelta import relativedelta

            nxt = (pd.Period(nowcast.reference_period, freq="M") + 1).strftime("%Y-%m")
        except Exception:
            nxt = nowcast.reference_period
        # Simple persistence forecast for illustration of distinction
        points.append(
            {
                "period": nxt,
                "value": round(nowcast.nowcast_value + 0.02, 4),
                "observation_type": ObservationType.FORECAST.value,
                "low": round(nowcast.confidence_low + 0.02, 4),
                "high": round(nowcast.confidence_high + 0.05, 4),
            }
        )
        return points

    def news_dashboard(self, cfg: WorkflowConfig | None = None) -> dict:
        cfg = cfg or WorkflowConfig()
        as_of = self._as_of(cfg)
        news = self.vintage.available_news(as_of)
        news["published_at"] = pd.to_datetime(news["published_at"])

        # Monthly stress series
        monthly = aggregate_news_to_monthly(news, as_of=pd.Timestamp(as_of))
        recent_30 = news[news["published_at"] >= pd.Timestamp(as_of) - pd.Timedelta(days=30)]
        articles = recent_30.to_dict(orient="records")
        idx = self.sentiment.news_stress_index(articles, model=cfg.sentiment_model)

        # Sector heatmap
        sector = []
        if not recent_30.empty and "sector" in recent_30.columns:
            tmp = recent_30.copy()
            if "stress_contribution" in tmp.columns:
                tmp["_s"] = tmp["stress_contribution"]
            else:
                tmp["_s"] = (tmp["sentiment_score"] * 2 - 1) * tmp["npl_relevance"]
            g = tmp.groupby("sector")["_s"].mean().sort_values(ascending=False)
            sector = [{"sector": k, "stress": float(v)} for k, v in g.items()]

        series = [
            {"period": r["reference_period"], "news_stress": float(r["news_stress"]), "count": int(r["news_count"])}
            for _, r in monthly.tail(36).iterrows()
        ]
        prev = series[-2]["news_stress"] if len(series) >= 2 else 0.0
        curr = series[-1]["news_stress"] if series else idx["news_stress_index"]

        return {
            "current": curr,
            "previous": prev,
            "change": curr - prev,
            "article_count": idx["article_count"],
            "negative_ratio": idx["negative_ratio"],
            "top_topics": idx["top_topics"],
            "series": series,
            "sector_heatmap": sector,
            "recent_articles": [
                {
                    "id": a["id"],
                    "headline": a["headline"],
                    "published_at": str(a["published_at"]),
                    "source": a["source"],
                    "url": a.get("url"),
                    "sentiment": a.get("sentiment"),
                    "sector": a.get("sector"),
                    "npl_relevance": a.get("npl_relevance"),
                    "is_synthetic": True,
                }
                for a in articles[-20:][::-1]
            ],
            "is_synthetic": self.is_synthetic,
            "disclaimer": "DEMO / SYNTHETIC DATA" if self.is_synthetic else None,
        }

    def scenario(self, baseline: ScenarioInput, adverse: ScenarioInput | None = None) -> dict:
        if "last_nowcast" not in self._cache:
            self.generate_nowcast(WorkflowConfig())
        nowcast = self._cache["last_nowcast"]
        # Linear sensitivity around current nowcast (explicitly a simulation)
        def estimate(inp: ScenarioInput) -> float:
            base = nowcast.nowcast_value
            # Coefficients are illustrative associations for simulation
            delta = (
                -0.06 * (inp.gdp_growth - 5.0)
                - 0.04 * (inp.credit_growth - 8.0)
                + 0.08 * (inp.policy_rate - 5.0)
                + 0.05 * (inp.inflation - 2.8)
                + 0.00002 * (inp.exchange_rate - 15800)
                + 0.35 * (inp.news_stress - 0.2)
            )
            return round(base + delta, 4)

        base_est = estimate(baseline)
        results = [
            ScenarioResult(
                label="BASELINE",
                inputs=baseline,
                estimated_npl=base_est,
                delta_from_baseline=0.0,
            )
        ]
        if adverse:
            adv_est = estimate(adverse)
            results.append(
                ScenarioResult(
                    label="ADVERSE",
                    inputs=adverse,
                    estimated_npl=adv_est,
                    delta_from_baseline=round(adv_est - base_est, 4),
                )
            )
        return {
            "results": [r.model_dump() for r in results],
            "disclaimer": "Model simulations only — not official forecasts.",
            "is_synthetic": self.is_synthetic,
        }

    def data_quality(self) -> list[dict]:
        as_of = self._as_of()
        obs = self.vintage.available_observations(as_of)
        items = []
        for ind, g in obs.groupby("indicator"):
            g = g.sort_values("reference_period")
            periods = pd.period_range(g["reference_period"].min(), g["reference_period"].max(), freq="M").astype(str)
            missing = 1 - (g["reference_period"].nunique() / max(len(periods), 1))
            vals = g["value"]
            z = (vals - vals.mean()) / (vals.std() + 1e-8)
            outliers = int((z.abs() > 3).sum())
            last = g.iloc[-1]
            pub = pd.to_datetime(last["publication_date"])
            ref = pd.Period(last["reference_period"], freq="M").to_timestamp("M")
            lag = int((pub - ref).days)
            items.append(
                {
                    "indicator": ind,
                    "status": "ok" if missing < 0.1 else "warning",
                    "frequency": str(last["frequency"]),
                    "last_update": last["reference_period"],
                    "publication_lag_days": max(lag, 0),
                    "missing_pct": round(float(missing) * 100, 2),
                    "outlier_count": outliers,
                    "source": last["source"],
                    "is_synthetic": True,
                }
            )
        # News
        news = self.vintage.available_news(as_of)
        items.append(
            {
                "indicator": "news",
                "status": "ok",
                "frequency": "daily",
                "last_update": str(pd.to_datetime(news["published_at"]).max().date()) if not news.empty else "n/a",
                "publication_lag_days": 0,
                "missing_pct": 0.0,
                "outlier_count": 0,
                "source": "Demo news corpus",
                "is_synthetic": True,
            }
        )
        return items

    def indicators_latest(self) -> list[dict]:
        as_of = self._as_of()
        panel = self.get_panel(as_of)
        if panel.empty:
            return []
        last = panel.iloc[-1]
        out = []
        for c in panel.columns:
            if c == "reference_period":
                continue
            val = last[c]
            if pd.isna(val):
                continue
            out.append({"indicator": c, "value": float(val), "period": last["reference_period"], "is_synthetic": True})
        return out

    def ai_analyst(self, cfg: WorkflowConfig | None = None) -> AIAnalystResponse:
        cfg = cfg or WorkflowConfig()
        if "last_nowcast" not in self._cache:
            self.generate_nowcast(cfg)
        nc = self._cache["last_nowcast"]
        news = self.news_dashboard(cfg)
        drivers = nc.drivers[:3]
        driver_txt = ", ".join(f"{d.feature} ({d.contribution_pp:+.2f} pp)" for d in drivers) or "limited feature attribution"
        sectors = news.get("sector_heatmap") or []
        top_sec = ", ".join(s["sector"] for s in sectors[:2]) if sectors else "several sectors"

        change = nc.change_pp or 0.0
        direction = "increasing" if change > 0 else "easing" if change < 0 else "broadly stable"
        summary = (
            f"NPL is estimated at {nc.nowcast_value:.2f}% for {nc.reference_period}, "
            f"{direction} from the last actual release"
            + (f" of {nc.actual_last_release:.2f}% ({nc.actual_last_period})." if nc.actual_last_release else ".")
        )
        bullets = [
            f"Nowcast change versus last actual: {change:+.2f} percentage points.",
            f"Primary model associations: {driver_txt}.",
            f"News Stress Index is {news['current']:.2f} (change {news['change']:+.2f}); negative news share {news['negative_ratio']*100:.0f}%.",
            f"Highest news-based sector stress currently: {top_sec}.",
            f"Model used: {nc.model_used}; confidence: {nc.model_confidence}.",
            f"Risk signal: {nc.risk_signal.value}.",
        ]
        risks = [
            "Interpretation reflects statistical association, not proven causation.",
            "High-frequency indicators may still be incomplete for the current reference period.",
        ]
        if nc.news_signals:
            risks.append("Recent headlines highlight: " + "; ".join(nc.news_signals[:2]))

        return AIAnalystResponse(
            summary=summary,
            bullets=bullets,
            risks=risks,
            confidence_note=(
                f"Confidence interval {nc.confidence_low:.2f}% – {nc.confidence_high:.2f}%. "
                "Nowcast is an estimate for an unreleased period, not an official statistic."
            ),
            is_synthetic=self.is_synthetic,
        )

    def track_record(self) -> dict:
        if "last_selection" not in self._cache:
            self.run_backtest(WorkflowConfig())
        sel = self._cache["last_selection"]
        best = sel.get("best_model")
        board = sel.get("leaderboard") or []
        row = next((r for r in board if r["model"] == best), board[0] if board else None)
        if not row:
            return {"records": [], "metrics": None, "is_synthetic": self.is_synthetic}
        return {
            "model": row["model"],
            "metrics": {
                "rmse": row["rmse"],
                "mae": row["mae"],
                "mape": row["mape"],
                "bias": row.get("bias"),
                "directional_accuracy": row.get("directional_accuracy"),
            },
            "records": row.get("predictions", [])[-24:],
            "is_synthetic": self.is_synthetic,
        }

    def catalog(self) -> dict:
        return {
            "models": MODEL_CATALOG,
            "sentiment_models": SentimentEngine.AVAILABLE,
            "targets": [
                "gross_npl_ratio",
                "net_npl_ratio",
                "npl_by_bank_group",
                "npl_by_credit_segment",
                "npl_by_sector",
                "npl_by_loan_type",
            ],
            "mode": settings.app_mode,
            "is_synthetic": self.is_synthetic,
            "disclaimer": self.meta.get("disclaimer"),
        }


@lru_cache
def get_service() -> NowcastingService:
    return NowcastingService()
