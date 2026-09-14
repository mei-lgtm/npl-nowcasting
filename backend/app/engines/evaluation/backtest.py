"""Backtesting and automatic model selection with temporal CV only."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from app.engines.nowcasting.models import MODEL_CATALOG, build_model, compute_metrics


DEFAULT_CANDIDATES = [
    "arima",
    "arimax",
    "var",
    "bvar",
    "dynamic_regression",
    "dfm",
    "midas_news",
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest",
    "xgboost",
    "hybrid_dfm_xgb",
]


def _prepare_xy(df: pd.DataFrame, target: str, features: list[str]):
    cols = [c for c in features if c in df.columns]
    work = df[["reference_period", target] + cols].copy()
    work = work.sort_values("reference_period").reset_index(drop=True)
    return work, cols


def expanding_backtest(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model_name: str,
    min_train: int = 36,
) -> dict:
    work, cols = _prepare_xy(df, target, features)
    # Only rows where target is known (actuals)
    known = work.dropna(subset=[target]).reset_index(drop=True)
    preds, actuals, periods = [], [], []
    # Cap folds for interactive UI latency (still temporal, never shuffled)
    idxs = list(range(min_train, len(known)))
    if len(idxs) > 18:
        step = max(1, len(idxs) // 14)
        idxs = idxs[::step][:16]

    for t in idxs:
        train = known.iloc[:t]
        test_row = known.iloc[[t]]
        y = train.set_index("reference_period")[target]
        X = train.set_index("reference_period")[cols] if cols else None
        X_future = test_row.set_index("reference_period")[cols] if cols else None
        try:
            model = build_model(model_name)
            # Models needing X
            if model_name in ("arima", "sarima", "state_space"):
                model.fit(y, None)
                r = model.predict(steps=1)
            else:
                if X is None or X.empty:
                    continue
                model.fit(y, X)
                r = model.predict(steps=1, X_future=X_future)
            preds.append(float(r.predictions[0]))
            actuals.append(float(test_row[target].iloc[0]))
            periods.append(str(test_row["reference_period"].iloc[0]))
        except Exception:
            continue

    if len(preds) < 3:
        return {
            "model": model_name,
            "metrics": None,
            "error": "Insufficient successful backtest points",
            "predictions": [],
        }

    metrics = compute_metrics(actuals, preds)
    return {
        "model": model_name,
        "display_name": MODEL_CATALOG.get(model_name, {}).get("name", model_name),
        "metrics": metrics,
        "predictions": [
            {"period": p, "actual": a, "nowcast": pr, "error": pr - a}
            for p, a, pr in zip(periods, actuals, preds)
        ],
    }


def rolling_backtest(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    model_name: str,
    window: int = 36,
) -> dict:
    work, cols = _prepare_xy(df, target, features)
    known = work.dropna(subset=[target]).reset_index(drop=True)
    preds, actuals, periods = [], [], []

    for t in range(window, len(known)):
        train = known.iloc[t - window : t]
        test_row = known.iloc[[t]]
        y = train.set_index("reference_period")[target]
        X = train.set_index("reference_period")[cols] if cols else None
        X_future = test_row.set_index("reference_period")[cols] if cols else None
        try:
            model = build_model(model_name)
            if model_name in ("arima", "sarima", "state_space"):
                model.fit(y, None)
                r = model.predict(steps=1)
            else:
                model.fit(y, X)
                r = model.predict(steps=1, X_future=X_future)
            preds.append(float(r.predictions[0]))
            actuals.append(float(test_row[target].iloc[0]))
            periods.append(str(test_row["reference_period"].iloc[0]))
        except Exception:
            continue

    if len(preds) < 3:
        return {"model": model_name, "metrics": None, "error": "Insufficient points", "predictions": []}

    metrics = compute_metrics(actuals, preds)
    return {
        "model": model_name,
        "display_name": MODEL_CATALOG.get(model_name, {}).get("name", model_name),
        "metrics": metrics,
        "predictions": [
            {"period": p, "actual": a, "nowcast": pr, "error": pr - a}
            for p, a, pr in zip(periods, actuals, preds)
        ],
    }


def rank_models(results: list[dict]) -> list[dict]:
    """Rank by out-of-sample RMSE then MAE. Never by in-sample R² alone."""
    valid = [r for r in results if r.get("metrics")]
    valid.sort(key=lambda r: (r["metrics"]["rmse"], r["metrics"]["mae"]))
    ranked = []
    for i, r in enumerate(valid, start=1):
        m = r["metrics"]
        ranked.append(
            {
                "model": r["model"],
                "display_name": r.get("display_name", r["model"]),
                "rmse": m["rmse"],
                "mae": m["mae"],
                "mape": m["mape"],
                "r2": m["r2"],
                "bias": m.get("bias"),
                "directional_accuracy": m.get("directional_accuracy"),
                "rank": i,
                "n_observations": m["n_observations"],
                "predictions": r.get("predictions", []),
            }
        )
    return ranked


def auto_select(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    candidates: Optional[list[str]] = None,
    mode: str = "expanding",
    rolling_window: int = 36,
) -> dict:
    candidates = candidates or DEFAULT_CANDIDATES
    # Skip DL if sample too small
    n = df[target].dropna().shape[0]
    filtered = []
    for c in candidates:
        fam = MODEL_CATALOG.get(c, {}).get("family")
        if fam == "dl" and n < 120:
            continue
        filtered.append(c)

    results = []
    for name in filtered:
        if mode == "rolling":
            results.append(rolling_backtest(df, target, features, name, window=rolling_window))
        else:
            results.append(expanding_backtest(df, target, features, name))

    ranked = rank_models(results)
    best = ranked[0] if ranked else None

    # Ensemble weights from inverse RMSE
    weights = {}
    if ranked:
        inv = np.array([1 / max(r["rmse"], 1e-6) for r in ranked[:4]])
        inv = inv / inv.sum()
        for r, w in zip(ranked[:4], inv):
            weights[r["model"]] = float(w)

    return {
        "leaderboard": ranked,
        "best_model": best["model"] if best else None,
        "ensemble_weights": weights,
        "failed": [r["model"] for r in results if not r.get("metrics")],
    }
