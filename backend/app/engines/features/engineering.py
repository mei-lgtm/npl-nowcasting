"""Feature engineering for NPL nowcasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


LAGS = [1, 2, 3, 6, 12]
ROLL_WINDOWS = [3, 6, 12]


def add_lags(df: pd.DataFrame, columns: list[str], lags: list[int] = LAGS) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_changes(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[f"{col}_mom"] = out[col].diff(1)
        out[f"{col}_yoy"] = out[col].diff(12)
        out[f"{col}_qoq"] = out[col].diff(3)
    return out


def add_rolling(df: pd.DataFrame, columns: list[str], windows: list[int] = ROLL_WINDOWS) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        for w in windows:
            out[f"{col}_ma{w}"] = out[col].rolling(w, min_periods=max(1, w // 2)).mean()
            out[f"{col}_vol{w}"] = out[col].rolling(w, min_periods=max(1, w // 2)).std()
            mean = out[col].rolling(w, min_periods=max(1, w // 2)).mean()
            std = out[col].rolling(w, min_periods=max(1, w // 2)).std()
            out[f"{col}_z{w}"] = (out[col] - mean) / (std + 1e-8)
    return out


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pairs = [
        ("policy_rate", "credit_growth"),
        ("lending_rate", "credit_growth"),
        ("gdp_growth", "news_stress"),
        ("exchange_rate_vol", "news_stress"),
        ("special_mention", "news_stress"),
    ]
    for a, b in pairs:
        if a in out.columns and b in out.columns:
            out[f"{a}_x_{b}"] = out[a] * out[b]
    return out


def aggregate_news_to_monthly(news: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    if news is None or news.empty:
        return pd.DataFrame(columns=["reference_period", "news_stress", "news_count", "negative_ratio", "news_momentum"])

    df = news.copy()
    df["published_at"] = pd.to_datetime(df["published_at"])
    if as_of is not None:
        df = df[df["published_at"] <= pd.Timestamp(as_of)]
    if df.empty:
        return pd.DataFrame(columns=["reference_period", "news_stress", "news_count", "negative_ratio", "news_momentum"])

    df["reference_period"] = df["published_at"].dt.to_period("M").astype(str)
    # Stress: map sentiment_score so higher = more stress for negative news
    if "stress_contribution" in df.columns:
        raw = df["stress_contribution"]
    else:
        sign = df["sentiment"].map({"negative": 1.0, "neutral": 0.0, "positive": -1.0}).fillna(0.0)
        raw = sign * df.get("sentiment_score", 0.5) * df.get("npl_relevance", 0.5)

    tmp = df.copy()
    tmp["_stress"] = raw.values if hasattr(raw, "values") else raw
    agg = tmp.groupby("reference_period").agg(
        news_stress=("_stress", "mean"),
        news_count=("published_at", "count"),
        negative_ratio=("sentiment", lambda s: (s == "negative").mean()),
        avg_sentiment=("sentiment_score", "mean"),
    ).reset_index()
    agg["news_stress"] = agg["news_stress"].clip(-1, 1)
    agg["news_momentum"] = agg["news_stress"].diff()
    if "sector" in tmp.columns:
        for sector in tmp["sector"].dropna().unique():
            sec = tmp[tmp["sector"] == sector].groupby("reference_period")["_stress"].mean()
            col = f"sector_stress_{str(sector).lower().replace(' ', '_')}"
            agg = agg.merge(sec.rename(col), left_on="reference_period", right_index=True, how="left")
    return agg


def build_feature_matrix(
    panel: pd.DataFrame,
    target: str = "gross_npl_ratio",
    news_monthly: pd.DataFrame | None = None,
    base_cols: list[str] | None = None,
) -> pd.DataFrame:
    df = panel.copy()
    if "reference_period" in df.columns:
        df = df.sort_values("reference_period").reset_index(drop=True)

    if news_monthly is not None and not news_monthly.empty:
        df = df.merge(news_monthly, on="reference_period", how="left")

    if base_cols is None:
        base_cols = [c for c in df.columns if c not in ("reference_period", target) and pd.api.types.is_numeric_dtype(df[c])]

    # Ensure news_stress in base if present
    feature_bases = [c for c in base_cols if c in df.columns]
    df = add_lags(df, [target] + feature_bases[:12], lags=[1, 2, 3, 6, 12])
    df = add_changes(df, feature_bases[:8])
    df = add_rolling(df, feature_bases[:6], windows=[3, 6, 12])
    df = add_interactions(df)
    return df


def select_features_automatic(
    df: pd.DataFrame,
    target: str,
    max_features: int = 25,
    mandatory: list[str] | None = None,
) -> list[str]:
    """Correlation + mutual information + Lasso hybrid selection."""
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    mandatory = mandatory or []
    work = df.dropna(subset=[target]).copy()
    feature_cols = [
        c
        for c in work.columns
        if c not in (target, "reference_period")
        and pd.api.types.is_numeric_dtype(work[c])
        and work[c].notna().mean() > 0.7
    ]
    if not feature_cols:
        return mandatory

    X = work[feature_cols].fillna(work[feature_cols].median())
    y = work[target]

    # Correlation filter
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    top_corr = set(corr.head(40).index)

    # Mutual information
    mi = mutual_info_regression(X[list(top_corr)], y, random_state=42)
    mi_rank = pd.Series(mi, index=list(top_corr)).sort_values(ascending=False)
    top_mi = set(mi_rank.head(30).index)

    # Lasso
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X[list(top_mi)])
    try:
        lasso = LassoCV(cv=min(5, max(2, len(y) // 10)), random_state=42, max_iter=5000)
        lasso.fit(Xs, y)
        coef = pd.Series(np.abs(lasso.coef_), index=list(top_mi))
        top_lasso = set(coef[coef > 1e-6].sort_values(ascending=False).head(max_features).index)
    except Exception:
        top_lasso = set(mi_rank.head(max_features).index)

    selected = list(dict.fromkeys(mandatory + list(top_lasso) + list(mi_rank.head(10).index)))
    # Prefer lag of target if present
    for lag in (1, 2, 3):
        col = f"{target}_lag{lag}"
        if col in feature_cols and col not in selected:
            selected.insert(0, col)
    return selected[:max_features]


def check_diagnostics(df: pd.DataFrame, target: str) -> dict:
    """Basic econometric diagnostics summary."""
    from statsmodels.tsa.stattools import adfuller

    out: dict = {"stationarity": {}, "missing": {}, "multicollinearity_hint": None}
    series = df[target].dropna()
    if len(series) >= 20:
        try:
            adf = adfuller(series, autolag="AIC")
            out["stationarity"][target] = {
                "adf_stat": float(adf[0]),
                "pvalue": float(adf[1]),
                "is_stationary_5pct": bool(adf[1] < 0.05),
            }
        except Exception as e:
            out["stationarity"][target] = {"error": str(e)}

    for col in df.columns:
        if col == "reference_period":
            continue
        out["missing"][col] = float(df[col].isna().mean())

    numeric = df.select_dtypes(include=[np.number]).dropna()
    if numeric.shape[1] >= 2 and len(numeric) > 10:
        corr = numeric.corr().abs()
        high = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr.iloc[i, j] > 0.95:
                    high.append((cols[i], cols[j], float(corr.iloc[i, j])))
        out["multicollinearity_hint"] = high[:20]
    return out
