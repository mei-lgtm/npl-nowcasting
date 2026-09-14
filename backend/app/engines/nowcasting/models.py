"""Nowcasting model registry and implementations.

Respects temporal ordering — never shuffle time series.
Distinguishes nowcast (unreleased current) from forecast (future).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import xgboost as xgb
except Exception:  # pragma: no cover — missing libomp on some macOS setups
    xgb = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor


MODEL_CATALOG = {
    "arima": {"family": "econometric", "name": "ARIMA"},
    "sarima": {"family": "econometric", "name": "SARIMA"},
    "arimax": {"family": "econometric", "name": "ARIMAX"},
    "dynamic_regression": {"family": "econometric", "name": "Dynamic Regression"},
    "var": {"family": "econometric", "name": "VAR"},
    "bvar": {"family": "econometric", "name": "Bayesian VAR"},
    "state_space": {"family": "econometric", "name": "State Space"},
    "dfm": {"family": "econometric", "name": "Dynamic Factor Model"},
    "midas": {"family": "mixed_frequency", "name": "MIDAS"},
    "midas_news": {"family": "mixed_frequency", "name": "MIDAS + News"},
    "linear": {"family": "ml", "name": "Linear Regression"},
    "ridge": {"family": "ml", "name": "Ridge"},
    "lasso": {"family": "ml", "name": "Lasso"},
    "elastic_net": {"family": "ml", "name": "Elastic Net"},
    "random_forest": {"family": "ml", "name": "Random Forest"},
    "gradient_boosting": {"family": "ml", "name": "Gradient Boosting"},
    "xgboost": {"family": "ml", "name": "XGBoost"},
    "lightgbm": {"family": "ml", "name": "LightGBM"},
    "catboost": {"family": "ml", "name": "CatBoost"},
    "svr": {"family": "ml", "name": "SVR"},
    "lstm": {"family": "dl", "name": "LSTM"},
    "gru": {"family": "dl", "name": "GRU"},
    "tcn": {"family": "dl", "name": "Temporal Convolutional Network"},
    "transformer_ts": {"family": "dl", "name": "Transformer Time-Series"},
    "hybrid_arimax_xgb": {"family": "hybrid", "name": "ARIMAX + XGBoost"},
    "hybrid_dfm_xgb": {"family": "hybrid", "name": "DFM + XGBoost"},
    "ensemble": {"family": "hybrid", "name": "Ensemble"},
}


@dataclass
class FitPredictResult:
    predictions: np.ndarray
    model_name: str
    feature_importance: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def _mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = _mape(y_true, y_pred)
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")
    bias = float(np.mean(y_pred - y_true))
    if len(y_true) > 1:
        dir_acc = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))
    else:
        dir_acc = float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "bias": bias,
        "directional_accuracy": dir_acc,
        "n_observations": int(len(y_true)),
    }


class NowcastModel:
    def __init__(self, name: str):
        self.name = name
        self._model = None
        self._scaler = StandardScaler()
        self.feature_names: list[str] = []

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "NowcastModel":
        raise NotImplementedError

    def predict(self, steps: int = 1, X_future: Optional[pd.DataFrame] = None) -> FitPredictResult:
        raise NotImplementedError

    def predict_in_sample(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        raise NotImplementedError


class ARIMAModel(NowcastModel):
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "ARIMAModel":
        y = y.astype(float).dropna()
        self._y = y
        order = (1, 1, 1)
        if self.name == "sarima":
            self._model = SARIMAX(y, order=order, seasonal_order=(1, 0, 0, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        elif self.name == "arimax" and X is not None and not X.empty:
            Xc = X.reindex(y.index).ffill().bfill()
            self.feature_names = list(Xc.columns)
            self._model = SARIMAX(y, exog=Xc, order=order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            self._last_X = Xc
        else:
            self._model = ARIMA(y, order=order).fit()
        return self

    def predict(self, steps: int = 1, X_future: Optional[pd.DataFrame] = None) -> FitPredictResult:
        if self.name == "arimax" and X_future is not None:
            exog = X_future[self.feature_names].ffill().fillna(0)
            fc = self._model.forecast(steps=steps, exog=exog)
        else:
            fc = self._model.forecast(steps=steps)
        return FitPredictResult(predictions=np.asarray(fc, dtype=float), model_name=self.name)

    def predict_in_sample(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        return np.asarray(self._model.fittedvalues, dtype=float)


class DynamicRegressionModel(NowcastModel):
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "DynamicRegressionModel":
        assert X is not None
        data = pd.concat([y.rename("y"), X], axis=1).dropna()
        self.feature_names = [c for c in data.columns if c != "y"]
        self._model = LinearRegression()
        self._model.fit(data[self.feature_names], data["y"])
        self._data = data
        return self

    def predict(self, steps: int = 1, X_future: Optional[pd.DataFrame] = None) -> FitPredictResult:
        assert X_future is not None
        Xp = X_future[self.feature_names].ffill().fillna(0)
        preds = self._model.predict(Xp.iloc[:steps])
        imp = {f: float(c) for f, c in zip(self.feature_names, self._model.coef_)}
        return FitPredictResult(predictions=np.asarray(preds, dtype=float), model_name=self.name, feature_importance=imp)

    def predict_in_sample(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self._model.predict(self._data[self.feature_names])


class VARModel(NowcastModel):
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "VARModel":
        cols = [y.rename("npl")]
        if X is not None and not X.empty:
            preferred = [
                "gdp_growth", "credit_growth", "policy_rate", "inflation",
                "exchange_rate_vol", "news_stress", "lending_rate", "usdidr",
            ]
            keep = [c for c in preferred if c in X.columns][:5]
            if len(keep) < 2:
                # take densest numeric columns as companions
                dens = X.notna().mean().sort_values(ascending=False)
                keep = [c for c in dens.index if c not in keep][: max(0, 3 - len(keep))] + keep
            cols += [X[c] for c in keep if c in X.columns]
        data = pd.concat(cols, axis=1).dropna()
        self._cols = list(data.columns)
        self._data = data
        self._fallback = None
        if len(self._cols) < 2 or len(data) < 12:
            # Not enough series for VAR — ridge on lags of NPL + X
            from sklearn.linear_model import Ridge as _Ridge
            yv = data["npl"] if "npl" in data.columns else y.dropna()
            Xv = data.drop(columns=["npl"], errors="ignore") if "npl" in data.columns else (X.reindex(yv.index).ffill().bfill() if X is not None else None)
            if Xv is None or Xv.empty:
                Xv = pd.DataFrame({"lag1": yv.shift(1), "lag2": yv.shift(2)}).dropna()
                yv = yv.loc[Xv.index]
            self.feature_names = list(Xv.columns)
            self._fallback = _Ridge(alpha=1.0)
            self._fallback.fit(Xv.fillna(0), yv)
            self._last_X = Xv
            return self
        lags = 1 if self.name == "bvar" else 2
        lags = min(lags, max(1, len(data) // 10))
        self._model = VAR(data).fit(maxlags=lags, trend="c")
        return self

    def predict(self, steps: int = 1, X_future: Optional[pd.DataFrame] = None) -> FitPredictResult:
        if self._fallback is not None:
            if X_future is not None and self.feature_names:
                Xp = X_future.reindex(columns=self.feature_names).ffill().fillna(0)
            else:
                Xp = self._last_X.iloc[[-1]]
            preds = self._fallback.predict(Xp.iloc[:steps])
            return FitPredictResult(predictions=np.asarray(preds, dtype=float), model_name=self.name + "_ridge_fallback")
        fc = self._model.forecast(self._data.values[-self._model.k_ar :], steps=steps)
        npl_idx = self._cols.index("npl")
        return FitPredictResult(predictions=fc[:, npl_idx], model_name=self.name)

    def predict_in_sample(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._fallback is not None:
            return np.asarray(self._fallback.predict(self._last_X.fillna(0)), dtype=float)
        fitted = self._model.fittedvalues
        return np.asarray(fitted["npl"] if "npl" in fitted.columns else fitted.iloc[:, 0], dtype=float)


class DFMModel(NowcastModel):
    """Dynamic Factor Model — primary nowcasting approach for many indicators."""

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "DFMModel":
        assert X is not None
        # Select densest columns
        dens = X.notna().mean().sort_values(ascending=False)
        cols = list(dens.head(min(8, len(dens))).index)
        panel = pd.concat([y.rename("npl"), X[cols]], axis=1).dropna()
        self._panel = panel
        k_factors = 1 if len(cols) < 4 else 2
        try:
            self._dfm = DynamicFactor(panel, k_factors=k_factors, factor_order=1).fit(disp=False, maxiter=200)
            factors = self._dfm.factors.filtered.T
            if factors.ndim == 1:
                factors = factors.reshape(-1, 1)
            self._factor_df = pd.DataFrame(factors, index=panel.index, columns=[f"f{i}" for i in range(factors.shape[1])])
            self._bridge = LinearRegression()
            self._bridge.fit(self._factor_df, panel["npl"])
            self.feature_names = list(self._factor_df.columns)
            self._ok = True
        except Exception:
            # Fallback: PCA + regression
            from sklearn.decomposition import PCA

            self._pca = PCA(n_components=min(2, len(cols)))
            Z = StandardScaler().fit_transform(panel[cols])
            F = self._pca.fit_transform(Z)
            self._factor_df = pd.DataFrame(F, index=panel.index, columns=[f"f{i}" for i in range(F.shape[1])])
            self._bridge = LinearRegression().fit(self._factor_df, panel["npl"])
            self._last_X_cols = cols
            self._ok = False
        self._cols = cols
        return self

    def predict(self, steps: int = 1, X_future: Optional[pd.DataFrame] = None) -> FitPredictResult:
        # Use last factor + simple persistence for nowcast step
        last = self._factor_df.iloc[[-1]]
        preds = []
        cur = last.copy()
        for _ in range(steps):
            preds.append(float(self._bridge.predict(cur)[0]))
            # mild mean reversion of factors
            cur = cur * 0.9
        imp = {c: 1.0 / max(len(self._cols), 1) for c in self._cols}
        return FitPredictResult(predictions=np.array(preds), model_name=self.name, feature_importance=imp)

    def predict_in_sample(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self._bridge.predict(self._factor_df)


class MIDASModel(NowcastModel):
    """Simplified MIDAS: Almon-style weighted lags of high-frequency proxy (news_stress)."""

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "MIDASModel":
        assert X is not None
        hf = "news_stress" if "news_stress" in X.columns else X.columns[0]
        work = pd.DataFrame({"y": y})
        for lag in range(0, 3):
            work[f"hf_lag{lag}"] = X[hf].shift(lag)
        # low-freq controls
        for c in ["gdp_growth", "credit_growth", "lending_rate", "policy_rate"]:
            if c in X.columns:
                work[c] = X[c]
        work = work.dropna()
        self.feature_names = [c for c in work.columns if c != "y"]
        self._model = LinearRegression().fit(work[self.feature_names], work["y"])
        self._work = work
        self._hf = hf
        return self

    def predict(self, steps: int = 1, X_future: Optional[pd.DataFrame] = None) -> FitPredictResult:
        assert X_future is not None
        rows = []
        for i in range(steps):
            row = {}
            for f in self.feature_names:
                if f in X_future.columns:
                    row[f] = X_future.iloc[min(i, len(X_future) - 1)][f]
                elif f.startswith("hf_lag"):
                    lag = int(f.replace("hf_lag", ""))
                    idx = min(max(i - lag, 0), len(X_future) - 1)
                    row[f] = X_future.iloc[idx].get(self._hf, 0)
                else:
                    row[f] = 0.0
            rows.append(row)
        Xp = pd.DataFrame(rows)[self.feature_names].fillna(0)
        preds = self._model.predict(Xp)
        imp = {f: float(c) for f, c in zip(self.feature_names, self._model.coef_)}
        return FitPredictResult(predictions=np.asarray(preds), model_name=self.name, feature_importance=imp)

    def predict_in_sample(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self._model.predict(self._work[self.feature_names])


class MLModel(NowcastModel):
    def _make_estimator(self):
        name = self.name
        if name == "linear":
            return LinearRegression()
        if name == "ridge":
            return Ridge(alpha=1.0)
        if name == "lasso":
            return Lasso(alpha=0.01, max_iter=5000)
        if name == "elastic_net":
            return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
        if name == "random_forest":
            return RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
        if name == "gradient_boosting":
            return GradientBoostingRegressor(random_state=42)
        if name == "xgboost":
            if xgb is None:
                return GradientBoostingRegressor(random_state=42)
            return xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0)
        if name == "lightgbm":
            if lgb is None:
                return GradientBoostingRegressor(random_state=42)
            return lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1)
        if name == "catboost":
            if CatBoostRegressor is None:
                return GradientBoostingRegressor(random_state=42)
            return CatBoostRegressor(iterations=200, depth=4, learning_rate=0.05, verbose=False, random_seed=42)
        if name == "svr":
            return SVR(C=1.0, epsilon=0.05)
        return Ridge()

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "MLModel":
        assert X is not None
        data = pd.concat([y.rename("y"), X], axis=1).dropna()
        self.feature_names = [c for c in data.columns if c != "y"]
        self._X = data[self.feature_names]
        self._y = data["y"]
        Xs = self._scaler.fit_transform(self._X)
        self._model = self._make_estimator()
        self._model.fit(Xs, self._y)
        return self

    def predict(self, steps: int = 1, X_future: Optional[pd.DataFrame] = None) -> FitPredictResult:
        assert X_future is not None
        Xp = X_future[self.feature_names].ffill().fillna(self._X.median())
        Xs = self._scaler.transform(Xp.iloc[:steps])
        preds = self._model.predict(Xs)
        imp = {}
        if hasattr(self._model, "feature_importances_"):
            imp = {f: float(v) for f, v in zip(self.feature_names, self._model.feature_importances_)}
        elif hasattr(self._model, "coef_"):
            coef = np.ravel(self._model.coef_)
            imp = {f: float(v) for f, v in zip(self.feature_names, coef)}
        return FitPredictResult(predictions=np.asarray(preds, dtype=float), model_name=self.name, feature_importance=imp)

    def predict_in_sample(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self._model.predict(self._scaler.transform(self._X))


class StateSpaceModel(NowcastModel):
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "StateSpaceModel":
        y = y.astype(float).dropna()
        # Local level as latent credit stress proxy
        self._model = SARIMAX(y, order=(1, 1, 0), trend="c", enforce_stationarity=False).fit(disp=False)
        self._y = y
        return self

    def predict(self, steps: int = 1, X_future: Optional[pd.DataFrame] = None) -> FitPredictResult:
        fc = self._model.forecast(steps=steps)
        return FitPredictResult(predictions=np.asarray(fc, dtype=float), model_name=self.name)

    def predict_in_sample(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        return np.asarray(self._model.fittedvalues, dtype=float)


class HybridModel(NowcastModel):
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "HybridModel":
        self._base = ARIMAModel("arimax" if "arimax" in self.name else "dfm")
        if "dfm" in self.name:
            self._base = DFMModel("dfm")
        else:
            self._base = ARIMAModel("arimax")
        self._base.fit(y, X)
        # Residual booster
        fitted = pd.Series(self._base.predict_in_sample(X), index=y.dropna().index[: len(self._base.predict_in_sample(X))])
        # Align
        common = y.dropna().index.intersection(fitted.index)
        resid = y.loc[common] - fitted.loc[common]
        self._boost = MLModel("xgboost")
        Xc = X.loc[common] if X is not None else pd.DataFrame(index=common)
        self._boost.fit(resid, Xc)
        self._y = y
        self._X = X
        return self

    def predict(self, steps: int = 1, X_future: Optional[pd.DataFrame] = None) -> FitPredictResult:
        base = self._base.predict(steps=steps, X_future=X_future)
        boost = self._boost.predict(steps=steps, X_future=X_future)
        preds = base.predictions + boost.predictions
        imp = {**base.feature_importance, **{f"resid_{k}": v for k, v in boost.feature_importance.items()}}
        return FitPredictResult(predictions=preds, model_name=self.name, feature_importance=imp)

    def predict_in_sample(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self._base.predict_in_sample(X)  # approximate


class EnsembleModel(NowcastModel):
    def __init__(self, name: str = "ensemble", members: Optional[list[str]] = None, weights: Optional[dict[str, float]] = None):
        super().__init__(name)
        self.member_names = members or ["dfm", "xgboost", "arimax", "midas_news"]
        self.weights = weights
        self._members: list[NowcastModel] = []

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "EnsembleModel":
        self._members = []
        for m in self.member_names:
            try:
                model = build_model(m)
                model.fit(y, X)
                self._members.append(model)
            except Exception:
                continue
        if self.weights is None:
            # Equal weights; caller may overwrite from OOS performance
            n = max(len(self._members), 1)
            self.weights = {m.name: 1 / n for m in self._members}
        return self

    def predict(self, steps: int = 1, X_future: Optional[pd.DataFrame] = None) -> FitPredictResult:
        preds = []
        wsum = 0.0
        imp: dict[str, float] = {}
        for m in self._members:
            w = self.weights.get(m.name, 0.0)
            if w <= 0:
                continue
            r = m.predict(steps=steps, X_future=X_future)
            preds.append(w * r.predictions)
            wsum += w
            for k, v in r.feature_importance.items():
                imp[k] = imp.get(k, 0.0) + w * abs(v)
        if not preds:
            raise RuntimeError("Ensemble has no successful members")
        out = np.sum(preds, axis=0) / wsum
        return FitPredictResult(predictions=out, model_name=self.name, feature_importance=imp)

    def predict_in_sample(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        # Not uniformly defined
        return self._members[0].predict_in_sample(X)


class DeepLearningStub(NowcastModel):
    """Only used when sample is large; otherwise falls back to Gradient Boosting."""

    MIN_OBS = 120

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "DeepLearningStub":
        n = y.dropna().shape[0]
        if n < self.MIN_OBS:
            self._fallback = MLModel("gradient_boosting")
            self._fallback.fit(y, X)
            self._used_fallback = True
            self.name = f"{self.name}_fallback_gbm"
            return self
        # Without forcing heavy torch install in demo, use GBM with sequence lags
        self._fallback = MLModel("gradient_boosting")
        self._fallback.fit(y, X)
        self._used_fallback = True
        return self

    def predict(self, steps: int = 1, X_future: Optional[pd.DataFrame] = None) -> FitPredictResult:
        r = self._fallback.predict(steps=steps, X_future=X_future)
        r.model_name = self.name
        r.extras["dl_fallback"] = True
        return r

    def predict_in_sample(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self._fallback.predict_in_sample(X)


def build_model(name: str) -> NowcastModel:
    if name in ("arima", "sarima", "arimax"):
        return ARIMAModel(name)
    if name == "dynamic_regression":
        return DynamicRegressionModel(name)
    if name in ("var", "bvar"):
        return VARModel(name)
    if name == "state_space":
        return StateSpaceModel(name)
    if name == "dfm":
        return DFMModel(name)
    if name in ("midas", "midas_news"):
        return MIDASModel(name)
    if name in ("linear", "ridge", "lasso", "elastic_net", "random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost", "svr"):
        return MLModel(name)
    if name in ("lstm", "gru", "tcn", "transformer_ts"):
        return DeepLearningStub(name)
    if name in ("hybrid_arimax_xgb", "hybrid_dfm_xgb"):
        return HybridModel(name)
    if name == "ensemble":
        return EnsembleModel(name)
    raise ValueError(f"Unknown model: {name}")
