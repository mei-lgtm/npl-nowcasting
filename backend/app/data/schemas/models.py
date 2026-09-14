from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class Frequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class DataMode(str, Enum):
    DEMO = "demo"
    LIVE = "live"


class ObservationType(str, Enum):
    ACTUAL = "actual"
    NOWCAST = "nowcast"
    FORECAST = "forecast"


class RiskSignal(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EconomicObservation(BaseModel):
    indicator: str
    value: float
    unit: str = "percent"
    frequency: Frequency
    reference_period: str
    publication_date: date
    source: str
    vintage_id: Optional[str] = None
    is_revised: bool = False
    is_synthetic: bool = False


class NewsArticle(BaseModel):
    id: str
    headline: str
    article_text: Optional[str] = None
    published_at: datetime
    collected_at: datetime
    source: str
    url: Optional[str] = None
    topic: Optional[str] = None
    sector: Optional[str] = None
    entities: list[str] = Field(default_factory=list)
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    credit_risk: Optional[str] = None
    credit_risk_score: Optional[float] = None
    economic_stress: Optional[str] = None
    banking_risk: Optional[str] = None
    financial_distress: Optional[str] = None
    npl_relevance: Optional[float] = None
    is_synthetic: bool = False


class SentimentResult(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    sentiment_score: float
    credit_risk: Literal["improving", "stable", "deteriorating"]
    financial_distress: Literal["low", "medium", "high"]
    economic_stress: Literal["low", "medium", "high"]
    banking_risk: Literal["low", "medium", "high"]
    sector: Optional[str] = None
    npl_relevance: float
    stress_contribution: float = 0.0
    model_used: str


class ModelMetrics(BaseModel):
    model: str
    rmse: float
    mae: float
    mape: float
    r2: float
    bias: Optional[float] = None
    directional_accuracy: Optional[float] = None
    rank: Optional[int] = None
    n_observations: int = 0
    is_demo: bool = False


class DriverContribution(BaseModel):
    feature: str
    contribution_pp: float
    direction: Literal["up", "down", "neutral"]
    note: str = "Model contribution — not a causal claim."


class NowcastResult(BaseModel):
    target: str
    reference_period: str
    nowcast_value: float
    actual_last_release: Optional[float] = None
    actual_last_period: Optional[str] = None
    change_pp: Optional[float] = None
    confidence_low: float
    confidence_high: float
    risk_signal: RiskSignal
    observation_type: ObservationType = ObservationType.NOWCAST
    model_used: str
    model_confidence: Literal["low", "moderate", "high"]
    drivers: list[DriverContribution] = Field(default_factory=list)
    news_signals: list[str] = Field(default_factory=list)
    is_synthetic: bool = False
    disclaimer: str = (
        "Statistical nowcast estimate. Not an official NPL release. "
        "Feature contributions reflect model association, not causation."
    )


class ScenarioInput(BaseModel):
    gdp_growth: float = 5.0
    credit_growth: float = 8.0
    policy_rate: float = 5.0
    inflation: float = 2.8
    exchange_rate: float = 15800.0
    news_stress: float = 0.20


class ScenarioResult(BaseModel):
    label: str
    inputs: ScenarioInput
    estimated_npl: float
    delta_from_baseline: float
    is_simulation: bool = True
    disclaimer: str = "Model simulation only — not an official forecast."


class WorkflowConfig(BaseModel):
    target: str = "gross_npl_ratio"
    frequency: Frequency = Frequency.MONTHLY
    model_selection: Literal["manual", "auto", "ensemble"] = "auto"
    selected_models: list[str] = Field(default_factory=list)
    sentiment_model: str = "tfidf_logistic"
    sentiment_auto: bool = False
    variable_selection: Literal["manual", "automatic", "hybrid"] = "automatic"
    selected_indicators: list[str] = Field(default_factory=list)
    mandatory_indicators: list[str] = Field(default_factory=list)
    backtest_mode: Literal["expanding", "rolling"] = "expanding"
    rolling_window: int = 36
    include_news: bool = True
    as_of: Optional[datetime] = None


class TimeSeriesPoint(BaseModel):
    period: str
    value: Optional[float] = None
    observation_type: ObservationType
    low: Optional[float] = None
    high: Optional[float] = None


class DataQualityItem(BaseModel):
    indicator: str
    status: Literal["ok", "warning", "error"]
    frequency: str
    last_update: str
    publication_lag_days: int
    missing_pct: float
    outlier_count: int
    source: str
    is_synthetic: bool = False


class AIAnalystResponse(BaseModel):
    summary: str
    bullets: list[str]
    risks: list[str]
    confidence_note: str
    is_synthetic: bool = False


class ApiEnvelope(BaseModel):
    data: Any
    mode: str
    is_synthetic: bool = False
    message: Optional[str] = None
