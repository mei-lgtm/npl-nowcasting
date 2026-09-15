from __future__ import annotations

from typing import Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.data.schemas.models import ScenarioInput, WorkflowConfig
from app.engines.sentiment.engine import SentimentEngine
from app.services.nowcasting_service import get_service
from app.api.data_input import router as data_input_router
from app.api.ask_ai import router as ask_ai_router

router = APIRouter()
router.include_router(data_input_router)
router.include_router(ask_ai_router)


class SentimentRequest(BaseModel):
    text: str
    model: str = "tfidf_logistic"
    source: str = "DemoWire"


class SentimentBatchItem(BaseModel):
    id: Optional[Union[str, int]] = None
    headline: str = ""
    text: Optional[str] = None
    source: str = "DemoWire"
    sector: Optional[str] = None
    topic: Optional[str] = None


class SentimentBatchRequest(BaseModel):
    articles: list[SentimentBatchItem] = Field(default_factory=list)
    model: str = "tfidf_logistic"


class ScenarioRequest(BaseModel):
    baseline: ScenarioInput = Field(default_factory=ScenarioInput)
    adverse: Optional[ScenarioInput] = Field(
        default_factory=lambda: ScenarioInput(
            gdp_growth=3.0,
            credit_growth=5.0,
            policy_rate=6.0,
            inflation=4.0,
            exchange_rate=16500,
            news_stress=0.75,
        )
    )


class WorkflowRequest(BaseModel):
    config: WorkflowConfig = Field(default_factory=WorkflowConfig)


def _envelope(data, message: Optional[str] = None):
    svc = get_service()
    return {
        "data": data,
        "mode": settings.app_mode,
        "is_synthetic": svc.is_synthetic,
        "message": message or ("DEMO / SYNTHETIC DATA" if svc.is_synthetic else None),
    }


@router.get("/catalog")
def catalog():
    return _envelope(get_service().catalog())


@router.get("/meta")
def meta():
    svc = get_service()
    return _envelope(svc.meta)


@router.post("/workflow/run")
def run_workflow(req: WorkflowRequest):
    """Full pipeline: features → selection → backtest → nowcast."""
    svc = get_service()
    cfg = req.config
    selection = svc.run_backtest(cfg)
    nowcast = svc.generate_nowcast(cfg)
    series = svc.timeseries(cfg)
    news = svc.news_dashboard(cfg)
    analyst = svc.ai_analyst(cfg)
    return _envelope(
        {
            "selection": {
                "leaderboard": selection.get("leaderboard"),
                "best_model": selection.get("best_model"),
                "ensemble_weights": selection.get("ensemble_weights"),
                "features": selection.get("features"),
                "failed": selection.get("failed"),
            },
            "nowcast": nowcast.model_dump(),
            "timeseries": series,
            "news": news,
            "analyst": analyst.model_dump(),
            "diagnostics": selection.get("diagnostics"),
        }
    )


@router.post("/nowcast")
def nowcast(req: WorkflowRequest):
    svc = get_service()
    if "last_selection" not in svc._cache:
        svc.run_backtest(req.config)
    result = svc.generate_nowcast(req.config)
    return _envelope(result.model_dump())


@router.post("/backtest")
def backtest(req: WorkflowRequest):
    result = get_service().run_backtest(req.config)
    board = []
    for r in result.get("leaderboard") or []:
        board.append({k: v for k, v in r.items() if k != "predictions"})
    return _envelope(
        {
            "leaderboard": board,
            "best_model": result.get("best_model"),
            "ensemble_weights": result.get("ensemble_weights"),
            "features": result.get("features"),
            "failed": result.get("failed"),
            "diagnostics": result.get("diagnostics"),
        }
    )


@router.get("/timeseries")
def timeseries():
    return _envelope(get_service().timeseries())


@router.get("/news")
def news():
    return _envelope(get_service().news_dashboard())


@router.post("/sentiment/analyze")
def analyze_sentiment(req: SentimentRequest):
    engine = SentimentEngine()
    result = engine.analyze(req.text, model=req.model, source=req.source)
    return _envelope(result.model_dump())


@router.post("/sentiment/batch")
def analyze_sentiment_batch(req: SentimentBatchRequest):
    """Run real sentiment models on many headlines (TF-IDF / transformers if available)."""
    engine = SentimentEngine()
    articles = [a.model_dump() for a in req.articles]
    # Prefer headline; fall back to text
    for a in articles:
        if not a.get("headline") and a.get("text"):
            a["headline"] = a["text"]
    scored = engine.analyze_batch(articles, model=req.model)
    nsi = engine.news_stress_index(articles, model=req.model)
    return _envelope(
        {
            "model_requested": req.model,
            "model_used": nsi.get("model_used"),
            "count": len(scored),
            "news_stress_index": nsi.get("news_stress_index"),
            "negative_ratio": nsi.get("negative_ratio"),
            "articles": scored,
        }
    )


@router.get("/sentiment/models")
def sentiment_models():
    """List available sentiment backends and whether transformers are importable."""
    try:
        import transformers  # noqa: F401

        hf = True
    except Exception:
        hf = False
    return _envelope(
        {
            "available": SentimentEngine.AVAILABLE,
            "transformers_installed": hf,
            "notes": {
                "tfidf_logistic": "sklearn TF-IDF + Logistic Regression (trained on Indo+EN credit lexicon)",
                "tfidf_svm": "sklearn TF-IDF + LinearSVC",
                "indobert": "HuggingFace IndoBERT if installed; else TF-IDF Indo fallback",
                "mbert": "HuggingFace mBERT sentiment if installed; else TF-IDF Indo fallback",
                "xlm_roberta": "HuggingFace XLM-R if installed; else TF-IDF Indo fallback",
                "ensemble": "Majority vote rules + TF-IDF LR/SVM",
            },
        }
    )


@router.post("/scenario")
def scenario(req: ScenarioRequest):
    return _envelope(get_service().scenario(req.baseline, req.adverse))


@router.get("/data-quality")
def data_quality():
    return _envelope(get_service().data_quality())


@router.get("/indicators")
def indicators():
    return _envelope(get_service().indicators_latest())


@router.get("/track-record")
def track_record():
    return _envelope(get_service().track_record())


@router.get("/analyst")
def analyst():
    return _envelope(get_service().ai_analyst().model_dump())


@router.get("/alerts")
def alerts():
    svc = get_service()
    if "last_nowcast" not in svc._cache:
        svc.run_backtest(WorkflowConfig())
        svc.generate_nowcast(WorkflowConfig())
    nc = svc._cache["last_nowcast"]
    return _envelope(
        {
            "title": "NPL RISK ALERT",
            "current_nowcast": nc.nowcast_value,
            "previous": nc.actual_last_release,
            "change_pp": nc.change_pp,
            "signal": nc.risk_signal.value,
            "drivers": [d.model_dump() for d in nc.drivers[:5]],
            "news_signals": nc.news_signals,
            "is_synthetic": nc.is_synthetic,
        }
    )


@router.post("/reload-demo")
def reload_demo():
    if settings.app_mode != "demo":
        raise HTTPException(400, "Reload only available in demo mode")
    from pathlib import Path

    from app.data.synthetic import write_demo_files

    d = Path(settings.data_dir)
    if not d.is_absolute():
        d = Path(__file__).resolve().parents[2] / d
    write_demo_files(d)
    get_service().reload()
    return _envelope({"ok": True}, message="Demo data regenerated")
