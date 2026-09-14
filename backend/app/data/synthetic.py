"""Synthetic DEMO data generator for NPL nowcasting.

All series are SYNTHETIC and must never be presented as official Indonesian data.
Relationships are realistic for demonstration but fabricated.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import settings

SECTORS = [
    "Manufacturing",
    "Mining",
    "Construction",
    "Property",
    "Trade",
    "Agriculture",
    "Transportation",
    "Consumer",
    "Financial Services",
    "Technology",
    "Energy",
]

SOURCES = ["DemoWire", "EconDaily", "BankingInsight", "MacroPulse", "CreditWatch"]

NEWS_TEMPLATES = {
    "negative": [
        "{sector} firms face rising repayment pressure amid tighter credit conditions",
        "Restructuring activity increases among mid-sized {sector} borrowers",
        "Corporate distress signals emerge in {sector} loan portfolios",
        "Layoffs announced at major {sector} companies raise credit concerns",
        "Default risk rises as {sector} demand softens",
        "Banks report higher special-mention loans linked to {sector}",
        "Property collateral values weaken, elevating NPL risk",
        "Manufacturing PMI softens, pointing to credit quality pressure",
    ],
    "neutral": [
        "{sector} credit conditions remain broadly stable",
        "Banking sector monitors {sector} exposures amid mixed data",
        "Regulators maintain oversight of {sector} lending standards",
        "Loan growth in {sector} holds near recent averages",
        "Credit quality in {sector} shows little month-to-month change",
    ],
    "positive": [
        "{sector} borrowers show improved repayment capacity",
        "Credit quality improves as {sector} activity strengthens",
        "Banks report lower restructuring needs in {sector}",
        "Corporate earnings recovery supports {sector} loan performance",
        "Stable demand reduces stress in {sector} credit books",
    ],
}


def _month_ends(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="ME")


def generate_macro_panel(seed: int = 42) -> pd.DataFrame:
    """Generate monthly synthetic macro/banking panel with NPL dynamics."""
    rng = np.random.default_rng(seed)
    dates = _month_ends("2018-01-01", "2026-08-31")
    n = len(dates)

    # Latent credit-stress factor
    shock = rng.normal(0, 0.15, n)
    stress = np.zeros(n)
    stress[0] = 0.2
    for t in range(1, n):
        stress[t] = 0.85 * stress[t - 1] + shock[t]

    gdp = 5.0 - 0.8 * stress + rng.normal(0, 0.25, n)
    credit = 9.0 - 1.2 * stress + rng.normal(0, 0.4, n)
    inflation = 2.8 + 0.3 * stress + rng.normal(0, 0.15, n)
    policy = 4.5 + 0.6 * np.maximum(stress, 0) + rng.normal(0, 0.1, n)
    lending = policy + 2.0 + 0.4 * stress + rng.normal(0, 0.08, n)
    fx = 14500 + 800 * stress + np.cumsum(rng.normal(0, 40, n))
    fx_vol = 0.05 + 0.04 * np.abs(stress) + rng.normal(0, 0.005, n)
    ihsg = 6500 - 400 * stress + np.cumsum(rng.normal(0, 30, n))
    unemp = 5.2 + 0.5 * stress + rng.normal(0, 0.1, n)
    industrial = 3.5 - 1.0 * stress + rng.normal(0, 0.3, n)
    consumer_conf = 110 - 15 * stress + rng.normal(0, 2, n)
    oil = 70 + 10 * rng.normal(0, 1, n).cumsum() * 0.02
    money = 8.0 - 0.3 * stress + rng.normal(0, 0.2, n)
    car = 22.5 - 0.4 * stress + rng.normal(0, 0.15, n)
    ldr = 88 + 2 * stress + rng.normal(0, 0.5, n)
    roa = 2.4 - 0.35 * stress + rng.normal(0, 0.05, n)
    provision = 120 + 15 * stress + rng.normal(0, 2, n)
    special_mention = 3.0 + 1.5 * stress + rng.normal(0, 0.15, n)
    loan_at_risk = 7.0 + 2.5 * stress + rng.normal(0, 0.25, n)

    # NPL ratio — driven by lagged stress + credit + rates + noise
    npl = np.zeros(n)
    npl[0] = 2.6
    for t in range(1, n):
        npl[t] = (
            0.55 * npl[t - 1]
            + 0.35 * stress[t - 1]
            + 0.08 * (lending[t] - 6.5)
            - 0.06 * (gdp[t] - 5.0)
            - 0.04 * (credit[t] - 8.0)
            + 2.0
            + rng.normal(0, 0.04)
        )
    npl = np.clip(npl, 1.5, 5.5)

    # Publication lags (days after month end)
    pub_lag = {
        "gross_npl_ratio": 30,
        "gdp_growth": 45,  # quarterly proxy stored monthly
        "credit_growth": 20,
        "inflation": 10,
        "policy_rate": 0,
        "lending_rate": 5,
        "exchange_rate": 0,
        "exchange_rate_vol": 1,
        "ihsg": 0,
        "unemployment": 15,
        "industrial_production": 25,
        "consumer_confidence": 12,
        "oil_price": 0,
        "money_supply": 18,
        "car": 30,
        "ldr": 30,
        "roa": 30,
        "provision_coverage": 30,
        "special_mention": 30,
        "loan_at_risk": 30,
        "news_stress": 0,
    }

    series = {
        "gross_npl_ratio": npl,
        "gdp_growth": gdp,
        "credit_growth": credit,
        "inflation": inflation,
        "policy_rate": policy,
        "lending_rate": lending,
        "exchange_rate": fx,
        "exchange_rate_vol": fx_vol,
        "ihsg": ihsg,
        "unemployment": unemp,
        "industrial_production": industrial,
        "consumer_confidence": consumer_conf,
        "oil_price": oil,
        "money_supply": money,
        "car": car,
        "ldr": ldr,
        "roa": roa,
        "provision_coverage": provision,
        "special_mention": special_mention,
        "loan_at_risk": loan_at_risk,
        "latent_stress": stress,
    }

    rows = []
    for i, dt in enumerate(dates):
        ref = dt.strftime("%Y-%m")
        for ind, values in series.items():
            if ind == "latent_stress":
                continue
            lag = pub_lag.get(ind, 15)
            pub = (dt + timedelta(days=lag)).date()
            # Official NPL only released through June 2026 in demo narrative
            if ind == "gross_npl_ratio" and dt > pd.Timestamp("2026-06-30"):
                continue
            source = {
                "gross_npl_ratio": "OJK (synthetic)",
                "gdp_growth": "BPS (synthetic)",
                "credit_growth": "BI (synthetic)",
                "policy_rate": "BI (synthetic)",
            }.get(ind, "Demo Macro DB")
            rows.append(
                {
                    "indicator": ind,
                    "value": float(round(values[i], 4)),
                    "unit": "index" if ind in ("ihsg", "consumer_confidence", "oil_price", "exchange_rate") else "percent",
                    "frequency": "monthly",
                    "reference_period": ref,
                    "publication_date": pub.isoformat(),
                    "source": source,
                    "is_synthetic": True,
                }
            )

    return pd.DataFrame(rows)


def generate_news(seed: int = 42, n_articles: int = 1800) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 7)
    start = datetime(2018, 1, 1)
    end = datetime(2026, 9, 14)
    span_days = (end - start).days

    # Align news stress loosely with latent macro stress by month
    macro = generate_macro_panel(seed)
    stress_map = (
        macro[macro["indicator"] == "credit_growth"]  # placeholder; rebuild from panel
    )
    # Rebuild monthly stress proxy from NPL path
    npl = macro[macro["indicator"] == "gross_npl_ratio"].copy()
    npl["period"] = pd.to_datetime(npl["reference_period"] + "-01")
    stress_by_month = {
        p.to_period("M"): float(v)
        for p, v in zip(npl["period"], (npl["value"] - npl["value"].mean()) / (npl["value"].std() + 1e-6))
    }

    rows = []
    for i in range(n_articles):
        day_offset = int(rng.integers(0, span_days + 1))
        published = start + timedelta(days=day_offset, hours=int(rng.integers(6, 22)))
        period = pd.Timestamp(published).to_period("M")
        stress = stress_by_month.get(period, 0.0)
        # Higher stress → more negative news
        p_neg = float(np.clip(0.33 + 0.25 * stress, 0.15, 0.75))
        p_pos = float(np.clip(0.33 - 0.20 * stress, 0.10, 0.50))
        p_neu = max(0.05, 1 - p_neg - p_pos)
        label = rng.choice(["negative", "neutral", "positive"], p=[p_neg, p_neu, p_pos])
        sector = rng.choice(SECTORS)
        headline = rng.choice(NEWS_TEMPLATES[label]).format(sector=sector)
        relevance = float(np.clip(rng.normal(0.75 if label == "negative" else 0.55, 0.12), 0.2, 0.98))
        sent_score = {
            "negative": float(np.clip(rng.normal(0.75, 0.1), 0.5, 0.98)),
            "neutral": float(np.clip(rng.normal(0.5, 0.08), 0.35, 0.65)),
            "positive": float(np.clip(rng.normal(0.25, 0.1), 0.02, 0.45)),
        }[label]
        # Map to distress dimensions
        if label == "negative":
            credit_risk, distress, econ, bank = "deteriorating", "high", "high", "medium"
            if relevance < 0.6:
                distress, econ = "medium", "medium"
        elif label == "positive":
            credit_risk, distress, econ, bank = "improving", "low", "low", "low"
        else:
            credit_risk, distress, econ, bank = "stable", "medium", "low", "low"

        stress_contrib = (sent_score * 2 - 1) * relevance  # -1..+1-ish, positive = stress
        rows.append(
            {
                "id": f"demo-news-{i:05d}",
                "headline": headline,
                "article_text": headline + ". Synthetic article body for demonstration purposes only.",
                "published_at": published.isoformat(),
                "collected_at": (published + timedelta(minutes=int(rng.integers(5, 120)))).isoformat(),
                "source": rng.choice(SOURCES),
                "url": f"https://example.demo/news/{i:05d}",
                "topic": "credit_quality" if label != "positive" else "banking",
                "sector": sector,
                "entities": json.dumps([f"{sector} Corp"]),
                "sentiment": label,
                "sentiment_score": round(sent_score, 4),
                "credit_risk": credit_risk,
                "credit_risk_score": round(float(np.clip(sent_score if label != "positive" else 1 - sent_score, 0, 1)), 4),
                "economic_stress": econ,
                "banking_risk": bank,
                "financial_distress": distress,
                "npl_relevance": round(relevance, 4),
                "stress_contribution": round(float(stress_contrib), 4),
                "is_synthetic": True,
            }
        )
    return pd.DataFrame(rows)


def generate_news_stress_series(news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily/weekly/monthly News Stress Index from articles."""
    df = news.copy()
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["date"] = df["published_at"].dt.floor("D")
    # Stress index: weighted negative-leaning score in [-1, +1]
    df["raw"] = (
        (df["sentiment_score"] * 2 - 1)
        * df["npl_relevance"]
        * df["published_at"].map(lambda _: 1.0)  # recency applied later in vintage queries
    )
    # Flip so +1 = high stress (negative news)
    # sentiment_score high for negative already → raw positive for negative
    daily = (
        df.groupby("date")
        .agg(
            news_stress=("raw", "mean"),
            article_count=("id", "count"),
            negative_ratio=("sentiment", lambda s: (s == "negative").mean()),
            avg_sentiment=("sentiment_score", "mean"),
        )
        .reset_index()
    )
    daily["news_stress"] = daily["news_stress"].clip(-1, 1)
    daily["is_synthetic"] = True
    return daily


def write_demo_files(data_dir: Path | None = None) -> dict[str, Path]:
    data_dir = Path(data_dir or settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    macro = generate_macro_panel(settings.random_seed)
    news = generate_news(settings.random_seed)
    daily_stress = generate_news_stress_series(news)

    paths = {}
    paths["observations"] = data_dir / "observations.parquet"
    paths["news"] = data_dir / "news.parquet"
    paths["news_daily"] = data_dir / "news_daily.parquet"
    paths["meta"] = data_dir / "meta.json"

    macro.to_parquet(paths["observations"], index=False)
    news.to_parquet(paths["news"], index=False)
    daily_stress.to_parquet(paths["news_daily"], index=False)

    # Also CSV for easy inspection
    macro.to_csv(data_dir / "observations.csv", index=False)
    news.to_csv(data_dir / "news.csv", index=False)

    meta = {
        "mode": "DEMO / SYNTHETIC DATA",
        "disclaimer": (
            "All data in this directory is synthetic and fabricated for demonstration. "
            "It must NEVER be presented as actual Indonesian official statistics."
        ),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_observations": len(macro),
        "n_news": len(news),
        "npl_actual_through": "2026-06",
        "current_demo_date": "2026-09-14",
    }
    paths["meta"].write_text(json.dumps(meta, indent=2))
    return paths


if __name__ == "__main__":
    write_demo_files()
    print("Demo data written.")
