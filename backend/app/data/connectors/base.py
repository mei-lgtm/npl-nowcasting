"""Modular data source connectors — standardized schema ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import pandas as pd

from app.data.schemas.models import EconomicObservation, Frequency, NewsArticle


class BaseConnector(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_observations(
        self,
        indicators: list[str],
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> list[EconomicObservation]:
        raise NotImplementedError

    def validate(self, rows: list[EconomicObservation]) -> list[EconomicObservation]:
        valid = []
        for r in rows:
            if r.value is None or r.reference_period is None or r.publication_date is None:
                continue
            valid.append(r)
        return valid


class DemoConnector(BaseConnector):
    name = "demo"

    def __init__(self, observations: pd.DataFrame):
        self.observations = observations

    def fetch_observations(self, indicators: list[str], start=None, end=None) -> list[EconomicObservation]:
        df = self.observations
        if indicators:
            df = df[df["indicator"].isin(indicators)]
        out = []
        for _, row in df.iterrows():
            out.append(
                EconomicObservation(
                    indicator=row["indicator"],
                    value=float(row["value"]),
                    unit=row.get("unit", "percent"),
                    frequency=Frequency(row.get("frequency", "monthly")),
                    reference_period=row["reference_period"],
                    publication_date=pd.to_datetime(row["publication_date"]).date(),
                    source=row.get("source", "demo"),
                    is_synthetic=True,
                )
            )
        return self.validate(out)


class OJKConnector(BaseConnector):
    """Placeholder for OJK banking statistics API / scrape adapter."""

    name = "ojk"

    def fetch_observations(self, indicators: list[str], start=None, end=None) -> list[EconomicObservation]:
        # Production: call OJK endpoints and map into EconomicObservation
        return []


class BIConnector(BaseConnector):
    name = "bank_indonesia"

    def fetch_observations(self, indicators: list[str], start=None, end=None) -> list[EconomicObservation]:
        return []


class BPSConnector(BaseConnector):
    name = "bps"

    def fetch_observations(self, indicators: list[str], start=None, end=None) -> list[EconomicObservation]:
        return []


class NewsRSSConnector(BaseConnector):
    name = "news_rss"

    def fetch_observations(self, indicators: list[str], start=None, end=None):
        return []

    def fetch_news(self, feeds: list[str]) -> list[NewsArticle]:
        # Production: parse RSS/Atom via feedparser
        return []


CONNECTOR_REGISTRY = {
    "demo": DemoConnector,
    "ojk": OJKConnector,
    "bank_indonesia": BIConnector,
    "bps": BPSConnector,
    "news_rss": NewsRSSConnector,
}
