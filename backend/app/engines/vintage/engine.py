"""Real-Time Data Vintage Engine — prevents look-ahead bias."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd


class VintageEngine:
    """Reconstruct the information set available as of a given timestamp.

    Rules:
    - Economic observations: publication_date <= as_of_date
    - News: published_at <= as_of_timestamp
    """

    def __init__(self, observations: pd.DataFrame, news: Optional[pd.DataFrame] = None):
        self.observations = observations.copy()
        self.news = news.copy() if news is not None else None
        if not self.observations.empty:
            self.observations["publication_date"] = pd.to_datetime(
                self.observations["publication_date"]
            ).dt.date
        if self.news is not None and not self.news.empty:
            self.news["published_at"] = pd.to_datetime(self.news["published_at"])

    def available_observations(self, as_of: datetime | date) -> pd.DataFrame:
        as_of_date = as_of.date() if isinstance(as_of, datetime) else as_of
        mask = self.observations["publication_date"] <= as_of_date
        return self.observations.loc[mask].copy()

    def available_news(self, as_of: datetime) -> pd.DataFrame:
        if self.news is None or self.news.empty:
            return pd.DataFrame()
        mask = self.news["published_at"] <= pd.Timestamp(as_of)
        return self.news.loc[mask].copy()

    def panel_as_of(
        self,
        as_of: datetime | date,
        indicators: Optional[list[str]] = None,
        frequency: str = "monthly",
    ) -> pd.DataFrame:
        """Wide panel of latest available values per reference_period as of timestamp."""
        obs = self.available_observations(as_of)
        if indicators:
            obs = obs[obs["indicator"].isin(indicators)]
        if obs.empty:
            return pd.DataFrame()

        # Keep latest vintage per (indicator, reference_period)
        obs = obs.sort_values("publication_date")
        obs = obs.drop_duplicates(subset=["indicator", "reference_period"], keep="last")
        wide = obs.pivot(index="reference_period", columns="indicator", values="value")
        wide = wide.sort_index()
        wide.index.name = "reference_period"
        return wide.reset_index()

    def latest_actual_npl(self, as_of: datetime | date, target: str = "gross_npl_ratio") -> dict | None:
        obs = self.available_observations(as_of)
        npl = obs[obs["indicator"] == target]
        if npl.empty:
            return None
        npl = npl.sort_values(["reference_period", "publication_date"])
        row = npl.iloc[-1]
        return {
            "period": row["reference_period"],
            "value": float(row["value"]),
            "publication_date": str(row["publication_date"]),
            "observation_type": "actual",
        }

    def unreleased_periods(
        self,
        as_of: datetime | date,
        target: str = "gross_npl_ratio",
        through_period: Optional[str] = None,
    ) -> list[str]:
        """Reference periods up to through_period with no published actual NPL yet."""
        as_of_date = as_of.date() if isinstance(as_of, datetime) else as_of
        if through_period is None:
            through_period = as_of_date.strftime("%Y-%m")

        latest = self.latest_actual_npl(as_of, target)
        if latest is None:
            return [through_period]

        start = pd.Period(latest["period"], freq="M") + 1
        end = pd.Period(through_period, freq="M")
        periods = []
        cur = start
        while cur <= end:
            periods.append(str(cur))
            cur += 1
        return periods
