"""Baseline predictors for honest model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MedianBaseline:
    """Predict the median target value from the training fold."""

    median_: float | None = None

    def fit(self, train_frame: pd.DataFrame, target: str) -> "MedianBaseline":
        """Fit the baseline on the training target distribution."""
        series = train_frame[target].dropna()
        self.median_ = float(series.median()) if not series.empty else None
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict a constant median for each row."""
        if self.median_ is None:
            raise ValueError("MedianBaseline is not fitted.")
        return np.full(len(frame), self.median_, dtype=float)


@dataclass
class LastValueBaseline:
    """Predict the last observed training target value."""

    last_value_: float | None = None

    def fit(self, train_frame: pd.DataFrame, target: str) -> "LastValueBaseline":
        """Fit the baseline on the last chronological training value."""
        series = train_frame[target].dropna()
        self.last_value_ = float(series.iloc[-1]) if not series.empty else None
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict the last seen value for each row."""
        if self.last_value_ is None:
            raise ValueError("LastValueBaseline is not fitted.")
        return np.full(len(frame), self.last_value_, dtype=float)


@dataclass
class SeasonalMedianBaseline:
    """Predict season-level medians with fallback to a global median."""

    seasonal_medians_: dict[str, float] | None = None
    global_median_: float | None = None

    def fit(self, train_frame: pd.DataFrame, target: str) -> "SeasonalMedianBaseline":
        """Fit seasonal medians from the training data."""
        valid = train_frame[["season", target]].dropna()
        self.seasonal_medians_ = {
            str(season): float(values.median())
            for season, values in valid.groupby("season", dropna=True)[target]
        }
        series = train_frame[target].dropna()
        self.global_median_ = float(series.median()) if not series.empty else None
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict seasonal medians with fallback to the global median."""
        if self.global_median_ is None:
            raise ValueError("SeasonalMedianBaseline is not fitted.")
        seasonal_medians = self.seasonal_medians_ or {}
        predictions: list[float] = []
        for season in frame["season"].astype(str):
            predictions.append(seasonal_medians.get(season, self.global_median_))
        return np.asarray(predictions, dtype=float)


def baseline_registry() -> dict[str, type[MedianBaseline | LastValueBaseline | SeasonalMedianBaseline]]:
    """Return supported baseline classes keyed by name."""
    return {
        "median_baseline": MedianBaseline,
        "last_value_baseline": LastValueBaseline,
        "seasonal_median_baseline": SeasonalMedianBaseline,
    }
