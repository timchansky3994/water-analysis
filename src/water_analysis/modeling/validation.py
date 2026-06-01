"""Time-aware validation and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class HoldoutSplit:
    """Chronological holdout split."""

    train_frame: pd.DataFrame
    test_frame: pd.DataFrame


@dataclass(frozen=True)
class BacktestSplit:
    """One rolling backtest split."""

    split_index: int
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame


def sort_chronologically(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort a modeling frame chronologically with stable tie-breaking."""
    return frame.sort_values(["SampleDate", "FullPointCode"]).reset_index(drop=True)


def chronological_holdout_split(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    min_train_size: int = 20,
) -> HoldoutSplit:
    """Split a modeling frame into chronological train and test subsets."""
    sorted_frame = sort_chronologically(frame)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    total_rows = len(sorted_frame)
    test_rows = max(1, int(round(total_rows * test_size)))
    train_rows = total_rows - test_rows
    if train_rows < min_train_size:
        raise ValueError(
            f"Chronological holdout would leave only {train_rows} training rows, below min_train_size={min_train_size}."
        )

    return HoldoutSplit(
        train_frame=sorted_frame.iloc[:train_rows].reset_index(drop=True),
        test_frame=sorted_frame.iloc[train_rows:].reset_index(drop=True),
    )


def rolling_backtest_splits(
    frame: pd.DataFrame,
    *,
    initial_train_size: int,
    test_window_size: int,
    step_size: int | None = None,
) -> list[BacktestSplit]:
    """Generate expanding-window rolling backtest splits."""
    sorted_frame = sort_chronologically(frame)
    if initial_train_size <= 0 or test_window_size <= 0:
        raise ValueError("initial_train_size and test_window_size must be positive.")

    step = step_size or test_window_size
    splits: list[BacktestSplit] = []
    split_index = 0
    train_end = initial_train_size

    while train_end + test_window_size <= len(sorted_frame):
        train_frame = sorted_frame.iloc[:train_end].reset_index(drop=True)
        test_frame = sorted_frame.iloc[train_end : train_end + test_window_size].reset_index(drop=True)
        splits.append(BacktestSplit(split_index=split_index, train_frame=train_frame, test_frame=test_frame))
        split_index += 1
        train_end += step

    return splits


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate symmetric MAPE in percent."""
    denominator = np.abs(y_true) + np.abs(y_pred)
    valid = denominator != 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(200.0 * np.abs(y_pred[valid] - y_true[valid]) / denominator[valid]))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics."""
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(sqrt(mean_squared_error(y_true, y_pred))),
        "smape": smape(y_true, y_pred),
    }
    metrics["r2"] = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    return metrics


def compute_combined_score(
    holdout_metrics: dict[str, float],
    backtest_metrics: dict[str, float] | None,
    *,
    primary_metric: str = "rmse",
    weight_holdout: float = 0.4,
    weight_backtest: float = 0.6,
) -> tuple[float | None, bool]:
    """Compute weighted score of holdout and backtest primary metric.

    Returns (combined_score, used_fallback).
    If backtest metrics are missing or have no value for primary_metric,
    falls back to holdout value and returns used_fallback=True.
    Returns (None, False) if holdout doesn't have the primary metric either.
    """
    holdout_value = holdout_metrics.get(primary_metric)
    if holdout_value is None:
        return None, False
    backtest_value = (backtest_metrics or {}).get(primary_metric)
    if backtest_value is None:
        return holdout_value, True
    return weight_holdout * holdout_value + weight_backtest * backtest_value, False


def compute_stability_ratio(
    holdout_metrics: dict[str, float],
    backtest_metrics: dict[str, float] | None,
    *,
    primary_metric: str = "rmse",
) -> float | None:
    """Compute backtest/holdout ratio of the primary metric.

    Returns None if either metric is missing or holdout is zero.
    """
    holdout_value = holdout_metrics.get(primary_metric)
    backtest_value = (backtest_metrics or {}).get(primary_metric)
    if holdout_value is None or backtest_value is None:
        return None
    if holdout_value == 0:
        return None
    return backtest_value / holdout_value
