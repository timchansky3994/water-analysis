"""Seasonal feature encoding for optional use in regression models."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

_SEASON_NAMES = ["winter", "spring", "summer", "autumn"]
_SEASON_COLUMNS = ["season_winter", "season_spring", "season_summer", "season_autumn"]


def build_seasonal_features(
    frame: pd.DataFrame,
    mode: Literal["none", "season", "month"],
) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy of frame with seasonal feature columns added and their names.

    mode="none"   → no changes, returns (frame_copy, [])
    mode="season" → one-hot of season column: season_winter/spring/summer/autumn.
                    All four columns are always generated so names are deterministic
                    regardless of which seasons appear in the split.
    mode="month"  → cyclic encoding: month_sin = sin(2π·month/12),
                    month_cos = cos(2π·month/12).

    The returned frame is always a copy; the original is not modified.
    """
    if mode == "none":
        return frame.copy(), []

    out = frame.copy()

    if mode == "season":
        if "season" not in out.columns:
            for col in _SEASON_COLUMNS:
                out[col] = 0.0
            return out, list(_SEASON_COLUMNS)
        for season_label, col_name in zip(_SEASON_NAMES, _SEASON_COLUMNS):
            out[col_name] = (out["season"] == season_label).astype(float)
        return out, list(_SEASON_COLUMNS)

    # mode == "month"
    if "month" not in out.columns:
        out["month_sin"] = np.nan
        out["month_cos"] = np.nan
        return out, ["month_sin", "month_cos"]

    month_rad = 2.0 * math.pi * pd.to_numeric(out["month"], errors="coerce") / 12.0
    out["month_sin"] = np.sin(month_rad)
    out["month_cos"] = np.cos(month_rad)
    return out, ["month_sin", "month_cos"]


def seasonal_feature_names(mode: Literal["none", "season", "month"]) -> list[str]:
    """Return the expected column names for a given seasonal feature mode."""
    if mode == "none":
        return []
    if mode == "season":
        return list(_SEASON_COLUMNS)
    return ["month_sin", "month_cos"]
