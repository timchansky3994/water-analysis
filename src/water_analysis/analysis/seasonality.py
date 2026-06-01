"""Seasonal analysis diagnostic layer (always-on analytical layer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from scipy.stats import kruskal, spearmanr

from water_analysis.analysis.feature_selection import indicator_columns
from water_analysis.analysis.scopes import ScopeSlice
from water_analysis.preprocessing.pivot_builder import build_indicator_pivot


@dataclass(frozen=True)
class SeasonalityAnalysis:
    """Seasonal diagnostic results for one target in one scope."""

    target: str
    granularity: str
    group_stats: pd.DataFrame
    seasonal_pattern_detected: bool
    pattern_test: dict
    per_season_correlations: pd.DataFrame
    diagnostics: list


def _group_stats_for_target(
    pivot: pd.DataFrame,
    *,
    target: str,
    group_col: str,
    min_group_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    diag: list[str] = []

    groups = sorted(pivot[group_col].dropna().unique().tolist())
    for group_label in groups:
        values = pivot.loc[pivot[group_col] == group_label, target].dropna()
        n = len(values)
        if n == 0:
            continue
        if n < min_group_size:
            diag.append(
                f"В группе '{group_label}' только {n} наблюдений — статистика по ней ненадёжна."
            )
        rows.append(
            {
                "group": group_label,
                "n_observations": n,
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if n > 1 else float("nan"),
            }
        )

    _cols = ["group", "n_observations", "median", "q25", "q75", "mean", "std"]
    if not rows:
        return pd.DataFrame(columns=_cols), diag
    return pd.DataFrame(rows), diag


def _run_kruskal_wallis(
    pivot: pd.DataFrame,
    *,
    target: str,
    group_col: str,
    min_group_size: int,
) -> tuple[bool, dict]:
    groups = sorted(pivot[group_col].dropna().unique().tolist())
    group_arrays = []
    for group_label in groups:
        vals = pivot.loc[pivot[group_col] == group_label, target].dropna().to_numpy(dtype=float)
        if len(vals) >= min_group_size:
            group_arrays.append(vals)

    if len(group_arrays) < 2:
        reason = f"менее 2 групп с n>={min_group_size} наблюдений"
        return False, {"test": "skipped", "reason": reason}

    try:
        stat, p_value = kruskal(*group_arrays)
        return bool(p_value < 0.05), {
            "test": "kruskal_wallis",
            "statistic": float(stat),
            "p_value": float(p_value),
            "groups_used": len(group_arrays),
        }
    except Exception as exc:
        return False, {"test": "skipped", "reason": str(exc)}


def _per_group_correlations(
    pivot: pd.DataFrame,
    *,
    target: str,
    group_col: str,
    min_group_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    features = [c for c in indicator_columns(pivot) if c != target]
    groups = sorted(pivot[group_col].dropna().unique().tolist())
    rows: list[dict] = []
    diag: list[str] = []

    for group_label in groups:
        group_df = pivot.loc[pivot[group_col] == group_label]
        for feature in features:
            shared = group_df[[target, feature]].dropna()
            n_shared = len(shared)
            if n_shared < min_group_size:
                if n_shared > 0:
                    diag.append(
                        f"Корреляция '{feature}' в группе '{group_label}' пропущена: "
                        f"только {n_shared} совместных наблюдений (порог {min_group_size})."
                    )
                continue
            if shared[target].nunique() <= 1 or shared[feature].nunique() <= 1:
                continue
            try:
                corr_val, _ = spearmanr(shared[target], shared[feature])
                if pd.isna(corr_val):
                    continue
                rows.append(
                    {
                        "group": group_label,
                        "feature": feature,
                        "corr": float(corr_val),
                        "n_shared": n_shared,
                        "method": "spearman",
                    }
                )
            except Exception:
                continue

    _cols = ["group", "feature", "corr", "n_shared", "method"]
    if not rows:
        return pd.DataFrame(columns=_cols), diag
    return pd.DataFrame(rows), diag


def analyze_seasonality(
    scope_slice: ScopeSlice,
    *,
    target: str,
    granularity: Literal["season", "month"] = "season",
    min_group_size: int = 5,
) -> SeasonalityAnalysis:
    """Describe how a target behaves across seasons or months in one scope.

    This is a pure diagnostic layer that does not affect modeling.
    """
    diagnostics: list[str] = []
    _empty_stats = pd.DataFrame(columns=["group", "n_observations", "median", "q25", "q75", "mean", "std"])
    _empty_corr = pd.DataFrame(columns=["group", "feature", "corr", "n_shared", "method"])

    try:
        pivot = build_indicator_pivot(scope_slice.dataframe, aggregation_level="sample_point_level")
    except Exception as exc:
        diagnostics.append(f"Не удалось построить пивот: {exc}")
        return SeasonalityAnalysis(
            target=target,
            granularity=granularity,
            group_stats=_empty_stats,
            seasonal_pattern_detected=False,
            pattern_test={"test": "skipped", "reason": "pivot_build_failed"},
            per_season_correlations=_empty_corr,
            diagnostics=diagnostics,
        )

    group_col = granularity  # "season" or "month" — both in SAMPLE_POINT_INDEX
    if group_col not in pivot.columns:
        diagnostics.append(f"Столбец '{group_col}' не найден в данных.")
        return SeasonalityAnalysis(
            target=target,
            granularity=granularity,
            group_stats=_empty_stats,
            seasonal_pattern_detected=False,
            pattern_test={"test": "skipped", "reason": f"column_missing:{group_col}"},
            per_season_correlations=_empty_corr,
            diagnostics=diagnostics,
        )

    if target not in pivot.columns:
        diagnostics.append(f"Целевой показатель '{target}' не найден в пивоте.")
        return SeasonalityAnalysis(
            target=target,
            granularity=granularity,
            group_stats=_empty_stats,
            seasonal_pattern_detected=False,
            pattern_test={"test": "skipped", "reason": "target_missing"},
            per_season_correlations=_empty_corr,
            diagnostics=diagnostics,
        )

    group_stats, stats_diag = _group_stats_for_target(
        pivot, target=target, group_col=group_col, min_group_size=min_group_size
    )
    diagnostics.extend(stats_diag)

    seasonal_pattern_detected, pattern_test = _run_kruskal_wallis(
        pivot, target=target, group_col=group_col, min_group_size=min_group_size
    )

    per_season_correlations, corr_diag = _per_group_correlations(
        pivot, target=target, group_col=group_col, min_group_size=min_group_size
    )
    diagnostics.extend(corr_diag)

    return SeasonalityAnalysis(
        target=target,
        granularity=granularity,
        group_stats=group_stats,
        seasonal_pattern_detected=seasonal_pattern_detected,
        pattern_test=pattern_test,
        per_season_correlations=per_season_correlations,
        diagnostics=diagnostics,
    )
