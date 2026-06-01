"""Dataset profiling over analytical scopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from water_analysis.analysis.scopes import ScopeSlice
from water_analysis.preprocessing.pivot_builder import SAMPLE_POINT_INDEX, build_indicator_pivot

INDICATOR_OBSERVATION_COLUMNS = [
    "Indicator",
    "record_count",
    "numeric_observation_count",
    "missing_numeric_count",
    "censored_count",
    "missing_ratio",
    "censored_ratio",
]
MISSINGNESS_COLUMNS = ["Indicator", "total_rows", "missing_count", "missing_ratio"]
POINT_TYPE_COVERAGE_COLUMNS = ["PointType_Code", "record_count", "unique_points", "unique_oktmo", "indicator_count"]
CONSTANT_SERIES_COLUMNS = ["Indicator", "status", "n_observations", "n_unique", "dominant_share"]


@dataclass(frozen=True)
class ProfileReport:
    """Profile tables for one analytical scope."""

    scope_name: str
    scope_id: str
    scope_label: str
    summary: dict[str, object]
    indicator_observations: pd.DataFrame
    missingness: pd.DataFrame
    point_type_coverage: pd.DataFrame
    constant_series: pd.DataFrame
    cooccurrence_matrix: pd.DataFrame

    def summary_frame(self) -> pd.DataFrame:
        """Return the summary as a single-row dataframe."""
        return pd.DataFrame([self.summary])


def _augment_with_scope(df: pd.DataFrame, scope_slice: ScopeSlice) -> pd.DataFrame:
    """Attach scope metadata to a detail table."""
    if df.empty:
        return df.copy()
    return df.assign(
        scope_name=scope_slice.scope_name,
        scope_id=scope_slice.scope_id,
        scope_label=scope_slice.scope_label,
    )


def _indicator_columns(sample_point_pivot: pd.DataFrame) -> list[str]:
    """Return pivot columns corresponding to indicators."""
    return [column for column in sample_point_pivot.columns if column not in SAMPLE_POINT_INDEX]


def _build_indicator_observations(long_df: pd.DataFrame) -> pd.DataFrame:
    """Count observations and censoring by indicator."""
    if long_df.empty or "Indicator" not in long_df.columns:
        return pd.DataFrame(columns=INDICATOR_OBSERVATION_COLUMNS)
    grouped = (
        long_df.groupby("Indicator", dropna=False)
        .agg(
            record_count=("Indicator", "size"),
            numeric_observation_count=("Value_Approx", lambda series: int(series.notna().sum())),
            missing_numeric_count=("Value_Approx", lambda series: int(series.isna().sum())),
            censored_count=("IsCensored", lambda series: int(series.fillna(False).sum())),
        )
        .reset_index()
    )
    grouped["missing_ratio"] = grouped["missing_numeric_count"] / grouped["record_count"]
    grouped["censored_ratio"] = grouped["censored_count"] / grouped["record_count"]
    return grouped.sort_values(["record_count", "Indicator"], ascending=[False, True]).reset_index(drop=True)


def _build_missingness_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missingness by indicator."""
    if long_df.empty or "Indicator" not in long_df.columns:
        return pd.DataFrame(columns=MISSINGNESS_COLUMNS)
    grouped = (
        long_df.groupby("Indicator", dropna=False)["Value_Approx"]
        .agg(total_rows="size", missing_count=lambda series: int(series.isna().sum()))
        .reset_index()
    )
    grouped["missing_ratio"] = grouped["missing_count"] / grouped["total_rows"]
    return grouped.sort_values(["missing_ratio", "Indicator"], ascending=[False, True]).reset_index(drop=True)


def _build_point_type_coverage(long_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize coverage by point type."""
    if long_df.empty or "PointType_Code" not in long_df.columns:
        return pd.DataFrame(columns=POINT_TYPE_COVERAGE_COLUMNS)
    grouped = (
        long_df.groupby("PointType_Code", dropna=False)
        .agg(
            record_count=("PointType_Code", "size"),
            unique_points=("FullPointCode", "nunique"),
            unique_oktmo=("OKTMO", "nunique"),
            indicator_count=("Indicator", "nunique"),
        )
        .reset_index()
    )
    return grouped.sort_values(["record_count", "PointType_Code"], ascending=[False, True]).reset_index(drop=True)


def _build_constant_series_table(sample_point_pivot: pd.DataFrame, near_constant_threshold: float) -> pd.DataFrame:
    """Identify constant and near-constant indicator series."""
    rows: list[dict[str, object]] = []
    for indicator in _indicator_columns(sample_point_pivot):
        series = sample_point_pivot[indicator].dropna()
        if series.empty:
            rows.append(
                {
                    "Indicator": indicator,
                    "status": "no_observations",
                    "n_observations": 0,
                    "n_unique": 0,
                    "dominant_share": None,
                }
            )
            continue

        dominant_share = float(series.value_counts(normalize=True, dropna=True).iloc[0])
        n_unique = int(series.nunique(dropna=True))
        status = "variable"
        if n_unique <= 1:
            status = "constant"
        elif dominant_share >= near_constant_threshold:
            status = "near_constant"

        rows.append(
            {
                "Indicator": indicator,
                "status": status,
                "n_observations": int(series.size),
                "n_unique": n_unique,
                "dominant_share": dominant_share,
            }
        )

    if not rows:
        return pd.DataFrame(columns=CONSTANT_SERIES_COLUMNS)
    return pd.DataFrame(rows, columns=CONSTANT_SERIES_COLUMNS).sort_values(["status", "Indicator"]).reset_index(drop=True)


def _build_cooccurrence_matrix(sample_point_pivot: pd.DataFrame) -> pd.DataFrame:
    """Build a co-occurrence matrix counting shared non-null samples."""
    indicator_columns = _indicator_columns(sample_point_pivot)
    if not indicator_columns:
        return pd.DataFrame()

    availability = sample_point_pivot[indicator_columns].notna().astype(int)
    matrix = availability.T.dot(availability)
    matrix.index.name = "Indicator"
    matrix.columns.name = None
    return matrix


def build_profile_reports(
    scope_slices: Sequence[ScopeSlice],
    *,
    near_constant_threshold: float = 0.95,
) -> list[ProfileReport]:
    """Build profile reports for one or more scope slices."""
    reports: list[ProfileReport] = []

    for scope_slice in scope_slices:
        sample_point_pivot = build_indicator_pivot(scope_slice.dataframe, aggregation_level="sample_point_level")
        indicator_observations = _build_indicator_observations(scope_slice.dataframe)
        missingness = _build_missingness_table(scope_slice.dataframe)
        point_type_coverage = _build_point_type_coverage(scope_slice.dataframe)
        constant_series = _build_constant_series_table(sample_point_pivot, near_constant_threshold)
        cooccurrence_matrix = _build_cooccurrence_matrix(sample_point_pivot)

        observation_start = scope_slice.dataframe["SampleDate"].min()
        observation_end = scope_slice.dataframe["SampleDate"].max()

        summary = {
            "scope_name": scope_slice.scope_name,
            "scope_id": scope_slice.scope_id,
            "scope_label": scope_slice.scope_label,
            "record_count": int(len(scope_slice.dataframe)),
            "sample_event_count": int(scope_slice.dataframe[["SampleDate", "FullPointCode"]].drop_duplicates().shape[0]),
            "unique_oktmo_count": int(scope_slice.dataframe["OKTMO"].nunique(dropna=True)),
            "unique_point_count": int(scope_slice.dataframe["FullPointCode"].nunique(dropna=True)),
            "observation_start": observation_start.isoformat() if pd.notna(observation_start) else None,
            "observation_end": observation_end.isoformat() if pd.notna(observation_end) else None,
            "indicator_count": int(scope_slice.dataframe["Indicator"].nunique(dropna=True)),
            "sample_point_rows": int(len(sample_point_pivot)),
            "censored_ratio": float(scope_slice.dataframe["IsCensored"].fillna(False).mean()),
            "constant_series_count": int((constant_series.get("status", pd.Series(dtype=str)) == "constant").sum()),
            "near_constant_series_count": int((constant_series.get("status", pd.Series(dtype=str)) == "near_constant").sum()),
        }

        reports.append(
            ProfileReport(
                scope_name=scope_slice.scope_name,
                scope_id=scope_slice.scope_id,
                scope_label=scope_slice.scope_label,
                summary=summary,
                indicator_observations=_augment_with_scope(indicator_observations, scope_slice),
                missingness=_augment_with_scope(missingness, scope_slice),
                point_type_coverage=_augment_with_scope(point_type_coverage, scope_slice),
                constant_series=_augment_with_scope(constant_series, scope_slice),
                cooccurrence_matrix=cooccurrence_matrix,
            )
        )

    return reports
