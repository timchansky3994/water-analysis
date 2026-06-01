"""Unified correlation engine for scoped analytical slices."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from water_analysis.analysis.scopes import ScopeSlice
from water_analysis.io.schemas import REQUIRED_TARGETS
from water_analysis.preprocessing.pivot_builder import SAMPLE_POINT_INDEX, build_indicator_pivot


@dataclass(frozen=True)
class CorrelationDiagnostic:
    """Diagnostic record for skipped or unavailable correlation analysis."""

    scope_name: str
    scope_id: str
    scope_label: str
    target: str
    feature: str | None
    method: str | None
    status: str
    message: str
    n_shared: int

    def to_record(self) -> dict[str, object]:
        """Convert the diagnostic to a flat record."""
        return asdict(self)


@dataclass(frozen=True)
class CorrelationAnalysis:
    """Correlation results plus explicit diagnostics."""

    results: pd.DataFrame
    diagnostics: pd.DataFrame


def _indicator_columns(sample_point_pivot: pd.DataFrame) -> list[str]:
    """Return pivot columns corresponding to indicators."""
    return [column for column in sample_point_pivot.columns if column not in SAMPLE_POINT_INDEX]


def _calculate_correlation(method: str, x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Calculate correlation coefficient and p-value."""
    if method == "spearman":
        corr_value, p_value = spearmanr(x, y)
        return float(corr_value), float(p_value)
    if method == "pearson":
        corr_value, p_value = pearsonr(x, y)
        return float(corr_value), float(p_value)
    raise ValueError(f"Unsupported correlation method: {method}")


def run_correlation_analysis(
    scope_slices: Sequence[ScopeSlice],
    *,
    targets: Iterable[str] | None = None,
    methods: Sequence[str] = ("spearman",),
    min_shared_samples: int = 20,
) -> CorrelationAnalysis:
    """Run correlation analysis for one or more scope slices."""
    target_list = list(targets) if targets is not None else list(REQUIRED_TARGETS)
    results_records: list[dict[str, object]] = []
    diagnostic_records: list[dict[str, object]] = []

    for scope_slice in scope_slices:
        sample_point_pivot = build_indicator_pivot(scope_slice.dataframe, aggregation_level="sample_point_level")
        indicator_columns = _indicator_columns(sample_point_pivot)

        for target in target_list:
            if target not in sample_point_pivot.columns:
                diagnostic_records.append(
                    CorrelationDiagnostic(
                        scope_name=scope_slice.scope_name,
                        scope_id=scope_slice.scope_id,
                        scope_label=scope_slice.scope_label,
                        target=target,
                        feature=None,
                        method=None,
                        status="target_unavailable",
                        message=f"Target '{target}' is absent in this scope.",
                        n_shared=0,
                    ).to_record()
                )
                continue

            target_series = sample_point_pivot[target].dropna()
            if target_series.empty:
                diagnostic_records.append(
                    CorrelationDiagnostic(
                        scope_name=scope_slice.scope_name,
                        scope_id=scope_slice.scope_id,
                        scope_label=scope_slice.scope_label,
                        target=target,
                        feature=None,
                        method=None,
                        status="target_unavailable",
                        message=f"Target '{target}' has no numeric observations in this scope.",
                        n_shared=0,
                    ).to_record()
                )
                continue

            if target_series.nunique(dropna=True) <= 1:
                diagnostic_records.append(
                    CorrelationDiagnostic(
                        scope_name=scope_slice.scope_name,
                        scope_id=scope_slice.scope_id,
                        scope_label=scope_slice.scope_label,
                        target=target,
                        feature=None,
                        method=None,
                        status="target_insufficient_variance",
                        message=f"Target '{target}' is constant in this scope.",
                        n_shared=len(target_series),
                    ).to_record()
                )
                continue

            produced_result = False
            max_shared = 0
            for feature in indicator_columns:
                if feature == target:
                    continue

                valid = sample_point_pivot[[target, feature]].dropna()
                n_shared = len(valid)
                max_shared = max(max_shared, n_shared)
                if n_shared < min_shared_samples:
                    continue
                if valid[target].nunique(dropna=True) <= 1:
                    continue
                if valid[feature].nunique(dropna=True) <= 1:
                    continue

                for method in methods:
                    corr_value, p_value = _calculate_correlation(method, valid[target], valid[feature])
                    if pd.isna(corr_value) or pd.isna(p_value):
                        continue
                    results_records.append(
                        {
                            "scope_name": scope_slice.scope_name,
                            "scope_id": scope_slice.scope_id,
                            "scope_label": scope_slice.scope_label,
                            **scope_slice.selector,
                            "target": target,
                            "feature": feature,
                            "method": method,
                            "corr": corr_value,
                            "n_shared": n_shared,
                            "p_value": p_value,
                        }
                    )
                    produced_result = True

            if produced_result:
                continue

            if max_shared < min_shared_samples:
                diagnostic_records.append(
                    CorrelationDiagnostic(
                        scope_name=scope_slice.scope_name,
                        scope_id=scope_slice.scope_id,
                        scope_label=scope_slice.scope_label,
                        target=target,
                        feature=None,
                        method=None,
                        status="insufficient_shared_samples",
                        message=(
                            f"No predictors reached min_shared_samples={min_shared_samples} "
                            f"for target '{target}'."
                        ),
                        n_shared=max_shared,
                    ).to_record()
                )
            else:
                diagnostic_records.append(
                    CorrelationDiagnostic(
                        scope_name=scope_slice.scope_name,
                        scope_id=scope_slice.scope_id,
                        scope_label=scope_slice.scope_label,
                        target=target,
                        feature=None,
                        method=None,
                        status="no_valid_predictors",
                        message=f"No predictors with sufficient variance were found for target '{target}'.",
                        n_shared=max_shared,
                    ).to_record()
                )

    results_df = pd.DataFrame(results_records)
    if not results_df.empty:
        results_df = results_df.sort_values(
            ["scope_id", "target", "method", "n_shared", "corr"],
            ascending=[True, True, True, False, False],
        ).reset_index(drop=True)

    diagnostics_df = pd.DataFrame(diagnostic_records)
    if not diagnostics_df.empty:
        diagnostics_df = diagnostics_df.sort_values(["scope_id", "target", "status"]).reset_index(drop=True)

    return CorrelationAnalysis(results=results_df, diagnostics=diagnostics_df)
