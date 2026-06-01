"""Readiness and suitability checks for modeling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import pandas as pd

from water_analysis.analysis.scopes import ScopeSlice
from water_analysis.io.schemas import REQUIRED_TARGETS
from water_analysis.preprocessing.pivot_builder import SAMPLE_POINT_INDEX, build_indicator_pivot


@dataclass(frozen=True)
class ReadinessIssue:
    """One readiness warning or blocking issue."""

    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class ReadinessAssessment:
    """Readiness assessment for one target within one scope."""

    scope_name: str
    scope_id: str
    scope_label: str
    target: str
    status: str
    sample_point_rows: int
    target_observation_count: int
    target_missing_ratio: float
    target_censored_ratio: float
    eligible_predictor_count: int
    max_shared_samples: int
    issues: tuple[ReadinessIssue, ...]

    def to_record(self) -> dict[str, object]:
        """Convert the assessment to a flat record for reporting."""
        issue_codes = "|".join(issue.code for issue in self.issues)
        issue_messages = " | ".join(issue.message for issue in self.issues)
        return {
            **asdict(self),
            "issues": None,
            "issue_codes": issue_codes,
            "issue_messages": issue_messages,
        }


def _indicator_columns(sample_point_pivot: pd.DataFrame) -> list[str]:
    """Return pivot columns corresponding to indicators."""
    return [column for column in sample_point_pivot.columns if column not in SAMPLE_POINT_INDEX]


def assess_readiness(
    scope_slices: Sequence[ScopeSlice],
    *,
    targets: Iterable[str] | None = None,
    min_target_observations: int = 30,
    min_shared_samples: int = 20,
    max_missing_ratio: float = 0.6,
    heavy_censoring_ratio: float = 0.5,
    min_eligible_predictors: int = 2,
    near_constant_threshold: float = 0.95,
) -> list[ReadinessAssessment]:
    """Assess modeling suitability for one or more targets across scopes."""
    target_list = list(targets) if targets is not None else list(REQUIRED_TARGETS)
    assessments: list[ReadinessAssessment] = []

    for scope_slice in scope_slices:
        sample_point_pivot = build_indicator_pivot(scope_slice.dataframe, aggregation_level="sample_point_level")
        indicator_columns = _indicator_columns(sample_point_pivot)
        sample_point_rows = int(len(sample_point_pivot))

        for target in target_list:
            issues: list[ReadinessIssue] = []
            target_source_rows = scope_slice.dataframe[scope_slice.dataframe["Indicator"] == target]
            target_censored_ratio = float(target_source_rows["IsCensored"].fillna(False).mean()) if not target_source_rows.empty else 0.0

            if target not in sample_point_pivot.columns:
                assessments.append(
                    ReadinessAssessment(
                        scope_name=scope_slice.scope_name,
                        scope_id=scope_slice.scope_id,
                        scope_label=scope_slice.scope_label,
                        target=target,
                        status="unsuitable",
                        sample_point_rows=sample_point_rows,
                        target_observation_count=0,
                        target_missing_ratio=1.0,
                        target_censored_ratio=target_censored_ratio,
                        eligible_predictor_count=0,
                        max_shared_samples=0,
                        issues=(
                            ReadinessIssue(
                                code="target_unavailable",
                                severity="critical",
                                message=f"Target '{target}' is absent in the source data for this scope.",
                            ),
                        ),
                    )
                )
                continue

            target_series = sample_point_pivot[target]
            target_valid = target_series.dropna()
            target_observation_count = int(target_valid.size)
            target_missing_ratio = float(1.0 - (target_observation_count / sample_point_rows)) if sample_point_rows else 1.0

            if target_observation_count == 0:
                issues.append(
                    ReadinessIssue(
                        code="target_unavailable",
                        severity="critical",
                        message=f"Target '{target}' has no numeric observations after parsing.",
                    )
                )
            elif target_observation_count < min_shared_samples:
                issues.append(
                    ReadinessIssue(
                        code="too_few_observations",
                        severity="critical",
                        message=(
                            f"Target '{target}' has only {target_observation_count} observations, "
                            f"below min_shared_samples={min_shared_samples}."
                        ),
                    )
                )
            elif target_observation_count < min_target_observations:
                issues.append(
                    ReadinessIssue(
                        code="limited_observations",
                        severity="warning",
                        message=(
                            f"Target '{target}' has {target_observation_count} observations, "
                            f"below preferred min_target_observations={min_target_observations}."
                        ),
                    )
                )

            if target_observation_count > 0:
                n_unique = int(target_valid.nunique(dropna=True))
                dominant_share = float(target_valid.value_counts(normalize=True, dropna=True).iloc[0])
                if n_unique <= 1:
                    issues.append(
                        ReadinessIssue(
                            code="target_constant",
                            severity="critical",
                            message=f"Target '{target}' is constant in this scope.",
                        )
                    )
                elif dominant_share >= near_constant_threshold:
                    issues.append(
                        ReadinessIssue(
                            code="target_near_constant",
                            severity="warning",
                            message=(
                                f"Target '{target}' is near-constant "
                                f"(dominant value share {dominant_share:.2f})."
                            ),
                        )
                    )

            if target_missing_ratio > 0.85:
                issues.append(
                    ReadinessIssue(
                        code="extreme_missingness",
                        severity="critical",
                        message=f"Target '{target}' missing ratio is {target_missing_ratio:.2f}.",
                    )
                )
            elif target_missing_ratio > max_missing_ratio:
                issues.append(
                    ReadinessIssue(
                        code="high_missingness",
                        severity="warning",
                        message=f"Target '{target}' missing ratio is {target_missing_ratio:.2f}.",
                    )
                )

            if target_censored_ratio > heavy_censoring_ratio:
                issues.append(
                    ReadinessIssue(
                        code="heavy_censoring",
                        severity="warning",
                        message=f"Target '{target}' censoring ratio is {target_censored_ratio:.2f}.",
                    )
                )

            eligible_predictor_count = 0
            max_shared_samples = 0
            if target in sample_point_pivot.columns:
                for feature in indicator_columns:
                    if feature == target:
                        continue
                    valid = sample_point_pivot[[target, feature]].dropna()
                    shared_count = int(len(valid))
                    max_shared_samples = max(max_shared_samples, shared_count)
                    if shared_count < min_shared_samples:
                        continue
                    if valid[feature].nunique(dropna=True) <= 1:
                        continue
                    eligible_predictor_count += 1

            if max_shared_samples < min_shared_samples:
                issues.append(
                    ReadinessIssue(
                        code="low_shared_measurements",
                        severity="critical",
                        message=(
                            f"Predictors do not reach min_shared_samples={min_shared_samples} "
                            f"with target '{target}'."
                        ),
                    )
                )
            elif eligible_predictor_count < min_eligible_predictors:
                issues.append(
                    ReadinessIssue(
                        code="weak_predictor_availability",
                        severity="warning",
                        message=(
                            f"Only {eligible_predictor_count} predictors have enough shared measurements "
                            f"for target '{target}'."
                        ),
                    )
                )

            has_critical = any(issue.severity == "critical" for issue in issues)
            has_warning = any(issue.severity == "warning" for issue in issues)
            status = "unsuitable" if has_critical else "weakly_suitable" if has_warning else "suitable"

            assessments.append(
                ReadinessAssessment(
                    scope_name=scope_slice.scope_name,
                    scope_id=scope_slice.scope_id,
                    scope_label=scope_slice.scope_label,
                    target=target,
                    status=status,
                    sample_point_rows=sample_point_rows,
                    target_observation_count=target_observation_count,
                    target_missing_ratio=target_missing_ratio,
                    target_censored_ratio=target_censored_ratio,
                    eligible_predictor_count=eligible_predictor_count,
                    max_shared_samples=max_shared_samples,
                    issues=tuple(issues),
                )
            )

    return assessments
