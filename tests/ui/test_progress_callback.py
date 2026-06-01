"""Tests: generate_report_bundle calls progress_callback in ascending order."""

from __future__ import annotations

from pathlib import Path

from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.io.schemas import REQUIRED_TARGETS
from water_analysis.preprocessing.long_format import build_canonical_long_format
from water_analysis.reporting.bundle import generate_report_bundle

from tests.helpers import build_linear_raw_dataframe


def test_progress_callback_called_in_ascending_order(tmp_path: Path) -> None:
    raw_df = build_linear_raw_dataframe(
        dates=[f"{day:02d}.01.2018" for day in range(1, 15)],
        target_values=[float(day) for day in range(1, 15)],
        feature_map={
            REQUIRED_TARGETS[1]: [float(day * 2) for day in range(1, 15)],
            REQUIRED_TARGETS[2]: [float(day) / 2 for day in range(1, 15)],
        },
    )
    long_df = build_canonical_long_format(raw_df)
    scope_slice = build_scope_slices(long_df, scope_name="global")[0]

    calls: list[tuple[str, float]] = []

    def callback(stage: str, fraction: float) -> None:
        calls.append((stage, fraction))

    generate_report_bundle(
        scope_slice,
        target=REQUIRED_TARGETS[0],
        output_dir=tmp_path / "bundle_progress",
        run_parameters={"scope": "global", "target": REQUIRED_TARGETS[0]},
        correlation_methods=["spearman"],
        correlation_min_shared_samples=4,
        readiness_kwargs={
            "min_target_observations": 6,
            "min_shared_samples": 4,
            "max_missing_ratio": 0.6,
            "heavy_censoring_ratio": 0.5,
            "min_eligible_predictors": 1,
        },
        modeling_kwargs={
            "model_names": ["bayesian_ridge"],
            "test_size": 0.3,
            "min_train_size": 6,
            "min_target_observations": 6,
            "min_shared_samples": 4,
            "max_missing_ratio": 0.6,
            "heavy_censoring_ratio": 0.5,
            "min_eligible_predictors": 1,
            "min_target_correlation": 0.1,
            "max_features": 5,
            "multicollinearity_threshold": 0.85,
        },
        reporting_kwargs={"plots_dpi": 72, "top_correlations": 5, "heatmap_max_features": 10},
        progress_callback=callback,
    )

    assert len(calls) >= 3, "Expected at least 3 progress callback invocations"

    fractions = [f for _, f in calls]
    # Each fraction must be >= the previous (non-decreasing)
    for prev, current in zip(fractions, fractions[1:]):
        assert current >= prev, f"Fractions must be non-decreasing: {prev} -> {current}"

    # Final call must be 1.0 (Готово)
    assert calls[-1][0] == "Готово"
    assert calls[-1][1] == 1.0


def test_progress_callback_none_does_not_crash(tmp_path: Path) -> None:
    raw_df = build_linear_raw_dataframe(
        dates=["01.01.2018", "02.01.2018", "03.01.2018"],
        target_values=[1.0, 2.0, 3.0],
        feature_map={"Цветность": [4.0, 5.0, 6.0]},
    )
    long_df = build_canonical_long_format(raw_df)
    scope_slice = build_scope_slices(long_df, scope_name="global")[0]

    bundle = generate_report_bundle(
        scope_slice,
        target="Жесткость общая",
        output_dir=tmp_path / "bundle_no_cb",
        run_parameters={},
        correlation_methods=["spearman"],
        correlation_min_shared_samples=2,
        readiness_kwargs={
            "min_target_observations": 2,
            "min_shared_samples": 2,
            "max_missing_ratio": 0.6,
            "heavy_censoring_ratio": 0.5,
            "min_eligible_predictors": 1,
        },
        modeling_kwargs={
            "model_names": ["bayesian_ridge"],
            "test_size": 0.5,
            "min_train_size": 2,
            "min_target_observations": 2,
            "min_shared_samples": 2,
            "max_missing_ratio": 0.6,
            "heavy_censoring_ratio": 0.5,
            "min_eligible_predictors": 1,
            "min_target_correlation": 0.1,
            "max_features": 5,
            "multicollinearity_threshold": 0.85,
        },
        reporting_kwargs={"plots_dpi": 72, "top_correlations": 5, "heatmap_max_features": 10},
        progress_callback=None,
    )
    assert bundle.output_dir.exists()
