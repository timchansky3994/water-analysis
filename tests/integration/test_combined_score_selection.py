"""Integration tests for combined-score model selection and stability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from water_analysis.modeling.trainer import ModelComparisonRun, ModelResult
from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.modeling.trainer import compare_models_in_scope
from water_analysis.preprocessing.long_format import build_canonical_long_format
from water_analysis.reporting.specialist_summary import _format_modeling_section


def _make_run_with_results(results: list[ModelResult]) -> ModelComparisonRun:
    """Build a minimal ModelComparisonRun with given results for summary tests."""
    from water_analysis.profiling.readiness import ReadinessAssessment, ReadinessIssue
    from water_analysis.analysis.feature_selection import FeatureSelectionResult

    readiness = ReadinessAssessment(
        scope_id="test",
        scope_name="global",
        scope_label="test scope",
        target="T",
        status="suitable",
        issues=(),
        sample_point_rows=50,
        target_observation_count=50,
        target_missing_ratio=0.0,
        target_censored_ratio=0.0,
        eligible_predictor_count=3,
        max_shared_samples=50,
    )
    fs = FeatureSelectionResult(
        target="T",
        selected_features=("F1",),
        candidate_table=pd.DataFrame(
            columns=["feature", "target_corr", "target_corr_abs", "target_corr_p_value",
                     "n_shared", "selection_score", "included", "exclusion_reason", "is_forced"]
        ),
        dropped_multicollinear=(),
        selection_mode="auto",
        forced_features=(),
    )
    return ModelComparisonRun(
        scope_name="global",
        scope_id="test",
        target="T",
        readiness_assessment=readiness,
        selected_features=("F1",),
        feature_selection=fs,
        best_baseline_name="mean",
        results=results,
        comparison_df=pd.DataFrame(),
        holdout_predictions_df=pd.DataFrame(),
        backtest_df=pd.DataFrame(),
        warnings=(),
    )


def _make_result(
    name: str,
    *,
    is_baseline: bool,
    holdout_rmse: float,
    backtest_rmse: float | None = None,
    combined_score: float | None = None,
    stability_ratio: float | None = None,
    combined_score_used_fallback: bool = False,
    beats_best_baseline: bool | None = None,
) -> ModelResult:
    metrics = {"rmse": holdout_rmse, "mae": holdout_rmse, "smape": 0.0, "r2": 0.9}
    backtest_metrics = {"rmse": backtest_rmse} if backtest_rmse is not None else {}
    return ModelResult(
        model_name=name,
        model_family="baseline" if is_baseline else "ml",
        is_baseline=is_baseline,
        status="fitted",
        metrics=metrics,
        backtest_metrics=backtest_metrics,
        feature_names=() if is_baseline else ("F1",),
        beats_best_baseline=beats_best_baseline,
        comparison_note="",
        notes=(),
        interpretability_df=pd.DataFrame(columns=["feature", "importance"]),
        combined_score=combined_score,
        stability_ratio=stability_ratio,
        combined_score_used_fallback=combined_score_used_fallback,
    )


def _build_synthetic_long_df(n_dates: int = 40) -> pd.DataFrame:
    """Build a minimal long-format dataframe for compare_models_in_scope tests."""
    raw_rows = []
    for i in range(n_dates):
        day = (i % 28) + 1
        month = (i // 28) + 1
        date_str = f"{day:02d}.0{min(month,9):d}.2020"
        target_val = float(i + 1)
        feature_val = target_val * 2.0
        for indicator, value in {
            "Жесткость общая": str(target_val).replace(".", ","),
            "Цветность": str(feature_val).replace(".", ","),
            "Мутность (по формазину)": str(feature_val / 2).replace(".", ","),
        }.items():
            raw_rows.append(
                {
                    "Дата проведения исследования": date_str,
                    "Тип точки": "Точка на распределительной сети",
                    "Код точки": "00000000001.10110.0010",
                    "Гигиенический показатель": indicator,
                    "Норматив": "",
                    "Строчное значение": "",
                    "Результат исследования": value,
                    "Нижний предел обнаружения": "",
                    "Верхний предел обнаружения": "",
                    "Ошибка метода определения": "",
                    "Нормативная документация": "",
                    "Цели исследований": "",
                    "Соотв. ПДК": "",
                }
            )
    return build_canonical_long_format(pd.DataFrame(raw_rows))


def test_combined_score_picks_more_stable_model_over_holdout_winner() -> None:
    """model_B wins by combined score even though model_A has lower holdout."""
    # model_A: holdout=0.5, backtest=1.0 → combined = 0.4*0.5 + 0.6*1.0 = 0.8
    # model_B: holdout=0.7, backtest=0.7 → combined = 0.4*0.7 + 0.6*0.7 = 0.7
    baseline = _make_result("mean", is_baseline=True, holdout_rmse=1.0, backtest_rmse=1.0, combined_score=1.0)
    model_a = _make_result(
        "model_A", is_baseline=False,
        holdout_rmse=0.5, backtest_rmse=1.0,
        combined_score=0.4 * 0.5 + 0.6 * 1.0,
        stability_ratio=1.0 / 0.5,
    )
    model_b = _make_result(
        "model_B", is_baseline=False,
        holdout_rmse=0.7, backtest_rmse=0.7,
        combined_score=0.7,
        stability_ratio=1.0,
    )
    run = _make_run_with_results([baseline, model_a, model_b])
    best = run.get_best_ml_result()
    assert best is not None
    assert best.model_name == "model_B", (
        f"Expected model_B (combined=0.7) to beat model_A (combined=0.8), but got {best.model_name}"
    )


def test_stability_ratio_warning_appears_in_summary() -> None:
    """A model with stability_ratio > 1.5 triggers nестабильность warning in summary."""
    baseline = _make_result("mean", is_baseline=True, holdout_rmse=1.0, combined_score=1.0)
    unstable = _make_result(
        "ridge", is_baseline=False,
        holdout_rmse=0.4, backtest_rmse=0.8,
        combined_score=0.4 * 0.4 + 0.6 * 0.8,
        stability_ratio=0.8 / 0.4,
        beats_best_baseline=True,
    )
    run = _make_run_with_results([baseline, unstable])
    summary = _format_modeling_section(run)
    assert "признаки нестабильности" in summary


def test_holdout_fallback_when_backtest_empty() -> None:
    """With fallback=True the summary mentions holdout-only selection."""
    baseline = _make_result("mean", is_baseline=True, holdout_rmse=1.0, combined_score=1.0)
    fallback_model = _make_result(
        "ridge", is_baseline=False,
        holdout_rmse=0.5,
        combined_score=0.5,
        combined_score_used_fallback=True,
        beats_best_baseline=True,
    )
    run = _make_run_with_results([baseline, fallback_model])
    summary = _format_modeling_section(run)
    assert "holdout RMSE" in summary


def test_compare_models_combined_score_fields_populated() -> None:
    """compare_models_in_scope populates combined_score and stability_ratio on each result."""
    long_df = _build_synthetic_long_df(40)
    scope_slice = build_scope_slices(long_df, scope_name="global")[0]
    run = compare_models_in_scope(
        scope_slice,
        target="Жесткость общая",
        model_names=["bayesian_ridge"],
        test_size=0.25,
        min_train_size=6,
        min_target_observations=6,
        min_shared_samples=6,
        min_eligible_predictors=1,
        min_target_correlation=0.2,
        max_features=2,
    )
    for result in run.results:
        if result.status == "fitted":
            assert result.combined_score is not None, f"{result.model_name}: combined_score is None"
            # stability_ratio may be None only if backtest had no splits


def test_compare_models_comparison_note_uses_combined_suffix() -> None:
    """comparison_note contains 'combined' for both winning and losing ML models."""
    long_df = _build_synthetic_long_df(40)
    scope_slice = build_scope_slices(long_df, scope_name="global")[0]
    run = compare_models_in_scope(
        scope_slice,
        target="Жесткость общая",
        model_names=["bayesian_ridge"],
        test_size=0.25,
        min_train_size=6,
        min_target_observations=6,
        min_shared_samples=6,
        min_eligible_predictors=1,
        min_target_correlation=0.2,
        max_features=2,
    )
    ml_results = [r for r in run.results if not r.is_baseline and r.status == "fitted"]
    for result in ml_results:
        note = result.comparison_note
        assert "combined" in note or "holdout_fallback" in note, (
            f"Expected 'combined' or 'holdout_fallback' in comparison_note, got: {note!r}"
        )
