import pandas as pd

from water_analysis.analysis.correlations import CorrelationAnalysis
from water_analysis.modeling.trainer import ModelComparisonRun, ModelResult
from water_analysis.profiling.passport import ProfileReport
from water_analysis.profiling.readiness import ReadinessAssessment, ReadinessIssue
from water_analysis.analysis.feature_selection import FeatureSelectionResult
from water_analysis.reporting.specialist_summary import SpecialistSummaryInput, build_specialist_summary


def _profile_report() -> ProfileReport:
    return ProfileReport(
        scope_name="global",
        scope_id="global",
        scope_label="Global dataset",
        summary={
            "scope_name": "global",
            "scope_id": "global",
            "scope_label": "Global dataset",
            "record_count": 100,
            "sample_event_count": 25,
            "unique_oktmo_count": 2,
            "unique_point_count": 4,
            "observation_start": "2018-01-01T00:00:00",
            "observation_end": "2018-12-31T00:00:00",
            "indicator_count": 5,
        },
        indicator_observations=pd.DataFrame(),
        missingness=pd.DataFrame(),
        point_type_coverage=pd.DataFrame(),
        constant_series=pd.DataFrame(),
        cooccurrence_matrix=pd.DataFrame(),
    )


def _correlations() -> CorrelationAnalysis:
    return CorrelationAnalysis(
        results=pd.DataFrame(
            [
                {
                    "scope_name": "global",
                    "scope_id": "global",
                    "scope_label": "Global dataset",
                    "target": "Жесткость общая",
                    "feature": "Цветность",
                    "method": "spearman",
                    "corr": 0.8,
                    "n_shared": 40,
                    "p_value": 0.001,
                }
            ]
        ),
        diagnostics=pd.DataFrame(),
    )


def test_specialist_summary_reports_baseline_better_case() -> None:
    readiness = ReadinessAssessment(
        scope_name="global",
        scope_id="global",
        scope_label="Global dataset",
        target="Жесткость общая",
        status="weakly_suitable",
        sample_point_rows=50,
        target_observation_count=40,
        target_missing_ratio=0.2,
        target_censored_ratio=0.0,
        eligible_predictor_count=2,
        max_shared_samples=40,
        issues=(ReadinessIssue(code="limited_observations", severity="warning", message="Small dataset."),),
    )
    baseline = ModelResult(
        model_name="median_baseline",
        model_family="baseline",
        is_baseline=True,
        status="fitted",
        metrics={"mae": 1.0, "rmse": 1.0, "r2": 0.2, "smape": 10.0},
        backtest_metrics={"mae": 1.1, "rmse": 1.2},
        feature_names=tuple(),
        beats_best_baseline=None,
        comparison_note="baseline_reference",
        notes=tuple(),
        interpretability_df=pd.DataFrame(),
    )
    ml = ModelResult(
        model_name="bayesian_ridge",
        model_family="linear",
        is_baseline=False,
        status="fitted",
        metrics={"mae": 1.2, "rmse": 1.4, "r2": 0.1, "smape": 12.0},
        backtest_metrics={"mae": 1.3, "rmse": 1.5},
        feature_names=("Цветность",),
        beats_best_baseline=False,
        comparison_note="ml_worse_than_best_baseline:median_baseline",
        notes=("Model underperforms best baseline.",),
        interpretability_df=pd.DataFrame([{"feature": "Цветность", "importance": 0.7}]),
    )
    run = ModelComparisonRun(
        scope_name="global",
        scope_id="global",
        target="Жесткость общая",
        readiness_assessment=readiness,
        selected_features=("Цветность",),
        feature_selection=type("Selection", (), {"selected_features": ("Цветность",), "candidate_table": pd.DataFrame()})(),
        best_baseline_name="median_baseline",
        results=[baseline, ml],
        comparison_df=pd.DataFrame([baseline.to_record(), ml.to_record()]),
        holdout_predictions_df=pd.DataFrame(),
        backtest_df=pd.DataFrame(),
        warnings=tuple(),
    )

    summary = build_specialist_summary(
        SpecialistSummaryInput(
            run_parameters={"scope": "global", "target": "Жесткость общая"},
            profile_report=_profile_report(),
            readiness_assessment=readiness,
            correlation_analysis=_correlations(),
            model_run=run,
        )
    )

    assert "ML-модель не лучше базовой модели" in summary
    assert "Расчетные значения" in summary
    assert "Цветность" in summary


def test_specialist_summary_reports_unsuitable_case() -> None:
    readiness = ReadinessAssessment(
        scope_name="global",
        scope_id="global",
        scope_label="Global dataset",
        target="Химическое потребление кислорода (ХПК)",
        status="unsuitable",
        sample_point_rows=50,
        target_observation_count=0,
        target_missing_ratio=1.0,
        target_censored_ratio=0.0,
        eligible_predictor_count=0,
        max_shared_samples=0,
        issues=(ReadinessIssue(code="target_unavailable", severity="critical", message="Target absent."),),
    )
    summary = build_specialist_summary(
        SpecialistSummaryInput(
            run_parameters={"scope": "global", "target": "ХПК"},
            profile_report=_profile_report(),
            readiness_assessment=readiness,
            correlation_analysis=CorrelationAnalysis(results=pd.DataFrame(), diagnostics=pd.DataFrame([{"status": "target_unavailable"}])),
            model_run=None,
        )
    )

    assert "обучение модели не рекомендуется" in summary
    assert "Целевой показатель отсутствует" in summary


def _suitable_readiness(target: str) -> ReadinessAssessment:
    return ReadinessAssessment(
        scope_name="global",
        scope_id="global",
        scope_label="Global dataset",
        target=target,
        status="suitable",
        sample_point_rows=60,
        target_observation_count=50,
        target_missing_ratio=0.1,
        target_censored_ratio=0.0,
        eligible_predictor_count=5,
        max_shared_samples=40,
        issues=tuple(),
    )


def _skipped_ml_result() -> ModelResult:
    return ModelResult(
        model_name="bayesian_ridge",
        model_family="ml",
        is_baseline=False,
        status="skipped_no_features",
        metrics={},
        backtest_metrics={},
        feature_names=tuple(),
        beats_best_baseline=False,
        comparison_note="ml_skipped_no_features",
        notes=("No predictors passed training-only feature selection.",),
        interpretability_df=pd.DataFrame(columns=["feature", "importance", "feature_kind"]),
    )


def _baseline_result() -> ModelResult:
    return ModelResult(
        model_name="median_baseline",
        model_family="baseline",
        is_baseline=True,
        status="fitted",
        metrics={"mae": 1.0, "rmse": 1.0, "r2": 0.0, "smape": 10.0},
        backtest_metrics={"mae": 1.1, "rmse": 1.2},
        feature_names=tuple(),
        beats_best_baseline=None,
        comparison_note="baseline_reference",
        notes=tuple(),
        interpretability_df=pd.DataFrame(),
        combined_score=1.1,
    )


def _run_with_selection(target: str, selection: FeatureSelectionResult) -> ModelComparisonRun:
    baseline = _baseline_result()
    skipped = _skipped_ml_result()
    return ModelComparisonRun(
        scope_name="global",
        scope_id="global",
        target=target,
        readiness_assessment=_suitable_readiness(target),
        selected_features=(),
        feature_selection=selection,
        best_baseline_name="median_baseline",
        results=[baseline, skipped],
        comparison_df=pd.DataFrame([baseline.to_record(), skipped.to_record()]),
        holdout_predictions_df=pd.DataFrame(),
        backtest_df=pd.DataFrame(),
        warnings=tuple(),
        selection_mode=selection.selection_mode,
        forced_features=selection.forced_features,
    )


def test_summary_explains_no_ml_auto_below_correlation() -> None:
    target = "Водородный показатель (pH)"
    candidate_table = pd.DataFrame(
        [
            {"feature": "A", "target_corr_abs": 0.24, "n_shared": 30, "exclusion_reason": "below_min_correlation"},
            {"feature": "B", "target_corr_abs": 0.10, "n_shared": 30, "exclusion_reason": "below_min_correlation"},
            {"feature": "C", "target_corr_abs": float("nan"), "n_shared": 25, "exclusion_reason": "constant_feature"},
        ]
    )
    selection = FeatureSelectionResult(
        target=target,
        selected_features=(),
        candidate_table=candidate_table,
        dropped_multicollinear=(),
        selection_mode="auto",
        forced_features=(),
    )
    summary = build_specialist_summary(
        SpecialistSummaryInput(
            run_parameters={"scope": "global", "target": target},
            profile_report=_profile_report(),
            readiness_assessment=_suitable_readiness(target),
            correlation_analysis=_correlations(),
            model_run=_run_with_selection(target, selection),
        )
    )
    assert "не прошёл автоматический отбор предикторов" in summary
    assert "корреляция с целевым показателем ниже требуемого порога" in summary
    assert "ρ≈0.24" in summary
    # The old generic phrasing must be gone.
    assert "ML-модель не была успешно обучена." not in summary


def test_summary_explains_no_ml_manual_too_few_shared() -> None:
    target = "Сульфаты (по SO4)"
    candidate_table = pd.DataFrame(
        [
            {"feature": "Железо", "target_corr_abs": float("nan"), "n_shared": 1, "exclusion_reason": "too_few_shared_samples"},
            {"feature": "Медь", "target_corr_abs": float("nan"), "n_shared": 3, "exclusion_reason": "too_few_shared_samples"},
        ]
    )
    selection = FeatureSelectionResult(
        target=target,
        selected_features=(),
        candidate_table=candidate_table,
        dropped_multicollinear=(),
        selection_mode="manual",
        forced_features=("Железо", "Медь"),
    )
    summary = build_specialist_summary(
        SpecialistSummaryInput(
            run_parameters={"scope": "global", "target": target},
            profile_report=_profile_report(),
            readiness_assessment=_suitable_readiness(target),
            correlation_analysis=_correlations(),
            model_run=_run_with_selection(target, selection),
        )
    )
    assert "нельзя использовать как предикторы" in summary
    assert "«Железо»" in summary
    assert "слишком мало совместных измерений" in summary
    assert "совместных измерений с целевым: 1" in summary
