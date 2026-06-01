"""Model comparison and training workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

LOGGER = logging.getLogger(__name__)

from water_analysis.analysis.feature_selection import (
    FeatureSelectionResult,
    prepare_modeling_frame,
    select_predictors,
)
from water_analysis.analysis.scopes import ScopeSlice
from water_analysis.analysis.seasonal_features import build_seasonal_features, seasonal_feature_names as _seasonal_names
from water_analysis.modeling.baselines import baseline_registry
from water_analysis.modeling.registry import available_model_specs
from water_analysis.modeling.validation import (
    BacktestSplit,
    chronological_holdout_split,
    compute_combined_score,
    compute_regression_metrics,
    compute_stability_ratio,
    rolling_backtest_splits,
)
from water_analysis.profiling.readiness import ReadinessAssessment, assess_readiness

PRIMARY_METRIC = "rmse"


class ModelingNotAllowedError(RuntimeError):
    """Raised when readiness explicitly blocks model training."""

    def __init__(self, assessment: ReadinessAssessment) -> None:
        super().__init__(
            f"Model training blocked for target '{assessment.target}' in scope '{assessment.scope_id}': "
            f"{assessment.status} ({assessment.to_record()['issue_codes']})"
        )
        self.assessment = assessment


@dataclass
class ModelResult:
    """Evaluation result for one baseline or ML model."""

    model_name: str
    model_family: str
    is_baseline: bool
    status: str
    metrics: dict[str, float]
    backtest_metrics: dict[str, float]
    feature_names: tuple[str, ...]
    beats_best_baseline: bool | None
    comparison_note: str
    notes: tuple[str, ...]
    interpretability_df: pd.DataFrame
    estimator: Any | None = None
    combined_score: float | None = None
    stability_ratio: float | None = None
    combined_score_used_fallback: bool = False
    seasonal_features: tuple[str, ...] = field(default_factory=tuple)

    def to_record(self) -> dict[str, object]:
        """Convert the result to a flat summary record."""
        record: dict[str, object] = {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "is_baseline": self.is_baseline,
            "status": self.status,
            "selected_feature_count": len(self.feature_names),
            "selected_features": "|".join(self.feature_names),
            "beats_best_baseline": self.beats_best_baseline,
            "comparison_note": self.comparison_note,
            "combined_score": self.combined_score,
            "stability_ratio": self.stability_ratio,
            "combined_score_used_fallback": self.combined_score_used_fallback,
            "notes": " | ".join(self.notes),
        }
        for metric_name, metric_value in self.metrics.items():
            record[f"holdout_{metric_name}"] = metric_value
        for metric_name, metric_value in self.backtest_metrics.items():
            record[f"backtest_{metric_name}"] = metric_value
        if not self.interpretability_df.empty:
            record["top_interpretation"] = " | ".join(
                f"{row.feature}={row.importance:.4f}" for row in self.interpretability_df.itertuples(index=False)
            )
        else:
            record["top_interpretation"] = ""
        return record


@dataclass
class ModelComparisonRun:
    """Full comparison output for one target and scope."""

    scope_name: str
    scope_id: str
    target: str
    readiness_assessment: ReadinessAssessment
    selected_features: tuple[str, ...]
    feature_selection: FeatureSelectionResult
    best_baseline_name: str | None
    results: list[ModelResult]
    comparison_df: pd.DataFrame
    holdout_predictions_df: pd.DataFrame
    backtest_df: pd.DataFrame
    warnings: tuple[str, ...]
    scope_selectors: dict[str, str] | None = None
    aggregation_level: str = "sample_point_level"
    training_period_start: str | None = None
    training_period_end: str | None = None
    train_rows: int = 0
    holdout_rows: int = 0
    selection_mode: str = "auto"
    forced_features: tuple[str, ...] = field(default_factory=tuple)
    combined_score_weight_holdout: float = 0.4
    combined_score_weight_backtest: float = 0.6
    seasonal_feature: str = "none"
    seasonal_features: tuple[str, ...] = field(default_factory=tuple)

    def get_result(self, model_name: str) -> ModelResult | None:
        """Return a model result by name."""
        return next((result for result in self.results if result.model_name == model_name), None)

    def get_best_baseline_result(self) -> ModelResult | None:
        """Return the best fitted baseline result by combined score (fallback to holdout metric)."""
        fitted = [result for result in self.results if result.is_baseline and result.status == "fitted"]
        if not fitted:
            return None
        with_score = [r for r in fitted if r.combined_score is not None]
        if with_score:
            return min(with_score, key=lambda r: r.combined_score)  # type: ignore[arg-type]
        LOGGER.warning("No combined_score available for baselines; falling back to holdout %s.", PRIMARY_METRIC)
        return min(fitted, key=lambda r: r.metrics[PRIMARY_METRIC])

    def get_best_ml_result(self) -> ModelResult | None:
        """Return the best fitted ML result by combined score (fallback to holdout metric)."""
        fitted = [result for result in self.results if not result.is_baseline and result.status == "fitted"]
        if not fitted:
            return None
        with_score = [r for r in fitted if r.combined_score is not None]
        if with_score:
            return min(with_score, key=lambda r: r.combined_score)  # type: ignore[arg-type]
        LOGGER.warning("No combined_score available for ML models; falling back to holdout %s.", PRIMARY_METRIC)
        return min(fitted, key=lambda r: r.metrics[PRIMARY_METRIC])

    def get_preferred_result_for_reporting(self) -> ModelResult | None:
        """Return the result to visualize in reports."""
        best_ml = self.get_best_ml_result()
        if best_ml is not None:
            return best_ml
        return self.get_best_baseline_result()


def _build_supervised_frame(scope_slice: ScopeSlice, target: str) -> pd.DataFrame:
    """Prepare a target-specific modeling frame with chronological ordering."""
    frame = prepare_modeling_frame(scope_slice.dataframe)
    if target not in frame.columns:
        return frame.iloc[0:0].copy()
    supervised = frame[frame[target].notna()].copy()
    return supervised.reset_index(drop=True)


def _interpretability_table(
    estimator: Any,
    *,
    feature_names: Sequence[str],
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    seasonal_feature_names: Sequence[str] = (),
) -> pd.DataFrame:
    """Extract model interpretation signals for downstream reporting."""
    if estimator is None or not feature_names:
        return pd.DataFrame(columns=["feature", "importance", "feature_kind"])

    model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    seasonal_set = set(seasonal_feature_names)

    def _with_kind(table: pd.DataFrame) -> pd.DataFrame:
        table["feature_kind"] = ["seasonal" if f in seasonal_set else "indicator" for f in table["feature"]]
        return table

    if hasattr(model, "coef_"):
        coefficients = np.ravel(np.asarray(model.coef_, dtype=float))
        table = pd.DataFrame({"feature": list(feature_names), "importance": coefficients})
        table = table.reindex(table["importance"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        return _with_kind(table)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        table = pd.DataFrame({"feature": list(feature_names), "importance": importances})
        table = table.sort_values("importance", ascending=False).reset_index(drop=True)
        return _with_kind(table)

    if len(X_reference) >= 3:
        permutation = permutation_importance(
            estimator,
            X_reference,
            y_reference,
            n_repeats=10,
            random_state=42,
        )
        table = pd.DataFrame({"feature": list(feature_names), "importance": permutation.importances_mean})
        table = table.sort_values("importance", ascending=False).reset_index(drop=True)
        return _with_kind(table)

    return pd.DataFrame(columns=["feature", "importance", "feature_kind"])


def _make_holdout_predictions(
    frame: pd.DataFrame,
    *,
    target: str,
    model_name: str,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Build a standardized holdout prediction table."""
    return pd.DataFrame(
        {
            "SampleDate": frame["SampleDate"],
            "FullPointCode": frame["FullPointCode"],
            "target": target,
            "actual": frame[target].to_numpy(dtype=float),
            "predicted": y_pred.astype(float),
            "model_name": model_name,
        }
    )


def _aggregate_backtest_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate metrics across rolling backtest splits."""
    if not metric_rows:
        return {}
    backtest_df = pd.DataFrame(metric_rows)
    return {column: float(backtest_df[column].mean()) for column in backtest_df.columns}


def _evaluate_baseline(
    model_name: str,
    *,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target: str,
    backtest_splits: Sequence[BacktestSplit],
    warnings: Sequence[str],
) -> tuple[ModelResult, pd.DataFrame]:
    """Fit and evaluate one baseline model."""
    baseline_cls = baseline_registry()[model_name]
    baseline = baseline_cls().fit(train_frame, target)
    y_pred = baseline.predict(test_frame)
    metrics = compute_regression_metrics(test_frame[target].to_numpy(dtype=float), y_pred)

    backtest_rows: list[dict[str, float]] = []
    for split in backtest_splits:
        split_baseline = baseline_cls().fit(split.train_frame, target)
        split_pred = split_baseline.predict(split.test_frame)
        backtest_rows.append(
            compute_regression_metrics(split.test_frame[target].to_numpy(dtype=float), split_pred)
        )

    result = ModelResult(
        model_name=model_name,
        model_family="baseline",
        is_baseline=True,
        status="fitted",
        metrics=metrics,
        backtest_metrics=_aggregate_backtest_metrics(backtest_rows),
        feature_names=tuple(),
        beats_best_baseline=None,
        comparison_note="baseline_reference",
        notes=tuple(warnings),
        interpretability_df=pd.DataFrame(columns=["feature", "importance"]),
        estimator=baseline,
    )
    return result, _make_holdout_predictions(test_frame, target=target, model_name=model_name, y_pred=y_pred)


def _evaluate_ml_model(
    model_name: str,
    *,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target: str,
    feature_selection: FeatureSelectionResult,
    backtest_splits: Sequence[BacktestSplit],
    selection_params: dict[str, Any],
    warnings: Sequence[str],
    seasonal_feature: str = "none",
) -> tuple[ModelResult, pd.DataFrame]:
    """Fit and evaluate one ML model."""
    sfnames = tuple(_seasonal_names(seasonal_feature))  # type: ignore[arg-type]

    if not feature_selection.selected_features and not sfnames:
        result = ModelResult(
            model_name=model_name,
            model_family="ml",
            is_baseline=False,
            status="skipped_no_features",
            metrics={},
            backtest_metrics={},
            feature_names=tuple(),
            beats_best_baseline=False,
            comparison_note="ml_skipped_no_features",
            notes=tuple(warnings) + ("No predictors passed training-only feature selection.",),
            interpretability_df=pd.DataFrame(columns=["feature", "importance", "feature_kind"]),
            estimator=None,
            seasonal_features=sfnames,
        )
        return result, pd.DataFrame(columns=["SampleDate", "FullPointCode", "target", "actual", "predicted", "model_name"])

    # Add seasonal columns to train/test frames (after feature selection — seasonal features
    # don't participate in correlation-based feature selection).
    train_aug, _ = build_seasonal_features(train_frame, seasonal_feature)  # type: ignore[arg-type]
    test_aug, _ = build_seasonal_features(test_frame, seasonal_feature)  # type: ignore[arg-type]

    indicator_names = list(feature_selection.selected_features)
    all_feature_names = indicator_names + list(sfnames)

    specs = available_model_specs()
    estimator = specs[model_name].builder()
    estimator.fit(train_aug[all_feature_names], train_aug[target].to_numpy(dtype=float))
    y_pred = estimator.predict(test_aug[all_feature_names])
    metrics = compute_regression_metrics(test_aug[target].to_numpy(dtype=float), y_pred)

    backtest_rows: list[dict[str, float]] = []
    for split in backtest_splits:
        split_aug_train, _ = build_seasonal_features(split.train_frame, seasonal_feature)  # type: ignore[arg-type]
        split_aug_test, _ = build_seasonal_features(split.test_frame, seasonal_feature)  # type: ignore[arg-type]
        split_selection = select_predictors(split.train_frame, target=target, **selection_params)
        split_indicator = list(split_selection.selected_features) if split_selection.selected_features else []
        split_features = split_indicator + list(sfnames)
        if not split_features:
            continue
        split_estimator = specs[model_name].builder()
        split_estimator.fit(split_aug_train[split_features], split_aug_train[target].to_numpy(dtype=float))
        split_pred = split_estimator.predict(split_aug_test[split_features])
        backtest_rows.append(compute_regression_metrics(split_aug_test[target].to_numpy(dtype=float), split_pred))

    interpretability_df = _interpretability_table(
        estimator,
        feature_names=all_feature_names,
        X_reference=test_aug[all_feature_names],
        y_reference=test_aug[target],
        seasonal_feature_names=sfnames,
    )
    result = ModelResult(
        model_name=model_name,
        model_family=specs[model_name].family,
        is_baseline=False,
        status="fitted",
        metrics=metrics,
        backtest_metrics=_aggregate_backtest_metrics(backtest_rows),
        feature_names=tuple(all_feature_names),
        beats_best_baseline=False,
        comparison_note="",
        notes=tuple(warnings),
        interpretability_df=interpretability_df,
        estimator=estimator,
        seasonal_features=sfnames,
    )
    return result, _make_holdout_predictions(test_aug, target=target, model_name=model_name, y_pred=y_pred)


def compare_models_in_scope(
    scope_slice: ScopeSlice,
    *,
    target: str,
    model_names: Sequence[str] | None = None,
    test_size: float = 0.2,
    min_train_size: int = 20,
    backtest_initial_train_size: int | None = None,
    backtest_test_window_size: int | None = None,
    backtest_step_size: int | None = None,
    min_target_observations: int = 30,
    min_shared_samples: int = 20,
    max_missing_ratio: float = 0.6,
    heavy_censoring_ratio: float = 0.5,
    min_eligible_predictors: int = 2,
    min_target_correlation: float = 0.3,
    significance_alpha: float = 0.05,
    max_features: int = 5,
    multicollinearity_threshold: float = 0.85,
    selection_mode: Literal["auto", "manual", "semi_auto"] = "auto",
    forced_features: Sequence[str] = (),
    combined_score_weight_holdout: float = 0.4,
    combined_score_weight_backtest: float = 0.6,
    seasonal_feature: Literal["none", "season", "month"] = "none",
) -> ModelComparisonRun:
    """Compare baselines and ML models within one analytical scope."""
    if combined_score_weight_holdout < 0 or combined_score_weight_backtest < 0:
        raise ValueError("combined_score weights must be >= 0.")
    total_weight = combined_score_weight_holdout + combined_score_weight_backtest
    if total_weight == 0:
        raise ValueError("combined_score weights must not both be zero.")
    if abs(total_weight - 1.0) > 1e-9:
        LOGGER.warning(
            "combined_score weights sum to %.4f (not 1.0); scores are not normalized.",
            total_weight,
        )

    readiness_assessment = assess_readiness(
        [scope_slice],
        targets=[target],
        min_target_observations=min_target_observations,
        min_shared_samples=min_shared_samples,
        max_missing_ratio=max_missing_ratio,
        heavy_censoring_ratio=heavy_censoring_ratio,
        min_eligible_predictors=min_eligible_predictors,
    )[0]
    if readiness_assessment.status == "unsuitable":
        raise ModelingNotAllowedError(readiness_assessment)

    warnings: list[str] = []
    if readiness_assessment.status == "weakly_suitable":
        warnings.append(
            f"Scope is only weakly suitable for modeling: {readiness_assessment.to_record()['issue_codes']}"
        )

    supervised_frame = _build_supervised_frame(scope_slice, target)
    holdout = chronological_holdout_split(supervised_frame, test_size=test_size, min_train_size=min_train_size)

    selection_params: dict[str, Any] = {
        "mode": selection_mode,
        "forced_features": tuple(forced_features),
        "min_shared_samples": min_shared_samples,
        "min_target_correlation": min_target_correlation,
        "significance_alpha": significance_alpha,
        "max_features": max_features,
        "multicollinearity_threshold": multicollinearity_threshold,
    }
    feature_selection = select_predictors(holdout.train_frame, target=target, **selection_params)

    if backtest_initial_train_size is None:
        backtest_initial_train_size = max(min_train_size, len(holdout.train_frame) // 2)
    if backtest_test_window_size is None:
        # Aim for at least 4 rolling windows on the supervised frame.
        target_windows = 4
        candidate = max(min_train_size // 2, len(holdout.test_frame) // 2)
        backtest_test_window_size = max(1, min(candidate, len(supervised_frame) // (target_windows + 1)))
    backtest_splits = rolling_backtest_splits(
        supervised_frame,
        initial_train_size=backtest_initial_train_size,
        test_window_size=backtest_test_window_size,
        step_size=backtest_step_size,
    )

    results: list[ModelResult] = []
    holdout_predictions: list[pd.DataFrame] = []

    for baseline_name in baseline_registry():
        result, predictions = _evaluate_baseline(
            baseline_name,
            train_frame=holdout.train_frame,
            test_frame=holdout.test_frame,
            target=target,
            backtest_splits=backtest_splits,
            warnings=warnings,
        )
        result.combined_score, result.combined_score_used_fallback = compute_combined_score(
            result.metrics,
            result.backtest_metrics,
            primary_metric=PRIMARY_METRIC,
            weight_holdout=combined_score_weight_holdout,
            weight_backtest=combined_score_weight_backtest,
        )
        result.stability_ratio = compute_stability_ratio(
            result.metrics,
            result.backtest_metrics,
            primary_metric=PRIMARY_METRIC,
        )
        results.append(result)
        holdout_predictions.append(predictions)

    baseline_results = [result for result in results if result.is_baseline and result.status == "fitted"]
    with_score = [r for r in baseline_results if r.combined_score is not None]
    if with_score:
        best_baseline = min(with_score, key=lambda r: r.combined_score)  # type: ignore[arg-type]
    else:
        LOGGER.warning("No combined_score for baselines; falling back to holdout %s.", PRIMARY_METRIC)
        best_baseline = min(baseline_results, key=lambda r: r.metrics[PRIMARY_METRIC])
    best_baseline_name = best_baseline.model_name
    best_baseline_score = best_baseline.combined_score

    selected_model_names = list(model_names) if model_names is not None else list(available_model_specs().keys())
    for model_name in selected_model_names:
        result, predictions = _evaluate_ml_model(
            model_name,
            train_frame=holdout.train_frame,
            test_frame=holdout.test_frame,
            target=target,
            feature_selection=feature_selection,
            backtest_splits=backtest_splits,
            selection_params=selection_params,
            warnings=warnings,
            seasonal_feature=seasonal_feature,
        )
        if result.status == "fitted":
            result.combined_score, result.combined_score_used_fallback = compute_combined_score(
                result.metrics,
                result.backtest_metrics,
                primary_metric=PRIMARY_METRIC,
                weight_holdout=combined_score_weight_holdout,
                weight_backtest=combined_score_weight_backtest,
            )
            result.stability_ratio = compute_stability_ratio(
                result.metrics,
                result.backtest_metrics,
                primary_metric=PRIMARY_METRIC,
            )
            if result.combined_score is not None and best_baseline_score is not None:
                result.beats_best_baseline = result.combined_score < best_baseline_score
                fallback_suffix = ":holdout_fallback" if result.combined_score_used_fallback else ""
                result.comparison_note = (
                    f"ml_beats_baseline_combined{fallback_suffix}"
                    if result.beats_best_baseline
                    else f"ml_worse_than_best_baseline:{best_baseline_name}:combined{fallback_suffix}"
                )
                if not result.beats_best_baseline:
                    result.notes = result.notes + (
                        f"Модель уступает базовой '{best_baseline_name}' по комбинированному скору "
                        f"(holdout {combined_score_weight_holdout*100:.0f}% + "
                        f"backtest {combined_score_weight_backtest*100:.0f}%).",
                    )
            else:
                # fallback: no combined score available — use holdout
                holdout_score = result.metrics.get(PRIMARY_METRIC)
                baseline_holdout_score = best_baseline.metrics.get(PRIMARY_METRIC)
                result.beats_best_baseline = (
                    holdout_score is not None
                    and baseline_holdout_score is not None
                    and holdout_score < baseline_holdout_score
                )
                result.comparison_note = (
                    "ml_beats_baseline_combined:holdout_fallback"
                    if result.beats_best_baseline
                    else f"ml_worse_than_best_baseline:{best_baseline_name}:holdout_fallback"
                )
                LOGGER.warning(
                    "No combined_score for model '%s'; comparison fell back to holdout %s.",
                    model_name,
                    PRIMARY_METRIC,
                )
        holdout_predictions.append(predictions)
        results.append(result)

    comparison_df = pd.DataFrame([result.to_record() for result in results]).sort_values(
        ["is_baseline", "status", "combined_score", "model_name"],
        ascending=[True, True, True, True],
        na_position="last",
    )

    backtest_rows = []
    for result in results:
        if result.backtest_metrics:
            backtest_rows.append({"model_name": result.model_name, **result.backtest_metrics})
    backtest_df = pd.DataFrame(backtest_rows)

    holdout_predictions_df = pd.concat(holdout_predictions, ignore_index=True) if holdout_predictions else pd.DataFrame()
    train_dates = pd.to_datetime(holdout.train_frame["SampleDate"], errors="coerce") if not holdout.train_frame.empty else pd.Series(dtype="datetime64[ns]")

    # Collect seasonal feature names from fitted ML models (should be identical for all).
    sfnames_for_run = tuple(_seasonal_names(seasonal_feature))  # type: ignore[arg-type]

    return ModelComparisonRun(
        scope_name=scope_slice.scope_name,
        scope_id=scope_slice.scope_id,
        target=target,
        readiness_assessment=readiness_assessment,
        selected_features=feature_selection.selected_features,
        feature_selection=feature_selection,
        best_baseline_name=best_baseline_name,
        results=results,
        comparison_df=comparison_df.reset_index(drop=True),
        holdout_predictions_df=holdout_predictions_df,
        backtest_df=backtest_df,
        warnings=tuple(warnings),
        scope_selectors=dict(scope_slice.selector),
        aggregation_level="sample_point_level",
        training_period_start=train_dates.min().isoformat() if train_dates.notna().any() else None,
        training_period_end=train_dates.max().isoformat() if train_dates.notna().any() else None,
        train_rows=int(len(holdout.train_frame)),
        holdout_rows=int(len(holdout.test_frame)),
        selection_mode=selection_mode,
        forced_features=tuple(forced_features),
        combined_score_weight_holdout=combined_score_weight_holdout,
        combined_score_weight_backtest=combined_score_weight_backtest,
        seasonal_feature=seasonal_feature,
        seasonal_features=sfnames_for_run,
    )


def train_model_in_scope(
    scope_slice: ScopeSlice,
    *,
    target: str,
    model_name: str | None = None,
    **comparison_kwargs: Any,
) -> tuple[ModelComparisonRun, ModelResult]:
    """Train one selected ML model after running the full comparison workflow."""
    selected_model_names = [model_name] if model_name else None
    run = compare_models_in_scope(
        scope_slice,
        target=target,
        model_names=selected_model_names,
        **comparison_kwargs,
    )

    ml_results = [result for result in run.results if not result.is_baseline and result.status == "fitted"]
    if not ml_results:
        raise ValueError("No fitted ML models are available for training output.")

    if model_name:
        chosen = next(result for result in ml_results if result.model_name == model_name)
    else:
        with_score = [r for r in ml_results if r.combined_score is not None]
        if with_score:
            chosen = min(with_score, key=lambda r: r.combined_score)  # type: ignore[arg-type]
        else:
            chosen = min(ml_results, key=lambda r: r.metrics[PRIMARY_METRIC])

    return run, chosen
