"""Tests for seasonal feature encoding (analysis/seasonal_features.py)."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from water_analysis.analysis.seasonal_features import (
    build_seasonal_features,
    seasonal_feature_names,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DATE_COL = "Дата проведения исследования"
_POINT_COL = "Код точки"
_INDICATOR_COL = "Гигиенический показатель"
_VALUE_COL = "Результат исследования"
_CODE = "00000000001.10110.0010"

_SEASON_MONTHS = {
    "winter": ["05.01.{y}", "06.01.{y}", "07.01.{y}", "08.01.{y}", "09.01.{y}"],
    "spring": ["05.04.{y}", "06.04.{y}", "07.04.{y}", "08.04.{y}", "09.04.{y}"],
    "summer": ["05.07.{y}", "06.07.{y}", "07.07.{y}", "08.07.{y}", "09.07.{y}"],
    "autumn": ["05.10.{y}", "06.10.{y}", "07.10.{y}", "08.10.{y}", "09.10.{y}"],
}


def _make_raw_df(*, n_per_season: int = 5, year: int = 2020) -> pd.DataFrame:
    """Build a minimal raw DataFrame compatible with build_canonical_long_format."""
    rows: list[dict] = []
    dates_by_season = {
        season: [tmpl.format(y=year) for tmpl in templates[:n_per_season]]
        for season, templates in _SEASON_MONTHS.items()
    }
    for season, dates in dates_by_season.items():
        for i, date_str in enumerate(dates):
            val_turb = float(10 + i)
            val_hard = float(val_turb * 2)
            for ind, val in [
                ("Мутность (по формазину)", val_turb),
                ("Жесткость общая", val_hard),
            ]:
                rows.append(
                    {
                        _DATE_COL: date_str,
                        "Тип точки": "Точка на распределительной сети",
                        _POINT_COL: _CODE,
                        _INDICATOR_COL: ind,
                        "Норматив": "",
                        "Строчное значение": "",
                        _VALUE_COL: str(int(val)),
                        "Нижний предел обнаружения": "",
                        "Верхний предел обнаружения": "",
                        "Ошибка метода определения": "",
                        "Нормативная документация": "",
                        "Цели исследований": "",
                        "Соотв. ПДК": "",
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. mode="none" — passthrough
# ---------------------------------------------------------------------------


def test_build_seasonal_features_none_returns_copy_unchanged() -> None:
    frame = pd.DataFrame({"season": ["winter", "summer"], "val": [1.0, 2.0]})
    out, names = build_seasonal_features(frame, "none")

    assert names == []
    pd.testing.assert_frame_equal(out, frame)
    assert out is not frame  # must be a copy


# ---------------------------------------------------------------------------
# 2. mode="season" — one-hot
# ---------------------------------------------------------------------------


def test_build_seasonal_features_season_onehot_all_four_columns() -> None:
    frame = pd.DataFrame(
        {"season": ["winter", "spring", "summer", "autumn", "winter"], "val": range(5)}
    )
    out, names = build_seasonal_features(frame, "season")

    assert set(names) == {"season_winter", "season_spring", "season_summer", "season_autumn"}
    assert len(names) == 4
    assert list(out["season_winter"]) == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert list(out["season_summer"]) == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert "season" in out.columns  # original column preserved


def test_build_seasonal_features_season_without_season_column_fills_zeros() -> None:
    """All four zero columns generated when 'season' column is absent."""
    frame = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
    out, names = build_seasonal_features(frame, "season")

    assert set(names) == {"season_winter", "season_spring", "season_summer", "season_autumn"}
    for col in names:
        assert (out[col] == 0.0).all(), f"{col} should be all zeros"


def test_build_seasonal_features_season_deterministic_names_across_partial_data() -> None:
    """Column names and order identical even if only one season appears in a split."""
    frame_full = pd.DataFrame({"season": ["winter", "spring", "summer", "autumn"]})
    frame_partial = pd.DataFrame({"season": ["winter", "winter", "winter"]})

    _, names_full = build_seasonal_features(frame_full, "season")
    _, names_partial = build_seasonal_features(frame_partial, "season")

    assert names_full == names_partial


# ---------------------------------------------------------------------------
# 3. mode="month" — cyclic encoding
# ---------------------------------------------------------------------------


def test_build_seasonal_features_month_cyclic_sin_cos_values() -> None:
    frame = pd.DataFrame({"month": [1, 6, 12], "val": [10.0, 20.0, 30.0]})
    out, names = build_seasonal_features(frame, "month")

    assert names == ["month_sin", "month_cos"]

    expected_sin_jan = math.sin(2 * math.pi * 1 / 12)
    expected_cos_jan = math.cos(2 * math.pi * 1 / 12)
    assert abs(out["month_sin"].iloc[0] - expected_sin_jan) < 1e-9
    assert abs(out["month_cos"].iloc[0] - expected_cos_jan) < 1e-9

    expected_sin_dec = math.sin(2 * math.pi * 12 / 12)
    assert abs(out["month_sin"].iloc[2] - expected_sin_dec) < 1e-9


def test_build_seasonal_features_month_without_month_column_fills_nan() -> None:
    """NaN placeholders generated when 'month' column is absent."""
    frame = pd.DataFrame({"val": [1.0, 2.0]})
    out, names = build_seasonal_features(frame, "month")

    assert names == ["month_sin", "month_cos"]
    assert out["month_sin"].isna().all()
    assert out["month_cos"].isna().all()


# ---------------------------------------------------------------------------
# 4. seasonal_feature_names() helper
# ---------------------------------------------------------------------------


def test_seasonal_feature_names_none() -> None:
    assert seasonal_feature_names("none") == []


def test_seasonal_feature_names_season() -> None:
    names = seasonal_feature_names("season")
    assert len(names) == 4
    assert set(names) == {"season_winter", "season_spring", "season_summer", "season_autumn"}


def test_seasonal_feature_names_month() -> None:
    assert seasonal_feature_names("month") == ["month_sin", "month_cos"]


# ---------------------------------------------------------------------------
# 5. build_seasonal_features does not mutate the original frame
# ---------------------------------------------------------------------------


def test_build_seasonal_features_does_not_mutate_original() -> None:
    frame = pd.DataFrame({"season": ["winter", "summer"], "val": [1.0, 2.0]})
    original_cols = list(frame.columns)
    build_seasonal_features(frame, "season")
    assert list(frame.columns) == original_cols


# ---------------------------------------------------------------------------
# 6. Seasonal features are added AFTER feature selection
# ---------------------------------------------------------------------------


def test_seasonal_features_not_in_selected_features() -> None:
    """ModelComparisonRun.selected_features must contain only indicator columns.

    Seasonal feature columns (season_winter etc.) must not appear in selected_features
    because they are added after correlation-based selection runs.
    """
    from water_analysis.analysis.scopes import build_scope_slices
    from water_analysis.modeling.trainer import compare_models_in_scope
    from water_analysis.preprocessing.long_format import build_canonical_long_format

    raw = _make_raw_df(n_per_season=8)
    long_df = build_canonical_long_format(raw)
    scope = build_scope_slices(long_df, scope_name="global")[0]

    run = compare_models_in_scope(
        scope,
        target="Мутность (по формазину)",
        model_names=["ridge"],
        seasonal_feature="season",
        min_train_size=5,
        min_target_observations=5,
        min_shared_samples=5,
        max_missing_ratio=0.95,
        heavy_censoring_ratio=0.95,
        min_eligible_predictors=1,
        min_target_correlation=0.05,
        significance_alpha=0.9,
        max_features=5,
    )

    seasonal_cols = set(seasonal_feature_names("season"))
    assert not seasonal_cols.intersection(set(run.selected_features)), (
        f"seasonal columns found in selected_features: "
        f"{seasonal_cols.intersection(set(run.selected_features))}"
    )
    # seasonal_features on the run should be populated
    assert set(run.seasonal_features) == seasonal_cols


# ---------------------------------------------------------------------------
# 7. Model card stores seasonal_feature and seasonal_feature_names correctly
# ---------------------------------------------------------------------------


def test_seasonal_feature_stored_in_model_card_payload() -> None:
    """build_model_card_payload captures seasonal_feature mode and column names."""
    from water_analysis.analysis.feature_selection import FeatureSelectionResult
    from water_analysis.inference.package import build_model_card_payload
    from water_analysis.modeling.trainer import ModelComparisonRun, ModelResult
    from water_analysis.profiling.readiness import ReadinessAssessment

    readiness = ReadinessAssessment(
        scope_name="global",
        scope_id="global_test",
        scope_label="global_test",
        target="Мутность (по формазину)",
        status="suitable",
        sample_point_rows=20,
        target_observation_count=20,
        target_missing_ratio=0.0,
        target_censored_ratio=0.0,
        eligible_predictor_count=1,
        max_shared_samples=20,
        issues=(),
    )
    sel_result = FeatureSelectionResult(
        target="Мутность (по формазину)",
        selected_features=("Жесткость общая",),
        candidate_table=pd.DataFrame(),
        dropped_multicollinear=(),
        selection_mode="auto",
        forced_features=(),
    )

    X_fit = pd.DataFrame(
        {
            "Жесткость общая": [1.0, 2.0, 3.0],
            "month_sin": [0.1, 0.2, 0.3],
            "month_cos": [0.9, 0.8, 0.7],
        }
    )
    estimator = Pipeline([("model", Ridge())])
    estimator.fit(X_fit, [10.0, 20.0, 30.0])

    candidate = ModelResult(
        model_name="ridge",
        model_family="linear",
        is_baseline=False,
        status="fitted",
        metrics={"rmse": 1.0, "mae": 0.8, "r2": 0.9},
        backtest_metrics={"rmse": 1.2, "mae": 0.9, "r2": 0.85},
        feature_names=("Жесткость общая", "month_sin", "month_cos"),
        beats_best_baseline=True,
        comparison_note="ml_beats_baseline_combined",
        notes=(),
        interpretability_df=pd.DataFrame(columns=["feature", "importance", "feature_kind"]),
        estimator=estimator,
        combined_score=1.0,
        stability_ratio=1.2,
        seasonal_features=("month_sin", "month_cos"),
    )

    run = ModelComparisonRun(
        scope_name="global",
        scope_id="global_test",
        target="Мутность (по формазину)",
        readiness_assessment=readiness,
        selected_features=("Жесткость общая",),
        feature_selection=sel_result,
        best_baseline_name="median_baseline",
        results=[candidate],
        comparison_df=pd.DataFrame(),
        holdout_predictions_df=pd.DataFrame(),
        backtest_df=pd.DataFrame(),
        warnings=(),
        seasonal_feature="month",
        seasonal_features=("month_sin", "month_cos"),
    )

    payload = build_model_card_payload(run, candidate)

    assert payload["seasonal_feature"] == "month"
    assert payload["seasonal_feature_names"] == ["month_sin", "month_cos"]
    # required_features excludes seasonal
    assert "month_sin" not in payload["required_features"]
    assert "month_cos" not in payload["required_features"]
    assert "Жесткость общая" in payload["required_features"]
    # feature_names includes all
    assert "month_sin" in payload["feature_names"]
    assert "Жесткость общая" in payload["feature_names"]


# ---------------------------------------------------------------------------
# 8. estimate_missing: seasonal features derived from date, not required in input
# ---------------------------------------------------------------------------


def test_estimate_missing_with_seasonal_feature_does_not_require_season_column() -> None:
    """Inference with seasonal_feature='month' must work without precomputed seasonal columns.

    The raw input file has no month_sin/month_cos columns.  The inference engine derives
    them from SampleDate via the standard pivot pipeline.
    """
    import joblib

    from water_analysis.inference.engine import estimate_missing_values
    from water_analysis.inference.model_card import ESTIMATED_VALUE_WARNING, SCHEMA_VERSION
    from water_analysis.inference.package import load_model_package

    raw = _make_raw_df(n_per_season=8)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        csv_path = tmp_path / "test_data.csv"
        raw.to_csv(csv_path, index=False, encoding="utf-8-sig", sep=";")

        # Train a simple estimator that expects indicator + cyclic month features
        X_train = pd.DataFrame(
            {
                "Жесткость общая": [10.0, 20.0, 30.0, 15.0, 25.0, 12.0],
                "month_sin": [0.5, 0.87, 0.5, -0.5, -0.87, 0.0],
                "month_cos": [0.87, 0.5, -0.87, 0.87, -0.5, 1.0],
            }
        )
        y_train = pd.Series([5.0, 10.0, 15.0, 7.0, 12.0, 6.0])
        estimator = Pipeline([("model", Ridge())])
        estimator.fit(X_train, y_train)

        feature_names = ["Жесткость общая", "month_sin", "month_cos"]
        card_payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": "2026-01-01T00:00:00+00:00",
            "target": "Мутность (по формазину)",
            "scope_name": "global",
            "scope_id": "global",
            "scope_selectors": {},
            "aggregation_level": "sample_point_level",
            "model_name": "ridge",
            "feature_names": feature_names,
            "required_features": ["Жесткость общая"],
            "readiness_status": "suitable",
            "readiness_reasons": "",
            "holdout_metrics": {"rmse": 2.0},
            "baseline_metrics": {"rmse": 3.0},
            "best_baseline_name": "median_baseline",
            "comparison_note": "ml_beats_baseline_combined",
            "ml_beats_baseline": True,
            "training_period_start": "2020-01-01",
            "training_period_end": "2020-12-31",
            "train_rows": 6,
            "holdout_rows": 2,
            "preprocessing_assumptions": {
                "pivot_aggregation_level": "sample_point_level",
            },
            "warning": ESTIMATED_VALUE_WARNING,
            "seasonal_feature": "month",
            "seasonal_feature_names": ["month_sin", "month_cos"],
        }

        pkg_dir = tmp_path / "model_package"
        pkg_dir.mkdir()
        (pkg_dir / "model_card.json").write_text(
            json.dumps(card_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        joblib.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "model_name": "ridge",
                "target": "Мутность (по формазину)",
                "feature_names": feature_names,
                "estimator": estimator,
            },
            pkg_dir / "model.joblib",
        )

        package = load_model_package(pkg_dir)

        result = estimate_missing_values(
            csv_path,
            package,
            predict_all=True,
            min_observed_features=1,
            min_feature_coverage=0.1,
        )

    # Must not crash.  Verify that no diagnostic blames missing seasonal columns.
    diag_reasons = (
        set(result.diagnostics["reason"].tolist())
        if not result.diagnostics.empty
        else set()
    )
    diag_details = (
        " ".join(str(d) for d in result.diagnostics["detail"].tolist())
        if not result.diagnostics.empty
        else ""
    )
    assert "month_sin" not in diag_details, "month_sin should not appear as missing feature"
    assert "month_cos" not in diag_details, "month_cos should not appear as missing feature"

    # At least some rows should be estimated
    assert result.summary.get("rows_for_estimation", 0) > 0, (
        "expected at least one candidate row for estimation"
    )


# ---------------------------------------------------------------------------
# 9. Seasonal features must not inflate per-row feature coverage
# ---------------------------------------------------------------------------


def test_seasonal_features_do_not_count_toward_row_coverage() -> None:
    """A candidate row with no real indicator must be skipped, not estimated.

    Seasonal columns are derived from the date and therefore always present.  They must not
    count toward per-row coverage; otherwise a row with zero measured predictors would pass
    the coverage gate on the always-present seasonal columns and get a misleading estimate.
    """
    import joblib

    from water_analysis.inference.engine import estimate_missing_values
    from water_analysis.inference.model_card import ESTIMATED_VALUE_WARNING, SCHEMA_VERSION
    from water_analysis.inference.package import load_model_package

    # Two winter sample rows for the same point: one has the indicator predictor
    # ("Жесткость общая"), the other has only an unrelated indicator ("pH").
    raw = pd.DataFrame(
        [
            {
                _DATE_COL: "05.01.2020",
                "Тип точки": "Точка на распределительной сети",
                _POINT_COL: _CODE,
                _INDICATOR_COL: "Жесткость общая",
                "Норматив": "",
                "Строчное значение": "",
                _VALUE_COL: "20",
                "Нижний предел обнаружения": "",
                "Верхний предел обнаружения": "",
                "Ошибка метода определения": "",
                "Нормативная документация": "",
                "Цели исследований": "",
                "Соотв. ПДК": "",
            },
            {
                _DATE_COL: "06.01.2020",
                "Тип точки": "Точка на распределительной сети",
                _POINT_COL: _CODE,
                _INDICATOR_COL: "pH",
                "Норматив": "",
                "Строчное значение": "",
                _VALUE_COL: "7",
                "Нижний предел обнаружения": "",
                "Верхний предел обнаружения": "",
                "Ошибка метода определения": "",
                "Нормативная документация": "",
                "Цели исследований": "",
                "Соотв. ПДК": "",
            },
        ]
    )

    season_cols = seasonal_feature_names("season")
    feature_names = ["Жесткость общая", *season_cols]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_path = tmp_path / "test_data.csv"
        raw.to_csv(csv_path, index=False, encoding="utf-8-sig", sep=";")

        X_train = pd.DataFrame(
            {
                "Жесткость общая": [10.0, 20.0, 30.0, 15.0],
                "season_winter": [1.0, 1.0, 0.0, 0.0],
                "season_spring": [0.0, 0.0, 1.0, 0.0],
                "season_summer": [0.0, 0.0, 0.0, 1.0],
                "season_autumn": [0.0, 0.0, 0.0, 0.0],
            }
        )
        estimator = Pipeline([("model", Ridge())])
        estimator.fit(X_train, [5.0, 10.0, 15.0, 7.0])

        card_payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": "2026-01-01T00:00:00+00:00",
            "target": "Мутность (по формазину)",
            "scope_name": "global",
            "scope_id": "global",
            "scope_selectors": {},
            "aggregation_level": "sample_point_level",
            "model_name": "ridge",
            "feature_names": feature_names,
            "required_features": ["Жесткость общая"],
            "readiness_status": "suitable",
            "readiness_reasons": "",
            "holdout_metrics": {"rmse": 2.0},
            "baseline_metrics": {"rmse": 3.0},
            "best_baseline_name": "median_baseline",
            "comparison_note": "ml_beats_baseline_combined",
            "ml_beats_baseline": True,
            "training_period_start": "2020-01-01",
            "training_period_end": "2020-12-31",
            "train_rows": 4,
            "holdout_rows": 2,
            "preprocessing_assumptions": {"pivot_aggregation_level": "sample_point_level"},
            "warning": ESTIMATED_VALUE_WARNING,
            "seasonal_feature": "season",
            "seasonal_feature_names": season_cols,
        }

        pkg_dir = tmp_path / "model_package"
        pkg_dir.mkdir()
        (pkg_dir / "model_card.json").write_text(
            json.dumps(card_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        joblib.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "model_name": "ridge",
                "target": "Мутность (по формазину)",
                "feature_names": feature_names,
                "estimator": estimator,
            },
            pkg_dir / "model.joblib",
        )

        package = load_model_package(pkg_dir)
        result = estimate_missing_values(
            csv_path,
            package,
            predict_all=True,
            min_observed_features=1,
            min_feature_coverage=0.5,
        )

    preds = result.predictions
    # Exactly one row has the indicator measured → estimated; the pH-only row → skipped.
    estimated = preds[preds["prediction_status"] == "estimated"]
    skipped = preds[preds["prediction_status"] == "skipped_insufficient_features"]
    assert len(estimated) == 1, f"expected one estimated row, got {len(estimated)}"
    assert len(skipped) == 1, f"expected one skipped row, got {len(skipped)}"
    # Coverage reported for the estimated row reflects indicator features only (1 of 1).
    assert estimated.iloc[0]["features_total"] == 1
    assert estimated.iloc[0]["features_observed"] == 1
