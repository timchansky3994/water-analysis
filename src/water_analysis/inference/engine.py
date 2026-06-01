"""Inference engine for estimating missing target values."""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from water_analysis.analysis.scopes import ScopeSlice
from water_analysis.analysis.seasonal_features import build_seasonal_features
from water_analysis.inference.model_card import ModelCard
from water_analysis.inference.package import LoadedModelPackage
from water_analysis.inference.results import InferenceResult
from water_analysis.inference.selector import check_feature_compatibility, scope_from_card_or_cli
from water_analysis.preprocessing.long_format import build_canonical_long_format, read_source_table
from water_analysis.preprocessing.pivot_builder import build_indicator_pivot

if TYPE_CHECKING:
    from water_analysis.io.source_profiles import SourceProfile

# A manual entry has no real sampling point. Its placeholder FullPointCode zeroes
# the two identifying parts (OKTMO and point type) so it can never be mistaken
# for — or collide with — a real point code: no real point has OKTMO
# "00000000000". Only the trailing index varies, purely to keep batch rows
# distinct after pivoting. The real analytical scope is carried by the model
# card / scope slice, not by this placeholder code (manual estimation builds its
# scope slice directly and never filters on this code).
_MANUAL_SYNTHETIC_OKTMO = "00000000000"
_MANUAL_SYNTHETIC_POINT_TYPE = "00000"

PREDICTION_COLUMNS: tuple[str, ...] = (
    "SampleDate",
    "FullPointCode",
    "OKTMO",
    "PointType_Code",
    "PointNumber",
    "target",
    "predicted_value",
    "lower_bound",
    "upper_bound",
    "prediction_status",
    "model_package",
    "model_name",
    "scope_name",
    "feature_names_used",
    "features_total",
    "features_observed",
    "feature_coverage",
    "missing_features",
    "warnings",
)


def _metadata_columns(pivot_df: pd.DataFrame) -> list[str]:
    """Return metadata columns carried into prediction outputs."""
    return [
        column
        for column in ["SampleDate", "FullPointCode", "OKTMO", "PointType_Code", "PointNumber"]
        if column in pivot_df.columns
    ]


def _empty_result(package: LoadedModelPackage, diagnostics: list[dict[str, Any]], warnings: list[str]) -> InferenceResult:
    """Build an empty inference result with diagnostics."""
    summary = {
        "target": package.model_card.target,
        "model_package": str(package.package_dir),
        "model_name": package.model_card.model_name,
        "scope_name": package.model_card.scope_name,
        "rows_for_estimation": 0,
        "predicted_rows": 0,
        "skipped_rows": 0,
        "warnings": warnings,
    }
    return InferenceResult(
        predictions=pd.DataFrame(columns=PREDICTION_COLUMNS),
        estimated_values_long=pd.DataFrame(),
        diagnostics=pd.DataFrame(diagnostics),
        summary=summary,
    )


def _prediction_interval(predictions: np.ndarray, residual_quantiles: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Build rough residual-based prediction intervals."""
    if "q05" not in residual_quantiles or "q95" not in residual_quantiles:
        nan_bounds = np.full_like(predictions, np.nan, dtype=float)
        return nan_bounds, nan_bounds
    return predictions + residual_quantiles["q05"], predictions + residual_quantiles["q95"]


def _estimated_long(predicted_rows: pd.DataFrame) -> pd.DataFrame:
    """Convert successful predictions to long-format estimated values."""
    if predicted_rows.empty:
        return pd.DataFrame()
    long_df = predicted_rows[
        [
            "SampleDate",
            "FullPointCode",
            "OKTMO",
            "PointType_Code",
            "PointNumber",
            "target",
            "predicted_value",
            "lower_bound",
            "upper_bound",
            "model_package",
            "model_name",
        ]
    ].copy()
    long_df = long_df.rename(columns={"target": "Indicator", "predicted_value": "Value_Approx"})
    long_df["RawValueText"] = ""
    long_df["ResultValueText"] = ""
    long_df["ValueSource"] = "estimated"
    long_df["IsEstimated"] = True
    long_df["ModelPackage"] = long_df["model_package"]
    long_df["OriginalValuePresent"] = False
    long_df["PredictionLowerBound"] = long_df["lower_bound"]
    long_df["PredictionUpperBound"] = long_df["upper_bound"]
    return long_df.drop(columns=["model_package", "lower_bound", "upper_bound"])


def estimate_missing_values(
    input_path: str | Path,
    package: LoadedModelPackage,
    *,
    target: str | None = None,
    scope_name: str | None = None,
    oktmo: str | None = None,
    point_type: str | None = None,
    point_code: str | None = None,
    min_observed_features: int = 2,
    min_feature_coverage: float = 0.5,
    allow_scope_fallback: bool = False,
    predict_all: bool = False,
    allow_missing_feature_columns: bool = False,
    source_profile: "SourceProfile | None" = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> InferenceResult:
    """Estimate missing target values in a new raw CSV/XLSX file."""

    def _progress(stage: str, fraction: float) -> None:
        if progress_callback is not None:
            progress_callback(stage, fraction)

    _progress("Чтение нового файла", 0.10)
    raw_df = read_source_table(input_path)
    long_df = build_canonical_long_format(raw_df, source_profile=source_profile)
    return _estimate_from_long_format(
        long_df,
        package,
        target=target,
        scope_name=scope_name,
        oktmo=oktmo,
        point_type=point_type,
        point_code=point_code,
        min_observed_features=min_observed_features,
        min_feature_coverage=min_feature_coverage,
        allow_scope_fallback=allow_scope_fallback,
        predict_all=predict_all,
        allow_missing_feature_columns=allow_missing_feature_columns,
        progress_callback=progress_callback,
    )


def _estimate_from_long_format(
    long_df: pd.DataFrame,
    package: LoadedModelPackage,
    *,
    target: str | None = None,
    scope_name: str | None = None,
    oktmo: str | None = None,
    point_type: str | None = None,
    point_code: str | None = None,
    min_observed_features: int = 2,
    min_feature_coverage: float = 0.5,
    allow_scope_fallback: bool = False,
    predict_all: bool = False,
    allow_missing_feature_columns: bool = False,
    scope_slice: ScopeSlice | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> InferenceResult:
    """Run estimation against an already-built canonical long-format frame.

    Shared core for both inference "faces": file-based estimation
    (``estimate_missing_values``) and manually entered values
    (``estimate_manual_values``). Keeping the model-application logic here
    guarantees both paths apply the model, gate on feature coverage, build
    prediction intervals and emit diagnostics identically.

    When ``scope_slice`` is given it is used directly and scope resolution is
    skipped — the manual path supplies a slice it built itself, so its synthetic
    placeholder rows never have to masquerade as real points to pass a scope
    filter.
    """

    def _progress(stage: str, fraction: float) -> None:
        if progress_callback is not None:
            progress_callback(stage, fraction)

    card = package.model_card
    requested_target = target or card.target
    diagnostics: list[dict[str, Any]] = []
    warnings: list[str] = []

    if requested_target != card.target:
        diagnostics.append({"level": "run", "reason": "target_mismatch", "detail": f"{requested_target} != {card.target}"})
        return _empty_result(package, diagnostics, warnings)
    if not card.ml_beats_baseline:
        warnings.append("Model did not outperform the best baseline during validation; use estimates with caution.")

    if scope_slice is None:
        _progress("Применение модели", 0.50)
        scope_slice, scope_reasons = scope_from_card_or_cli(
            long_df,
            card,
            scope_name=scope_name,
            oktmo=oktmo,
            point_type=point_type,
            point_code=point_code,
            allow_scope_fallback=allow_scope_fallback,
        )
        if scope_slice is None:
            diagnostics.extend({"level": "run", "reason": reason, "detail": ""} for reason in scope_reasons)
            return _empty_result(package, diagnostics, warnings)
    else:
        _progress("Применение модели", 0.50)

    pivot_df = build_indicator_pivot(scope_slice.dataframe, aggregation_level="sample_point_level")
    if pivot_df.empty:
        diagnostics.append({"level": "run", "reason": "empty_scope_after_pivot", "detail": scope_slice.scope_id})
        return _empty_result(package, diagnostics, warnings)

    # Seasonal features are derived from SampleDate, not read from input — skip them in the
    # compatibility check and build them from the date columns already present in the pivot.
    seasonal_names = frozenset(card.seasonal_feature_names)
    feature_compatibility = check_feature_compatibility(card, pivot_df, skip_features=seasonal_names)
    if feature_compatibility.missing_features:
        diagnostics.extend(
            {"level": "feature", "reason": "missing_feature_column", "detail": feature}
            for feature in feature_compatibility.missing_features
        )
    if not feature_compatibility.compatible:
        diagnostics.extend({"level": "run", "reason": reason, "detail": ""} for reason in feature_compatibility.reasons)
        return _empty_result(package, diagnostics, warnings)
    if feature_compatibility.missing_features and not allow_missing_feature_columns:
        diagnostics.append({"level": "run", "reason": "missing_feature_columns_not_allowed", "detail": ""})
        return _empty_result(package, diagnostics, warnings)

    for feature in feature_compatibility.missing_features:
        pivot_df[feature] = np.nan

    # Build seasonal features from the date columns already present in the pivot.
    if seasonal_names:
        pivot_df, _ = build_seasonal_features(pivot_df, card.seasonal_feature)

    feature_names = list(card.feature_names)
    # Seasonal features are derived from the date and therefore always present; they must not
    # count toward per-row coverage, or a row with no real measurements could pass the gate on
    # the always-present seasonal columns alone. Gate and report coverage on indicator features
    # only; the estimator still receives the full feature set (indicators + seasonal).
    coverage_features = [feature for feature in feature_names if feature not in seasonal_names]
    target_present = requested_target in pivot_df.columns
    if predict_all:
        candidate_mask = pd.Series(True, index=pivot_df.index)
    elif target_present:
        candidate_mask = pivot_df[requested_target].isna()
    else:
        candidate_mask = pd.Series(True, index=pivot_df.index)
    candidates = pivot_df[candidate_mask].copy()

    rows: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    for row_index, row in candidates.iterrows():
        observed_features = [feature for feature in coverage_features if pd.notna(row[feature])]
        missing_features = [feature for feature in coverage_features if pd.isna(row[feature])]
        features_total = len(coverage_features)
        feature_coverage = len(observed_features) / features_total if features_total else 1.0
        row_warnings = list(warnings) + [f"missing_feature:{feature}" for feature in missing_features]

        status = "pending"
        if features_total == 0:
            # Seasonal-only model: no indicator predictors to gate on; seasonal features
            # are always available, so every candidate row can be estimated.
            valid_indices.append(row_index)
        elif len(observed_features) < min_observed_features:
            status = "skipped_insufficient_features"
        elif feature_coverage < min_feature_coverage:
            status = "skipped_low_feature_coverage"
        else:
            valid_indices.append(row_index)

        output_row = {column: row.get(column) for column in _metadata_columns(pivot_df)}
        output_row.update(
            {
                "target": requested_target,
                "predicted_value": np.nan,
                "lower_bound": np.nan,
                "upper_bound": np.nan,
                "prediction_status": status,
                "model_package": str(package.package_dir),
                "model_name": card.model_name,
                "scope_name": scope_slice.scope_name,
                "feature_names_used": "|".join(observed_features),
                "features_total": features_total,
                "features_observed": len(observed_features),
                "feature_coverage": feature_coverage,
                "missing_features": "|".join(missing_features),
                "warnings": " | ".join(row_warnings),
            }
        )
        rows.append(output_row)

    predictions_df = pd.DataFrame(rows)
    if valid_indices:
        X = pivot_df.loc[valid_indices, feature_names]
        y_pred = np.asarray(package.estimator.predict(X), dtype=float)
        lower, upper = _prediction_interval(y_pred, card.residual_quantiles)
        valid_positions = predictions_df[predictions_df["prediction_status"] == "pending"].index
        predictions_df.loc[valid_positions, "predicted_value"] = y_pred
        predictions_df.loc[valid_positions, "lower_bound"] = lower
        predictions_df.loc[valid_positions, "upper_bound"] = upper
        predictions_df.loc[valid_positions, "prediction_status"] = "estimated"

    for row in predictions_df.itertuples(index=False):
        if row.prediction_status != "estimated":
            diagnostics.append(
                {
                    "level": "row",
                    "reason": row.prediction_status,
                    "detail": getattr(row, "FullPointCode", ""),
                }
            )

    _progress("Сохранение оценок", 0.90)
    predictions_df = predictions_df.reindex(columns=PREDICTION_COLUMNS)
    estimated_long = _estimated_long(predictions_df[predictions_df["prediction_status"] == "estimated"])
    summary = {
        "target": requested_target,
        "model_package": str(package.package_dir),
        "model_name": card.model_name,
        "scope_name": scope_slice.scope_name,
        "scope_id": scope_slice.scope_id,
        "rows_for_estimation": int(len(candidates)),
        "predicted_rows": int((predictions_df["prediction_status"] == "estimated").sum()) if not predictions_df.empty else 0,
        "skipped_rows": int((predictions_df["prediction_status"] != "estimated").sum()) if not predictions_df.empty else 0,
        "warnings": warnings,
    }
    return InferenceResult(
        predictions=predictions_df,
        estimated_values_long=estimated_long,
        diagnostics=pd.DataFrame(diagnostics),
        summary=summary,
    )


# ── Manual value entry ──────────────────────────────────────────────────────
# The second "face" of estimation: instead of a file, the specialist types the
# indicator values directly. These helpers synthesize a tiny canonical frame so
# the manual path flows through the exact same estimation core above.


def manual_input_feature_names(card: ModelCard) -> list[str]:
    """Return the indicator features a user must enter for manual estimation.

    Seasonal features are derived from the sample date rather than typed, so
    they are excluded — the caller collects a date separately.
    """
    seasonal = frozenset(card.seasonal_feature_names)
    return [feature for feature in card.feature_names if feature not in seasonal]


def _manual_point_code(suffix: int = 0) -> str:
    """Return a clearly-synthetic placeholder FullPointCode for a manual row.

    The identifying parts (OKTMO and point type) are zeroed so the code cannot
    coincide with a real point — no real point has OKTMO "00000000000". Only the
    trailing index varies, solely to keep batch rows distinct after pivoting. The
    real analytical scope is recorded separately on the scope slice / model card;
    manual estimation builds its slice directly and never filters on this code,
    so it does not need to resemble (and must not be confused with) a real point.
    """
    return f"{_MANUAL_SYNTHETIC_OKTMO}.{_MANUAL_SYNTHETIC_POINT_TYPE}.{suffix + 1:04d}"


def _manual_value_text(value: Any) -> str:
    """Normalize a single manually entered cell to source-style text."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def _manual_date_text(value: Any) -> str:
    """Normalize a manually entered date to a string the normalizer can parse."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    return str(value).strip()


def build_manual_long_format(card: ModelCard, samples: pd.DataFrame) -> pd.DataFrame:
    """Synthesize a canonical long-format frame from manually entered values.

    Each row of ``samples`` is one sample: a ``SampleDate`` plus a value for any
    subset of the model's indicator features (see ``manual_input_feature_names``).
    The frame is built through the standard normalization path so censoring
    parsing and temporal/seasonal feature derivation match file-based inference
    exactly. Each row gets a clearly-synthetic placeholder point code
    (see ``_manual_point_code``) that cannot be confused with a real point.
    """
    feature_names = manual_input_feature_names(card)
    raw_rows: list[dict[str, str]] = []
    for position, (_, sample) in enumerate(samples.reset_index(drop=True).iterrows()):
        date_text = _manual_date_text(sample.get("SampleDate"))
        code = _manual_point_code(position)
        if feature_names:
            for feature in feature_names:
                raw_rows.append(
                    {
                        "Date": date_text,
                        "Code": code,
                        "Indicator": feature,
                        "Value": _manual_value_text(sample.get(feature)),
                    }
                )
        else:
            # Seasonal-only model: emit an anchor row so the date survives the pivot.
            raw_rows.append({"Date": date_text, "Code": code, "Indicator": card.target, "Value": ""})

    # Columns use the default-profile English aliases (Date/Code/Indicator/Value),
    # so the canonical builder maps them without a custom source profile.
    raw_df = pd.DataFrame(raw_rows, columns=["Date", "Code", "Indicator", "Value"])
    return build_canonical_long_format(raw_df)


def estimate_manual_values(
    package: LoadedModelPackage,
    samples: "pd.DataFrame | list[dict[str, Any]]",
    *,
    min_observed_features: int = 1,
    min_feature_coverage: float = 0.0,
    predict_all: bool = True,
    allow_missing_feature_columns: bool = True,
    progress_callback: Callable[[str, float], None] | None = None,
) -> InferenceResult:
    """Estimate target values from manually entered indicator values.

    The manual counterpart of :func:`estimate_missing_values`: instead of
    reading a file, it synthesizes a one-sample-per-row canonical frame and runs
    the same estimation core, producing an identical :class:`InferenceResult`
    (so the same predictions/diagnostics/export pipeline applies).

    The scope slice is built directly from the model card rather than resolved by
    filtering, so the synthetic placeholder rows never have to carry a real-
    looking point code to match a scope. The slice records the model's true scope
    name/id for the summary; its rows use clearly-synthetic placeholder codes.

    Gating defaults are looser than the file path because the specialist
    deliberately enters whatever values are on hand; per-row feature coverage is
    still reported so unreliable estimates remain visible rather than hidden.
    """

    def _progress(stage: str, fraction: float) -> None:
        if progress_callback is not None:
            progress_callback(stage, fraction)

    _progress("Подготовка введённых значений", 0.10)
    card = package.model_card
    samples_df = samples if isinstance(samples, pd.DataFrame) else pd.DataFrame(list(samples))
    long_df = build_manual_long_format(card, samples_df)
    manual_slice = ScopeSlice(
        scope_name=card.scope_name,
        scope_id=card.scope_id,
        scope_label="Ручной ввод значений",
        selector={},
        dataframe=long_df,
    )
    return _estimate_from_long_format(
        long_df,
        package,
        min_observed_features=min_observed_features,
        min_feature_coverage=min_feature_coverage,
        predict_all=predict_all,
        allow_missing_feature_columns=allow_missing_feature_columns,
        scope_slice=manual_slice,
        progress_callback=progress_callback,
    )
