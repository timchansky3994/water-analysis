"""Save and load deployable model packages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd

from water_analysis.inference.model_card import ESTIMATED_VALUE_WARNING, SCHEMA_VERSION, ModelCard, dump_model_card, load_model_card

if TYPE_CHECKING:
    from water_analysis.modeling.trainer import ModelComparisonRun, ModelResult


@dataclass(frozen=True)
class LoadedModelPackage:
    """Loaded estimator and validated metadata."""

    package_dir: Path
    model_card: ModelCard
    estimator: Any


def _json_float(value: Any) -> float | None:
    """Convert numeric values to JSON-friendly floats."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _residual_rows(run: "ModelComparisonRun", model_name: str) -> pd.DataFrame:
    """Return validation residuals for a selected model."""
    if run.holdout_predictions_df.empty:
        return pd.DataFrame(columns=["SampleDate", "FullPointCode", "actual", "predicted", "residual", "model_name"])
    rows = run.holdout_predictions_df[run.holdout_predictions_df["model_name"] == model_name].copy()
    if rows.empty:
        return pd.DataFrame(columns=["SampleDate", "FullPointCode", "actual", "predicted", "residual", "model_name"])
    rows["residual"] = rows["actual"].astype(float) - rows["predicted"].astype(float)
    return rows


def _residual_summary(residuals: pd.Series) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Build residual summary and quantiles for rough prediction intervals."""
    clean = residuals.dropna().astype(float)
    if clean.empty:
        return {}, {}
    summary = {
        "count": int(clean.size),
        "mean": _json_float(clean.mean()),
        "std": _json_float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "mae": _json_float(clean.abs().mean()),
    }
    quantiles = {
        "q05": _json_float(clean.quantile(0.05)),
        "q10": _json_float(clean.quantile(0.10)),
        "q50": _json_float(clean.quantile(0.50)),
        "q90": _json_float(clean.quantile(0.90)),
        "q95": _json_float(clean.quantile(0.95)),
    }
    return summary, quantiles


def build_model_card_payload(run: "ModelComparisonRun", candidate: "ModelResult") -> dict[str, Any]:
    """Build model_card.json payload from an evaluated ML result."""
    best_baseline = run.get_best_baseline_result()
    validation_residuals = _residual_rows(run, candidate.model_name)
    residual_summary, residual_quantiles = _residual_summary(validation_residuals.get("residual", pd.Series(dtype=float)))
    readiness_record = run.readiness_assessment.to_record()

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": run.target,
        "scope_name": run.scope_name,
        "scope_id": run.scope_id,
        "scope_selectors": run.scope_selectors or {},
        "aggregation_level": run.aggregation_level,
        "model_name": candidate.model_name,
        "feature_names": list(candidate.feature_names),
        "required_features": [
            f for f in candidate.feature_names
            if f not in set(getattr(candidate, "seasonal_features", ()))
        ],
        "readiness_status": run.readiness_assessment.status,
        "readiness_reasons": readiness_record.get("issue_codes", ""),
        "holdout_metrics": candidate.metrics,
        "baseline_metrics": best_baseline.metrics if best_baseline is not None else {},
        "best_baseline_name": run.best_baseline_name,
        "comparison_note": candidate.comparison_note,
        "ml_beats_baseline": bool(candidate.beats_best_baseline),
        "training_period_start": run.training_period_start,
        "training_period_end": run.training_period_end,
        "train_rows": run.train_rows,
        "holdout_rows": run.holdout_rows,
        "residual_summary": residual_summary,
        "residual_quantiles": residual_quantiles,
        "preprocessing_assumptions": {
            "canonical_format": "long",
            "pivot_aggregation_level": run.aggregation_level,
            "default_inference_level": "sample_point_level",
            "target_not_used_as_feature": True,
            "censored_values_use_value_approx_and_metadata_is_not_recreated_at_inference": True,
        },
        "selection_mode": getattr(run, "selection_mode", "auto"),
        "forced_features": list(getattr(run, "forced_features", ())),
        "combined_score": _json_float(candidate.combined_score),
        "stability_ratio": _json_float(candidate.stability_ratio),
        "combined_score_used_fallback": bool(candidate.combined_score_used_fallback),
        "combined_score_weight_holdout": _json_float(getattr(run, "combined_score_weight_holdout", 0.4)),
        "combined_score_weight_backtest": _json_float(getattr(run, "combined_score_weight_backtest", 0.6)),
        "seasonal_feature": getattr(run, "seasonal_feature", "none"),
        "seasonal_feature_names": list(getattr(candidate, "seasonal_features", ())),
        "warning": ESTIMATED_VALUE_WARNING,
    }


def save_deployable_model_package(
    run: "ModelComparisonRun",
    candidate: "ModelResult",
    output_dir: str | Path,
) -> Path | None:
    """Save a deployable model package, or return None if the model is not deployable."""
    if run.readiness_assessment.status == "unsuitable":
        return None
    if candidate.is_baseline or candidate.estimator is None or candidate.status != "fitted" or not candidate.feature_names:
        return None
    if run.target in candidate.feature_names:
        raise ValueError("Refusing to save deployable package: target is included in feature_names.")

    package_dir = Path(output_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    payload = build_model_card_payload(run, candidate)
    dump_model_card(payload, package_dir / "model_card.json")
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "model_name": candidate.model_name,
            "target": run.target,
            "feature_names": list(candidate.feature_names),
            "estimator": candidate.estimator,
        },
        package_dir / "model.joblib",
    )

    seasonal_set = set(getattr(candidate, "seasonal_features", ()))
    feature_schema = pd.DataFrame(
        [
            {
                "feature": feature,
                "required": feature not in seasonal_set,
                "feature_kind": "seasonal" if feature in seasonal_set else "indicator",
                "dtype": "numeric",
            }
            for feature in candidate.feature_names
        ]
    )
    feature_schema.to_json(package_dir / "feature_schema.json", orient="records", force_ascii=False, indent=2)

    validation_residuals = _residual_rows(run, candidate.model_name)
    if not validation_residuals.empty:
        validation_residuals.to_csv(package_dir / "validation_residuals.csv", index=False, encoding="utf-8-sig")

    return package_dir


def load_model_package(package_dir: str | Path) -> LoadedModelPackage:
    """Load a deployable model package from disk."""
    root = Path(package_dir)
    card = load_model_card(root / "model_card.json")
    model_payload = joblib.load(root / "model.joblib")
    estimator = model_payload["estimator"]
    model_features = tuple(str(feature) for feature in model_payload.get("feature_names", ()))
    if model_features != card.feature_names:
        raise ValueError("Model package feature_names mismatch between model.joblib and model_card.json.")
    if str(model_payload.get("target")) != card.target:
        raise ValueError("Model package target mismatch between model.joblib and model_card.json.")
    return LoadedModelPackage(package_dir=root, model_card=card, estimator=estimator)
