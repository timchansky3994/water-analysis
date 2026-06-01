"""Model card contracts for deployable estimation packages."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
ESTIMATED_VALUE_WARNING = (
    "Расчетные значения являются аналитическими оценками модели; "
    "они не являются лабораторными измерениями и не должны заменять исходные наблюдения."
)

REQUIRED_MODEL_CARD_FIELDS: tuple[str, ...] = (
    "schema_version",
    "created_at",
    "target",
    "scope_name",
    "scope_selectors",
    "aggregation_level",
    "model_name",
    "feature_names",
    "required_features",
    "readiness_status",
    "readiness_reasons",
    "holdout_metrics",
    "baseline_metrics",
    "best_baseline_name",
    "comparison_note",
    "ml_beats_baseline",
    "training_period_start",
    "training_period_end",
    "train_rows",
    "holdout_rows",
    "preprocessing_assumptions",
    "warning",
)

# Added in schema 1.1; absent in older packages — log WARNING but do not reject.
RECOMMENDED_MODEL_CARD_FIELDS: tuple[str, ...] = (
    "selection_mode",
    "forced_features",
    "combined_score",
    "stability_ratio",
    "combined_score_used_fallback",
    "combined_score_weight_holdout",
    "combined_score_weight_backtest",
)


@dataclass(frozen=True)
class ModelCard:
    """Validated metadata needed to apply a saved model package."""

    payload: dict[str, Any]

    @property
    def target(self) -> str:
        """Return the target indicator name."""
        return str(self.payload["target"])

    @property
    def scope_name(self) -> str:
        """Return the training scope name."""
        return str(self.payload["scope_name"])

    @property
    def scope_id(self) -> str:
        """Return the training scope id (falls back to scope name if absent)."""
        return str(self.payload.get("scope_id", self.payload["scope_name"]))

    @property
    def scope_selectors(self) -> dict[str, str]:
        """Return the training scope selectors."""
        return {str(key): str(value) for key, value in dict(self.payload.get("scope_selectors", {})).items()}

    @property
    def aggregation_level(self) -> str:
        """Return the pivot aggregation level expected by the model."""
        return str(self.payload["aggregation_level"])

    @property
    def model_name(self) -> str:
        """Return the saved model name."""
        return str(self.payload["model_name"])

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return all model input feature names."""
        return tuple(str(feature) for feature in self.payload.get("feature_names", ()))

    @property
    def required_features(self) -> tuple[str, ...]:
        """Return features expected by the model at inference time."""
        return tuple(str(feature) for feature in self.payload.get("required_features", ()))

    @property
    def ml_beats_baseline(self) -> bool:
        """Return whether ML beat the best baseline in validation."""
        return bool(self.payload.get("ml_beats_baseline", False))

    @property
    def selection_mode(self) -> str:
        """Return the feature selection mode used during training."""
        return str(self.payload.get("selection_mode", "auto"))

    @property
    def forced_features(self) -> tuple[str, ...]:
        """Return specialist-forced features (empty tuple for auto mode)."""
        return tuple(str(f) for f in self.payload.get("forced_features", ()))

    @property
    def residual_quantiles(self) -> dict[str, float]:
        """Return residual quantiles used for rough prediction intervals."""
        values = self.payload.get("residual_quantiles", {})
        return {str(key): float(value) for key, value in dict(values).items()}

    @property
    def combined_score(self) -> float | None:
        """Return the combined score used for model selection, or None for older packages."""
        value = self.payload.get("combined_score")
        return float(value) if value is not None else None

    @property
    def stability_ratio(self) -> float | None:
        """Return backtest/holdout ratio of the primary metric, or None for older packages."""
        value = self.payload.get("stability_ratio")
        return float(value) if value is not None else None

    @property
    def combined_score_weights(self) -> tuple[float, float]:
        """Return (weight_holdout, weight_backtest) used during training."""
        w_holdout = self.payload.get("combined_score_weight_holdout", 0.4)
        w_backtest = self.payload.get("combined_score_weight_backtest", 0.6)
        return float(w_holdout), float(w_backtest)

    @property
    def seasonal_feature(self) -> str:
        """Return the seasonal feature mode used during training (default 'none' for older packages)."""
        return str(self.payload.get("seasonal_feature", "none"))

    @property
    def seasonal_feature_names(self) -> tuple[str, ...]:
        """Return the seasonal feature column names used by the model (empty for older packages)."""
        return tuple(str(f) for f in self.payload.get("seasonal_feature_names", ()))


def validate_model_card(payload: dict[str, Any]) -> ModelCard:
    """Validate a model card payload and return a typed wrapper."""
    missing = [field for field in REQUIRED_MODEL_CARD_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Model card is missing required fields: {', '.join(missing)}")

    missing_recommended = [f for f in RECOMMENDED_MODEL_CARD_FIELDS if f not in payload]
    if missing_recommended:
        _LOGGER.warning(
            "Model card is missing recommended fields (older package): %s. "
            "Defaults will be used: selection_mode='auto', forced_features=[].",
            ", ".join(missing_recommended),
        )

    feature_names = tuple(str(feature) for feature in payload.get("feature_names", ()))
    required_features = tuple(str(feature) for feature in payload.get("required_features", ()))
    if not feature_names:
        raise ValueError("Model card has no feature_names; the package is not deployable.")
    if payload["target"] in feature_names:
        raise ValueError("Model card is invalid: target is listed as a feature.")
    if not set(required_features).issubset(set(feature_names)):
        raise ValueError("Model card required_features must be a subset of feature_names.")
    if payload.get("readiness_status") == "unsuitable":
        raise ValueError("Model card is invalid: unsuitable models are not deployable.")
    if payload.get("aggregation_level") != "sample_point_level":
        raise ValueError("Only sample_point_level model packages are currently supported for inference.")

    return ModelCard(payload=payload)


def load_model_card(path: str | Path) -> ModelCard:
    """Load and validate a model_card.json file."""
    card_path = Path(path)
    payload = json.loads(card_path.read_text(encoding="utf-8"))
    return validate_model_card(payload)


def dump_model_card(payload: dict[str, Any], path: str | Path) -> Path:
    """Write model card JSON with stable UTF-8 formatting."""
    card_path = Path(path)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return card_path
