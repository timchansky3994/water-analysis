"""ML model registry and pipeline construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None
    XGBOOST_AVAILABLE = False


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a trainable model."""

    name: str
    family: str
    builder: Callable[[], Pipeline]


def _linear_pipeline(estimator: object) -> Pipeline:
    """Build a pipeline for linear-like estimators."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _tree_pipeline(estimator: object) -> Pipeline:
    """Build a pipeline for tree-based estimators."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def available_model_specs() -> dict[str, ModelSpec]:
    """Return the supported ML model specs keyed by model name."""
    specs: dict[str, ModelSpec] = {
        "bayesian_ridge": ModelSpec(
            name="bayesian_ridge",
            family="linear",
            builder=lambda: _linear_pipeline(BayesianRidge()),
        ),
        "ridge": ModelSpec(
            name="ridge",
            family="linear",
            builder=lambda: _linear_pipeline(Ridge(alpha=1.0)),
        ),
        "elastic_net": ModelSpec(
            name="elastic_net",
            family="linear",
            builder=lambda: _linear_pipeline(ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42, max_iter=5000)),
        ),
        "huber": ModelSpec(
            name="huber",
            family="linear",
            builder=lambda: _linear_pipeline(HuberRegressor()),
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            family="tree",
            builder=lambda: _tree_pipeline(
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=2,
                )
            ),
        ),
        "hist_gradient_boosting": ModelSpec(
            name="hist_gradient_boosting",
            family="tree",
            builder=lambda: _tree_pipeline(HistGradientBoostingRegressor(random_state=42)),
        ),
    }

    if XGBOOST_AVAILABLE and XGBRegressor is not None:
        specs["xgboost"] = ModelSpec(
            name="xgboost",
            family="tree",
            builder=lambda: _tree_pipeline(
                XGBRegressor(
                    random_state=42,
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                )
            ),
        )

    return specs
