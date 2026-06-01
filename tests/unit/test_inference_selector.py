import pandas as pd
import pytest

from water_analysis.inference.model_card import validate_model_card
from water_analysis.inference.selector import check_feature_compatibility


def _card(feature_names: list[str] | None = None):
    return validate_model_card(
        {
            "schema_version": "1.0",
            "created_at": "2026-05-06T00:00:00+00:00",
            "target": "target",
            "scope_name": "global",
            "scope_selectors": {},
            "aggregation_level": "sample_point_level",
            "model_name": "ridge",
            "feature_names": feature_names or ["feature_a", "feature_b"],
            "required_features": feature_names or ["feature_a", "feature_b"],
            "readiness_status": "suitable",
            "readiness_reasons": "",
            "holdout_metrics": {"rmse": 1.0},
            "baseline_metrics": {"rmse": 2.0},
            "best_baseline_name": "median_baseline",
            "comparison_note": "ml_beats_baseline",
            "ml_beats_baseline": True,
            "training_period_start": "2020-01-01T00:00:00",
            "training_period_end": "2020-02-01T00:00:00",
            "train_rows": 10,
            "holdout_rows": 3,
            "preprocessing_assumptions": {"pivot_aggregation_level": "sample_point_level"},
            "warning": "Estimated values are not laboratory measurements.",
        }
    )


def test_feature_compatibility_reports_missing_feature_columns() -> None:
    pivot_df = pd.DataFrame({"feature_a": [1.0], "other": [2.0]})

    result = check_feature_compatibility(_card(), pivot_df)

    assert result.compatible is True
    assert result.available_features == ("feature_a",)
    assert result.missing_features == ("feature_b",)


def test_feature_compatibility_rejects_no_available_features() -> None:
    pivot_df = pd.DataFrame({"other": [2.0]})

    result = check_feature_compatibility(_card(), pivot_df)

    assert result.compatible is False
    assert "no_model_features_available" in result.reasons


def test_feature_compatibility_rejects_target_feature_before_prediction() -> None:
    with pytest.raises(ValueError, match="target is listed as a feature"):
        _card(["target", "feature_a"])
