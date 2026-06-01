import json
from pathlib import Path

import pytest

from water_analysis.inference.model_card import ESTIMATED_VALUE_WARNING, load_model_card, validate_model_card


def _payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "created_at": "2026-05-06T00:00:00+00:00",
        "target": "target",
        "scope_name": "global",
        "scope_selectors": {},
        "aggregation_level": "sample_point_level",
        "model_name": "ridge",
        "feature_names": ["feature_a", "feature_b"],
        "required_features": ["feature_a", "feature_b"],
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
        "residual_quantiles": {"q05": -1.0, "q95": 1.0},
        "preprocessing_assumptions": {"pivot_aggregation_level": "sample_point_level"},
        "warning": ESTIMATED_VALUE_WARNING,
    }


def test_load_model_card_validates_required_fields(tmp_path: Path) -> None:
    card_path = tmp_path / "model_card.json"
    card_path.write_text(json.dumps(_payload(), ensure_ascii=False), encoding="utf-8")

    card = load_model_card(card_path)

    assert card.target == "target"
    assert card.feature_names == ("feature_a", "feature_b")


def test_model_card_rejects_target_as_feature() -> None:
    payload = _payload()
    payload["feature_names"] = ["target", "feature_a"]
    payload["required_features"] = ["target", "feature_a"]

    with pytest.raises(ValueError, match="target is listed as a feature"):
        validate_model_card(payload)


def test_model_card_rejects_unsuitable_package() -> None:
    payload = _payload()
    payload["readiness_status"] = "unsuitable"

    with pytest.raises(ValueError, match="unsuitable"):
        validate_model_card(payload)
