import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tests.helpers import build_linear_raw_dataframe, write_raw_csv
from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.io.schemas import REQUIRED_TARGETS
from water_analysis.inference.engine import estimate_missing_values
from water_analysis.inference.package import load_model_package, save_deployable_model_package
from water_analysis.inference.results import save_inference_outputs
from water_analysis.modeling.trainer import compare_models_in_scope
from water_analysis.preprocessing.long_format import build_canonical_long_format

TARGET = REQUIRED_TARGETS[0]
FEATURE_A = REQUIRED_TARGETS[1]
FEATURE_B = REQUIRED_TARGETS[2]
INDICATOR_COLUMN = "Гигиенический показатель"


def _train_package(tmp_path: Path, *, scope_name: str = "global") -> tuple[Path, pd.DataFrame]:
    dates = [f"{day:02d}.01.2018" for day in range(1, 15)]
    raw_df = build_linear_raw_dataframe(
        dates=dates,
        target_values=[float(day) for day in range(1, 15)],
        feature_map={
            FEATURE_A: [float(day * 2) for day in range(1, 15)],
            FEATURE_B: [float(day) / 2 for day in range(1, 15)],
        },
    )
    long_df = build_canonical_long_format(raw_df)
    scope_slice = build_scope_slices(long_df, scope_name=scope_name)[0]
    run = compare_models_in_scope(
        scope_slice,
        target=TARGET,
        model_names=["bayesian_ridge"],
        test_size=0.3,
        min_train_size=6,
        min_target_observations=6,
        min_shared_samples=4,
        min_eligible_predictors=1,
        min_target_correlation=0.1,
        max_features=5,
    )
    model = run.get_best_ml_result()
    assert model is not None
    package_dir = save_deployable_model_package(run, model, tmp_path / "best_model_package")
    assert package_dir is not None
    return package_dir, raw_df


def _new_raw_without_target() -> pd.DataFrame:
    raw_df = build_linear_raw_dataframe(
        dates=["15.01.2018", "16.01.2018"],
        target_values=[15.0, 16.0],
        feature_map={
            FEATURE_A: [30.0, 32.0],
            FEATURE_B: [7.5, 8.0],
        },
    )
    return raw_df[raw_df[INDICATOR_COLUMN] != TARGET].copy()


def test_estimate_missing_from_saved_package_writes_estimated_long(tmp_path: Path) -> None:
    package_dir, _ = _train_package(tmp_path)
    input_path = write_raw_csv(_new_raw_without_target(), tmp_path / "new.csv")

    package = load_model_package(package_dir)
    result = estimate_missing_values(
        input_path,
        package,
        min_observed_features=1,
        min_feature_coverage=0.5,
    )
    saved = save_inference_outputs(
        result,
        tmp_path / "estimate",
        model_card_payload=package.model_card.payload,
        run_parameters={"input": str(input_path), "model_package": str(package_dir)},
    )

    assert result.predicted_rows == 2
    assert result.estimated_values_long["ValueSource"].eq("estimated").all()
    assert result.estimated_values_long["IsEstimated"].eq(True).all()
    assert Path(saved["predictions"]).exists()
    assert Path(saved["predictions_xlsx"]).exists()
    assert Path(saved["estimated_values_long_xlsx"]).exists()
    assert Path(saved["inference_diagnostics_xlsx"]).exists()
    assert Path(saved["inference_summary"]).exists()
    summary_text = Path(saved["inference_summary"]).read_text(encoding="utf-8")
    assert "расчетной оценке прогнозируемых значений" in summary_text
    assert "не являются лабораторными измерениями" in summary_text


def test_estimate_missing_refuses_incompatible_scope_without_fallback(tmp_path: Path) -> None:
    package_dir, _ = _train_package(tmp_path)
    input_path = write_raw_csv(_new_raw_without_target(), tmp_path / "new.csv")

    result = estimate_missing_values(
        input_path,
        load_model_package(package_dir),
        scope_name="point",
        point_code="00000000001.10110.0010",
        min_observed_features=1,
    )

    assert result.predicted_rows == 0
    assert "incompatible_scope" in result.diagnostics["reason"].iloc[0]


def test_missing_feature_columns_are_diagnostics_not_traceback(tmp_path: Path) -> None:
    package_dir, _ = _train_package(tmp_path)
    input_df = _new_raw_without_target()
    input_df = input_df[input_df[INDICATOR_COLUMN] == FEATURE_A].copy()
    input_path = write_raw_csv(input_df, tmp_path / "new_missing_feature.csv")

    result = estimate_missing_values(
        input_path,
        load_model_package(package_dir),
        min_observed_features=1,
        allow_missing_feature_columns=False,
    )

    if result.predicted_rows == 0:
        assert "missing_feature" in "|".join(result.diagnostics["reason"].astype(str))
    else:
        assert "missing_feature" in result.predictions["warnings"].iloc[0]


def test_estimate_missing_cli_generates_outputs(tmp_path: Path) -> None:
    package_dir, _ = _train_package(tmp_path)
    input_path = write_raw_csv(_new_raw_without_target(), tmp_path / "new.csv")
    output_dir = tmp_path / "estimate_cli"

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["PYTHONIOENCODING"] = "utf-8"
    command = [
        sys.executable,
        "-m",
        "water_analysis.cli",
        "estimate-missing",
        "--input",
        str(input_path),
        "--model-package",
        str(package_dir),
        "--output-dir",
        str(output_dir),
        "--min-observed-features",
        "1",
        "--min-feature-coverage",
        "0.5",
    ]
    result = subprocess.run(command, cwd=Path.cwd(), env=env, capture_output=True, text=True, encoding="utf-8")

    assert result.returncode == 0, result.stderr
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "predictions.xlsx").exists()
    assert (output_dir / "estimated_values_long.csv").exists()
    assert (output_dir / "estimated_values_long.xlsx").exists()
    assert (output_dir / "inference_diagnostics.xlsx").exists()
    predictions = pd.read_csv(output_dir / "predictions.csv")
    assert predictions["prediction_status"].eq("estimated").all()
    card = json.loads((output_dir / "model_card_used.json").read_text(encoding="utf-8"))
    assert card["target"] == TARGET
