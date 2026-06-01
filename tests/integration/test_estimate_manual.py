import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tests.helpers import build_linear_raw_dataframe
from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.inference.engine import (
    estimate_manual_values,
    manual_input_feature_names,
)
from water_analysis.inference.package import load_model_package, save_deployable_model_package
from water_analysis.inference.results import save_inference_outputs
from water_analysis.io.schemas import REQUIRED_TARGETS
from water_analysis.modeling.trainer import compare_models_in_scope
from water_analysis.preprocessing.long_format import build_canonical_long_format

TARGET = REQUIRED_TARGETS[0]
FEATURE_A = REQUIRED_TARGETS[1]
FEATURE_B = REQUIRED_TARGETS[2]


def _train_package(tmp_path: Path, *, scope_name: str = "global") -> Path:
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
    return package_dir


def test_manual_input_feature_names_excludes_target(tmp_path: Path) -> None:
    package = load_model_package(_train_package(tmp_path))
    features = manual_input_feature_names(package.model_card)
    assert TARGET not in features
    assert set(features).issubset({FEATURE_A, FEATURE_B})
    assert features  # at least one indicator feature


def test_estimate_manual_single_sample_predicts(tmp_path: Path) -> None:
    package = load_model_package(_train_package(tmp_path))
    features = manual_input_feature_names(package.model_card)

    sample = {"SampleDate": "15.01.2018"}
    sample.update({feature: 10.0 for feature in features})
    result = estimate_manual_values(package, pd.DataFrame([sample]))

    assert result.predicted_rows == 1
    assert result.predictions["prediction_status"].eq("estimated").all()
    estimated = result.estimated_values_long
    assert estimated["ValueSource"].eq("estimated").all()
    assert estimated["IsEstimated"].eq(True).all()
    assert estimated["OriginalValuePresent"].eq(False).all()


def test_estimate_manual_uses_clearly_synthetic_point_code(tmp_path: Path) -> None:
    # The helper trains on synthetic point code 00000000001.10110.0010. The manual
    # path must NOT reuse the training point code: the placeholder must
    # zero the identifying parts so it cannot collide with a real point.
    package = load_model_package(_train_package(tmp_path))
    features = manual_input_feature_names(package.model_card)
    sample = {"SampleDate": "15.01.2018"}
    sample.update({feature: 10.0 for feature in features})
    result = estimate_manual_values(package, pd.DataFrame([sample]))

    codes = result.predictions["FullPointCode"].astype(str).tolist()
    assert codes  # there is a prediction row
    for code in codes:
        oktmo = code.split(".")[0]
        assert set(oktmo) == {"0"}, f"synthetic OKTMO must be zeroed, got {code!r}"
        assert "00000000001" not in code  # never the training point code


def test_estimate_manual_works_for_point_scope_model(tmp_path: Path) -> None:
    # A point-scope model previously forced reuse of the real FullPointCode.
    # The direct-slice path must estimate without resurrecting that code.
    package = load_model_package(_train_package(tmp_path, scope_name="point"))
    assert package.model_card.scope_name == "point"
    features = manual_input_feature_names(package.model_card)
    sample = {"SampleDate": "15.01.2018"}
    sample.update({feature: 10.0 for feature in features})
    result = estimate_manual_values(package, pd.DataFrame([sample]))

    assert result.predicted_rows == 1
    assert result.summary["scope_name"] == "point"
    assert (result.predictions["FullPointCode"].astype(str).str.startswith("00000000000")).all()


def test_estimate_manual_batch_rows_kept_distinct(tmp_path: Path) -> None:
    package = load_model_package(_train_package(tmp_path))
    features = manual_input_feature_names(package.model_card)

    # Two samples sharing the same date must not collapse into one prediction.
    rows = []
    for value in (5.0, 20.0):
        row = {"SampleDate": "15.01.2018"}
        row.update({feature: value for feature in features})
        rows.append(row)
    result = estimate_manual_values(package, pd.DataFrame(rows))

    assert result.rows_for_estimation == 2
    assert result.predicted_rows == 2
    # Different inputs should give different estimates.
    values = result.predictions.loc[
        result.predictions["prediction_status"] == "estimated", "predicted_value"
    ].tolist()
    assert len(values) == 2
    assert values[0] != values[1]


def test_estimate_manual_outputs_are_exportable(tmp_path: Path) -> None:
    package = load_model_package(_train_package(tmp_path))
    features = manual_input_feature_names(package.model_card)
    sample = {"SampleDate": "15.01.2018"}
    sample.update({feature: 12.0 for feature in features})
    result = estimate_manual_values(package, pd.DataFrame([sample]))

    saved = save_inference_outputs(
        result,
        tmp_path / "manual_out",
        model_card_payload=package.model_card.payload,
        run_parameters={"input": "manual_entry", "model_package": str(package.package_dir)},
    )
    assert Path(saved["predictions"]).exists()
    assert Path(saved["predictions_xlsx"]).exists()
    assert Path(saved["estimated_values_long_xlsx"]).exists()


def test_estimate_manual_cli_prints_and_saves(tmp_path: Path) -> None:
    package_dir = _train_package(tmp_path)
    output_dir = tmp_path / "manual_cli"

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["PYTHONIOENCODING"] = "utf-8"
    command = [
        sys.executable,
        "-m",
        "water_analysis.cli",
        "estimate-manual",
        "--model-package",
        str(package_dir),
        "--value",
        f"{FEATURE_A}=20",
        "--value",
        f"{FEATURE_B}=5",
        "--date",
        "2018-01-15",
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(command, cwd=Path.cwd(), env=env, capture_output=True, text=True, encoding="utf-8")

    assert result.returncode == 0, result.stderr
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "predictions.xlsx").exists()
    predictions = pd.read_csv(output_dir / "predictions.csv")
    assert predictions["prediction_status"].eq("estimated").all()
    card = json.loads((output_dir / "model_card_used.json").read_text(encoding="utf-8"))
    assert card["target"] == TARGET


def test_estimate_manual_cli_list_features(tmp_path: Path) -> None:
    package_dir = _train_package(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["PYTHONIOENCODING"] = "utf-8"
    command = [
        sys.executable,
        "-m",
        "water_analysis.cli",
        "estimate-manual",
        "--model-package",
        str(package_dir),
        "--list-features",
    ]
    result = subprocess.run(command, cwd=Path.cwd(), env=env, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert TARGET in combined
