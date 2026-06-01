"""Persistence helpers for modeling runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import joblib

from water_analysis.inference.package import save_deployable_model_package
from water_analysis.reporting.xlsx_export import save_dataframe_csv_and_xlsx

if TYPE_CHECKING:
    from water_analysis.modeling.trainer import ModelComparisonRun


def save_model_comparison_run(
    run: "ModelComparisonRun",
    output_dir: str | Path,
    *,
    chosen_model_name: str | None = None,
) -> dict[str, str]:
    """Persist comparison artifacts and optionally a chosen trained model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, str] = {}

    comparison_saved = save_dataframe_csv_and_xlsx(
        run.comparison_df,
        output_path / "comparison_summary.csv",
        table_name="comparison_summary",
    )
    saved_paths["comparison_summary"] = str(comparison_saved["csv"])
    saved_paths["comparison_summary_xlsx"] = str(comparison_saved["xlsx"])

    predictions_saved = save_dataframe_csv_and_xlsx(
        run.holdout_predictions_df,
        output_path / "holdout_predictions.csv",
        table_name="holdout_predictions",
    )
    saved_paths["holdout_predictions"] = str(predictions_saved["csv"])
    saved_paths["holdout_predictions_xlsx"] = str(predictions_saved["xlsx"])

    if not run.backtest_df.empty:
        backtest_saved = save_dataframe_csv_and_xlsx(
            run.backtest_df,
            output_path / "backtest_summary.csv",
            table_name="backtest_summary",
        )
        saved_paths["backtest_summary"] = str(backtest_saved["csv"])
        saved_paths["backtest_summary_xlsx"] = str(backtest_saved["xlsx"])

    readiness_path = output_path / "readiness.json"
    readiness_path.write_text(
        json.dumps(run.readiness_assessment.to_record(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    saved_paths["readiness"] = str(readiness_path)

    metadata_path = output_path / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "target": run.target,
                "scope_name": run.scope_name,
                "scope_id": run.scope_id,
                "selected_features": list(run.selected_features),
                "best_baseline_name": run.best_baseline_name,
                "warnings": list(run.warnings),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    saved_paths["run_metadata"] = str(metadata_path)

    if chosen_model_name:
        candidate = next((result for result in run.results if result.model_name == chosen_model_name), None)
        if candidate and candidate.estimator is not None and not candidate.is_baseline:
            model_path = output_path / f"{chosen_model_name}.joblib"
            joblib.dump(
                {
                    "model_name": candidate.model_name,
                    "target": run.target,
                    "scope_name": run.scope_name,
                    "scope_id": run.scope_id,
                    "features": list(candidate.feature_names),
                    "estimator": candidate.estimator,
                    "metrics": candidate.metrics,
                },
                model_path,
            )
            saved_paths["model"] = str(model_path)
            package_path = save_deployable_model_package(run, candidate, output_path / "best_model_package")
            if package_path is not None:
                saved_paths["deployable_model_package"] = str(package_path)
    else:
        candidate = run.get_best_ml_result()
        if candidate is not None:
            package_path = save_deployable_model_package(run, candidate, output_path / "best_model_package")
            if package_path is not None:
                saved_paths["deployable_model_package"] = str(package_path)

    return saved_paths
