"""End-to-end report bundle generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from water_analysis.analysis.correlations import CorrelationAnalysis, run_correlation_analysis
from water_analysis.analysis.scopes import ScopeSlice
from water_analysis.analysis.seasonality import SeasonalityAnalysis, analyze_seasonality
from water_analysis.inference.package import save_deployable_model_package
from water_analysis.modeling.trainer import ModelComparisonRun, ModelingNotAllowedError, compare_models_in_scope
from water_analysis.profiling.passport import ProfileReport, build_profile_reports
from water_analysis.profiling.readiness import ReadinessAssessment, assess_readiness
from water_analysis.reporting.plots import (
    plot_backtest,
    plot_cooccurrence_heatmap,
    plot_correlation_heatmap,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_seasonal_profile,
)
from water_analysis.reporting.specialist_summary import SpecialistSummaryInput, build_specialist_summary
from water_analysis.reporting.xlsx_export import save_dataframe_csv_and_xlsx


@dataclass(frozen=True)
class ReportBundle:
    """Result of a generated report bundle."""

    output_dir: Path
    summary_path: Path
    metadata_path: Path
    generated_files: dict[str, str]
    readiness_assessment: ReadinessAssessment
    model_run: ModelComparisonRun | None


def _ensure_dir(path: Path) -> Path:
    """Create a directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(raw_text: str) -> str:
    """Convert a string to a filesystem-safe path component."""
    allowed = [character if character.isalnum() or character in "._-" else "_" for character in raw_text]
    return "".join(allowed)


def _save_table(dataframe: pd.DataFrame, path: Path, *, table_name: str) -> dict[str, str]:
    """Save a dataframe to CSV and XLSX."""
    saved = save_dataframe_csv_and_xlsx(dataframe, path, table_name=table_name)
    return {table_name: str(saved["csv"]), f"{table_name}_xlsx": str(saved["xlsx"])}


def _cooccurrence_to_long(profile_report: ProfileReport) -> pd.DataFrame:
    """Flatten a co-occurrence matrix for CSV export."""
    matrix = profile_report.cooccurrence_matrix
    if matrix.empty:
        return pd.DataFrame(columns=["indicator_x", "indicator_y", "n_shared"])
    long_df = matrix.stack().reset_index()
    long_df.columns = ["indicator_x", "indicator_y", "n_shared"]
    return long_df


def _serialize_profile(profile_report: ProfileReport, tables_dir: Path) -> dict[str, str]:
    """Save profile-related tables."""
    saved: dict[str, str] = {}
    saved.update(_save_table(profile_report.summary_frame(), tables_dir / "profile_summary.csv", table_name="profile_summary"))
    saved.update(
        _save_table(profile_report.indicator_observations, tables_dir / "indicator_observations.csv", table_name="indicator_observations")
    )
    saved.update(_save_table(profile_report.missingness, tables_dir / "missingness.csv", table_name="missingness"))
    saved.update(_save_table(profile_report.point_type_coverage, tables_dir / "point_type_coverage.csv", table_name="point_type_coverage"))
    saved.update(_save_table(profile_report.constant_series, tables_dir / "constant_series.csv", table_name="constant_series"))
    saved.update(_save_table(_cooccurrence_to_long(profile_report), tables_dir / "cooccurrence.csv", table_name="cooccurrence"))
    return saved


def _serialize_correlations(correlation_analysis: CorrelationAnalysis, tables_dir: Path) -> dict[str, str]:
    """Save correlation outputs."""
    saved: dict[str, str] = {}
    saved.update(_save_table(correlation_analysis.results, tables_dir / "correlation_results.csv", table_name="correlation_results"))
    saved.update(_save_table(correlation_analysis.diagnostics, tables_dir / "correlation_diagnostics.csv", table_name="correlation_diagnostics"))
    return saved


def _serialize_model_run(model_run: ModelComparisonRun | None, tables_dir: Path, metadata_dir: Path, models_dir: Path) -> dict[str, str]:
    """Save modeling outputs into the bundle layout."""
    if model_run is None:
        return {}

    saved: dict[str, str] = {}
    saved.update(_save_table(model_run.comparison_df, tables_dir / "comparison_summary.csv", table_name="comparison_summary"))
    saved.update(_save_table(model_run.holdout_predictions_df, tables_dir / "holdout_predictions.csv", table_name="holdout_predictions"))
    if not model_run.backtest_df.empty:
        saved.update(_save_table(model_run.backtest_df, tables_dir / "backtest_summary.csv", table_name="backtest_summary"))
    saved.update(
        _save_table(model_run.feature_selection.candidate_table, tables_dir / "feature_selection_candidates.csv", table_name="feature_selection_candidates")
    )

    best_ml = model_run.get_best_ml_result()
    if best_ml is not None and not best_ml.interpretability_df.empty:
        interpretability_saved = _save_table(
            best_ml.interpretability_df,
            tables_dir / f"{best_ml.model_name}_interpretability.csv",
            table_name="interpretability",
        )
        saved.update(interpretability_saved)

    metadata_path = metadata_dir / "model_run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "scope_name": model_run.scope_name,
                "scope_id": model_run.scope_id,
                "target": model_run.target,
                "selected_features": list(model_run.selected_features),
                "best_baseline_name": model_run.best_baseline_name,
                "warnings": list(model_run.warnings),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    saved["model_run_metadata"] = str(metadata_path)

    if best_ml is not None and best_ml.estimator is not None:
        import joblib

        model_path = models_dir / f"{best_ml.model_name}.joblib"
        joblib.dump(
            {
                "target": model_run.target,
                "scope_name": model_run.scope_name,
                "scope_id": model_run.scope_id,
                "model_name": best_ml.model_name,
                "feature_names": list(best_ml.feature_names),
                "estimator": best_ml.estimator,
                "metrics": best_ml.metrics,
            },
            model_path,
        )
        saved["best_model"] = str(model_path)
        package_path = save_deployable_model_package(model_run, best_ml, models_dir / "best_model_package")
        if package_path is not None:
            saved["deployable_model_package"] = str(package_path)

    return saved


def generate_report_bundle(
    scope_slice: ScopeSlice,
    *,
    target: str,
    output_dir: str | Path,
    run_parameters: dict[str, Any],
    correlation_methods: list[str],
    correlation_min_shared_samples: int,
    readiness_kwargs: dict[str, Any],
    modeling_kwargs: dict[str, Any],
    reporting_kwargs: dict[str, Any],
    seasonality_granularity: str = "season",
    seasonality_min_group_size: int = 5,
    progress_callback: Callable[[str, float], None] | None = None,
) -> ReportBundle:
    """Generate a full specialist-facing report bundle for one scope and target."""

    def _progress(stage: str, fraction: float) -> None:
        if progress_callback is not None:
            progress_callback(stage, fraction)

    output_path = Path(output_dir)
    tables_dir = _ensure_dir(output_path / "tables")
    plots_dir = _ensure_dir(output_path / "plots")
    metadata_dir = _ensure_dir(output_path / "metadata")
    summary_dir = _ensure_dir(output_path / "summary")
    models_dir = _ensure_dir(output_path / "models")

    _progress("Профилирование данных", 0.10)
    profile_report = build_profile_reports([scope_slice])[0]
    _progress("Проверка пригодности к моделированию", 0.20)
    readiness_assessment = assess_readiness([scope_slice], targets=[target], **readiness_kwargs)[0]
    _progress("Сезонный анализ", 0.28)
    seasonality_analysis: SeasonalityAnalysis | None = None
    try:
        seasonality_analysis = analyze_seasonality(
            scope_slice,
            target=target,
            granularity=seasonality_granularity,  # type: ignore[arg-type]
            min_group_size=seasonality_min_group_size,
        )
    except Exception:
        pass
    _progress("Поиск корреляций", 0.35)
    correlation_analysis = run_correlation_analysis(
        [scope_slice],
        targets=[target],
        methods=correlation_methods,
        min_shared_samples=correlation_min_shared_samples,
    )

    model_run: ModelComparisonRun | None = None
    if readiness_assessment.status != "unsuitable":
        _progress("Сравнение моделей", 0.55)
        try:
            model_run = compare_models_in_scope(scope_slice, target=target, **modeling_kwargs)
        except (ModelingNotAllowedError, ValueError):
            model_run = None

    _progress("Сохранение таблиц", 0.80)
    readiness_csv_path = tables_dir / "readiness.csv"
    readiness_saved = save_dataframe_csv_and_xlsx(
        pd.DataFrame([readiness_assessment.to_record()]),
        readiness_csv_path,
        table_name="readiness",
    )
    readiness_json_path = metadata_dir / "readiness.json"
    readiness_json_path.write_text(
        json.dumps(readiness_assessment.to_record(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path = summary_dir / "specialist_summary.md"
    generated_files: dict[str, str] = {
        "readiness_csv": str(readiness_saved["csv"]),
        "readiness_xlsx": str(readiness_saved["xlsx"]),
        "readiness_json": str(readiness_json_path),
    }
    generated_files.update(_serialize_profile(profile_report, tables_dir))
    generated_files.update(_serialize_correlations(correlation_analysis, tables_dir))
    generated_files.update(_serialize_model_run(model_run, tables_dir, metadata_dir, models_dir))
    if seasonality_analysis is not None:
        if not seasonality_analysis.group_stats.empty:
            generated_files.update(
                _save_table(seasonality_analysis.group_stats, tables_dir / "seasonality_group_stats.csv", table_name="seasonality_group_stats")
            )
        if not seasonality_analysis.per_season_correlations.empty:
            generated_files.update(
                _save_table(seasonality_analysis.per_season_correlations, tables_dir / "seasonality_correlations.csv", table_name="seasonality_correlations")
            )
        seasonality_meta_path = metadata_dir / "seasonality_analysis.json"
        seasonality_meta_path.write_text(
            json.dumps(
                {
                    "target": seasonality_analysis.target,
                    "granularity": seasonality_analysis.granularity,
                    "seasonal_pattern_detected": seasonality_analysis.seasonal_pattern_detected,
                    "pattern_test": seasonality_analysis.pattern_test,
                    "diagnostics": seasonality_analysis.diagnostics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        generated_files["seasonality_metadata"] = str(seasonality_meta_path)

    summary_text = build_specialist_summary(
        SpecialistSummaryInput(
            run_parameters=run_parameters,
            profile_report=profile_report,
            readiness_assessment=readiness_assessment,
            correlation_analysis=correlation_analysis,
            model_run=model_run,
            deployable_model_package=generated_files.get("deployable_model_package"),
            seasonality_analysis=seasonality_analysis,
        ),
        top_correlations=int(reporting_kwargs["top_correlations"]),
    )

    summary_path.write_text(summary_text, encoding="utf-8")
    generated_files["summary"] = str(summary_path)

    _progress("Построение графиков", 0.90)
    plots_dpi = int(reporting_kwargs["plots_dpi"])
    generated_files["predicted_vs_actual_plot"] = plot_predicted_vs_actual(model_run, plots_dir / "predicted_vs_actual.png", dpi=plots_dpi)
    generated_files["residuals_plot"] = plot_residuals(model_run, plots_dir / "residuals.png", dpi=plots_dpi)
    generated_files["backtest_plot"] = plot_backtest(model_run, plots_dir / "backtest.png", dpi=plots_dpi)
    generated_files["correlation_heatmap"] = plot_correlation_heatmap(
        correlation_analysis.results,
        plots_dir / "correlation_heatmap.png",
        target=target,
        max_features=int(reporting_kwargs["heatmap_max_features"]),
        dpi=plots_dpi,
    )
    generated_files["cooccurrence_heatmap"] = plot_cooccurrence_heatmap(
        profile_report.cooccurrence_matrix,
        plots_dir / "cooccurrence_heatmap.png",
        max_features=int(reporting_kwargs["heatmap_max_features"]),
        dpi=plots_dpi,
    )
    if seasonality_analysis is not None:
        generated_files["seasonal_profile"] = plot_seasonal_profile(
            seasonality_analysis,
            plots_dir / "seasonal_profile.png",
            dpi=plots_dpi,
        )

    metadata_path = metadata_dir / "run_parameters.json"
    metadata_path.write_text(json.dumps(run_parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    generated_files["run_parameters"] = str(metadata_path)

    _progress("Готово", 1.0)
    return ReportBundle(
        output_dir=output_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
        generated_files=generated_files,
        readiness_assessment=readiness_assessment,
        model_run=model_run,
    )


def default_report_output_dir(config: dict[str, Any], *, scope_id: str, target: str) -> Path:
    """Build a timestamped default report output directory."""
    reporting_config = config.get("reporting", {})
    timestamp_format = reporting_config.get("timestamp_format", "%Y%m%d_%H%M%S")
    root_dir = Path(reporting_config.get("reports_dir", "reports"))
    timestamp = datetime.now().strftime(timestamp_format)
    return root_dir / timestamp / _safe_name(scope_id) / _safe_name(target)
