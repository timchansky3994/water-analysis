from pathlib import Path

from water_analysis.io.schemas import REQUIRED_TARGETS
from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.preprocessing.long_format import build_canonical_long_format
from water_analysis.reporting.bundle import generate_report_bundle

from tests.helpers import build_linear_raw_dataframe


def test_report_bundle_generated_for_unsuitable_scope(tmp_path: Path) -> None:
    raw_df = build_linear_raw_dataframe(
        dates=["01.01.2018", "02.01.2018", "03.01.2018", "04.01.2018"],
        target_values=[5.0, 5.0, 5.0, 5.0],
        feature_map={"Цветность": [1.0, 1.0, 1.0, 1.0]},
    )
    long_df = build_canonical_long_format(raw_df)
    scope_slice = build_scope_slices(long_df, scope_name="global")[0]

    bundle = generate_report_bundle(
        scope_slice,
        target="Жесткость общая",
        output_dir=tmp_path / "report_bundle",
        run_parameters={"scope": "global", "target": "Жесткость общая"},
        correlation_methods=["spearman", "pearson"],
        correlation_min_shared_samples=2,
        readiness_kwargs={
            "min_target_observations": 2,
            "min_shared_samples": 2,
            "max_missing_ratio": 0.6,
            "heavy_censoring_ratio": 0.5,
            "min_eligible_predictors": 1,
        },
        modeling_kwargs={
            "model_names": ["bayesian_ridge"],
            "test_size": 0.5,
            "min_train_size": 2,
            "min_target_observations": 2,
            "min_shared_samples": 2,
            "max_missing_ratio": 0.6,
            "heavy_censoring_ratio": 0.5,
            "min_eligible_predictors": 1,
            "min_target_correlation": 0.1,
            "max_features": 5,
            "multicollinearity_threshold": 0.85,
        },
        reporting_kwargs={"plots_dpi": 100, "top_correlations": 5, "heatmap_max_features": 10},
    )

    assert bundle.output_dir.exists()
    assert bundle.summary_path.exists()
    assert (bundle.output_dir / "plots" / "predicted_vs_actual.png").exists()
    assert (bundle.output_dir / "tables" / "readiness.csv").exists()
    assert (bundle.output_dir / "tables" / "readiness.xlsx").exists()
    assert (bundle.output_dir / "tables" / "profile_summary.xlsx").exists()
    assert bundle.model_run is None

    summary_text = bundle.summary_path.read_text(encoding="utf-8")
    assert "Пригодность данных к моделированию" in summary_text
    assert "обучение модели не рекомендуется" in summary_text


def test_report_bundle_creates_best_model_package_and_xlsx_tables(tmp_path: Path) -> None:
    raw_df = build_linear_raw_dataframe(
        dates=[f"{day:02d}.01.2018" for day in range(1, 15)],
        target_values=[float(day) for day in range(1, 15)],
        feature_map={
            REQUIRED_TARGETS[1]: [float(day * 2) for day in range(1, 15)],
            REQUIRED_TARGETS[2]: [float(day) / 2 for day in range(1, 15)],
        },
    )
    long_df = build_canonical_long_format(raw_df)
    scope_slice = build_scope_slices(long_df, scope_name="global")[0]

    bundle = generate_report_bundle(
        scope_slice,
        target=REQUIRED_TARGETS[0],
        output_dir=tmp_path / "report_bundle_fit",
        run_parameters={"scope": "global", "target": REQUIRED_TARGETS[0]},
        correlation_methods=["spearman", "pearson"],
        correlation_min_shared_samples=4,
        readiness_kwargs={
            "min_target_observations": 6,
            "min_shared_samples": 4,
            "max_missing_ratio": 0.6,
            "heavy_censoring_ratio": 0.5,
            "min_eligible_predictors": 1,
        },
        modeling_kwargs={
            "model_names": ["bayesian_ridge"],
            "test_size": 0.3,
            "min_train_size": 6,
            "min_target_observations": 6,
            "min_shared_samples": 4,
            "max_missing_ratio": 0.6,
            "heavy_censoring_ratio": 0.5,
            "min_eligible_predictors": 1,
            "min_target_correlation": 0.1,
            "max_features": 5,
            "multicollinearity_threshold": 0.85,
        },
        reporting_kwargs={"plots_dpi": 100, "top_correlations": 5, "heatmap_max_features": 10},
    )

    assert (bundle.output_dir / "models" / "best_model_package" / "model_card.json").exists()
    assert not (bundle.output_dir / "models" / "model_package").exists()
    assert (bundle.output_dir / "tables" / "comparison_summary.xlsx").exists()
    assert (bundle.output_dir / "tables" / "holdout_predictions.xlsx").exists()
    summary_text = bundle.summary_path.read_text(encoding="utf-8")
    assert "Возможность расчетной оценки прогнозируемых значений" in summary_text
    assert "best_model_package" in summary_text
