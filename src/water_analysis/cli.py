"""CLI for ingestion, profiling, readiness, and correlation analysis."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from water_analysis.analysis.correlations import run_correlation_analysis
from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.config import load_config
from water_analysis.io.schemas import REQUIRED_TARGETS, SCOPE_NAMES
from water_analysis.io.source_profiles import (
    SourceProfile,
    autodetect_source_profile,
    list_available_profiles,
    load_source_profile,
)
from water_analysis.inference.engine import (
    estimate_manual_values,
    estimate_missing_values,
    manual_input_feature_names,
)
from water_analysis.inference.package import load_model_package
from water_analysis.inference.results import save_inference_outputs
from water_analysis.inference.selector import select_model_from_catalog
from water_analysis.modeling.persistence import save_model_comparison_run
from water_analysis.modeling.registry import available_model_specs
from water_analysis.modeling.trainer import ModelingNotAllowedError, compare_models_in_scope, train_model_in_scope
from water_analysis.preprocessing.long_format import build_canonical_long_format, read_source_table
from water_analysis.preprocessing.pivot_builder import build_indicator_pivot
from water_analysis.profiling.passport import build_profile_reports
from water_analysis.profiling.readiness import assess_readiness
from water_analysis.reporting.bundle import default_report_output_dir, generate_report_bundle

LOGGER = logging.getLogger(__name__)


def _add_scope_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common scope selection arguments to a subcommand parser."""
    parser.add_argument("--scope", default="global", choices=SCOPE_NAMES, help="Analytical scope to evaluate.")
    parser.add_argument("--oktmo", default=None, help="Optional OKTMO filter for scoped analysis.")
    parser.add_argument("--point-type", default=None, help="Optional point type filter for scoped analysis.")
    parser.add_argument("--point-code", default=None, help="Optional full point code for point-level analysis.")


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add an optional config override path to a subcommand parser."""
    parser.add_argument("--config", default=None, help="Optional YAML config override.")


def _add_source_profile_argument(parser: argparse.ArgumentParser) -> None:
    """Add --source-profile argument to a subcommand parser."""
    parser.add_argument(
        "--source-profile",
        default="auto",
        help="Source data format profile name, file path, or 'auto' for autodetection.",
    )


def _resolve_source_profile(profile_arg: str, columns: Any) -> SourceProfile:
    """Resolve the --source-profile argument to a SourceProfile."""
    if profile_arg == "auto":
        detected = autodetect_source_profile(columns)
        if detected is not None:
            LOGGER.info("Source profile auto-detected: %s — %s", detected.name, detected.description)
            return detected
        LOGGER.info("No profile auto-detected; using default profile.")
        return load_source_profile("_default")
    try:
        profile = load_source_profile(profile_arg)
        LOGGER.info("Using source profile: %s — %s", profile.name, profile.description)
        return profile
    except FileNotFoundError:
        available = [p.name for p in list_available_profiles()]
        raise ValueError(f"Unknown source profile '{profile_arg}'. Available profiles: {available}")


def _setup_logging(
    log_level: str,
    log_file: str | None,
    *,
    auto_log_dir: Path | None = None,
) -> logging.FileHandler | None:
    """Configure logging with optional file output.

    Returns the FileHandler if one was created, so callers can set its path
    after the output directory is known (report command).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

    file_handler: logging.FileHandler | None = None
    log_path: Path | None = None

    if log_file:
        log_path = Path(log_file)
    elif auto_log_dir is not None:
        log_path = auto_log_dir / "metadata" / "run.log"

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logging.getLogger().addHandler(file_handler)
        LOGGER.info("Log file: %s", log_path)

    return file_handler


def _load_long_format(input_path: Path, source_profile: SourceProfile | None = None) -> pd.DataFrame:
    """Read and normalize a raw source table."""
    raw_df = read_source_table(input_path)
    return build_canonical_long_format(raw_df, source_profile=source_profile)


def _save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save a dataframe as UTF-8 CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")


def _save_json(payload: object, output_path: Path) -> None:
    """Save a JSON payload as UTF-8 text."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _cooccurrence_to_long(report_scope_name: str, report_scope_id: str, report_scope_label: str, matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert a co-occurrence matrix to a long dataframe for CSV export."""
    if matrix.empty:
        return pd.DataFrame(columns=["scope_name", "scope_id", "scope_label", "indicator_x", "indicator_y", "n_shared"])
    long_df = matrix.stack().reset_index()
    long_df.columns = ["indicator_x", "indicator_y", "n_shared"]
    long_df.insert(0, "scope_label", report_scope_label)
    long_df.insert(0, "scope_id", report_scope_id)
    long_df.insert(0, "scope_name", report_scope_name)
    return long_df


def _print_scope_count(scope_slices: list) -> None:
    """Log the number of built scope slices."""
    LOGGER.info("Built %s scope slice(s).", len(scope_slices))


def _safe_scope_dirname(scope_id: str) -> str:
    """Convert a scope id into a filesystem-safe directory name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", scope_id)


def _add_modeling_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common modeling arguments to a subcommand parser."""
    parser.add_argument("--target", required=True, help="Target indicator for modeling.")
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=None,
        choices=sorted(available_model_specs().keys()),
        help="ML model to include. Repeat to compare several models.",
    )
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--min-train-size", type=int, default=None)
    parser.add_argument("--min-target-observations", type=int, default=None)
    parser.add_argument("--min-shared-samples", type=int, default=None)
    parser.add_argument("--max-missing-ratio", type=float, default=None)
    parser.add_argument("--heavy-censoring-ratio", type=float, default=None)
    parser.add_argument("--min-eligible-predictors", type=int, default=None)
    parser.add_argument("--min-target-correlation", type=float, default=None)
    parser.add_argument("--significance-alpha", type=float, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--multicollinearity-threshold", type=float, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=["auto", "manual", "semi_auto"],
        default=None,
        help="Feature selection mode: auto (default), manual, or semi_auto.",
    )
    parser.add_argument(
        "--forced-feature",
        dest="forced_features",
        action="append",
        default=None,
        metavar="INDICATOR",
        help="Indicator name to force into the model. Repeat for multiple. Required for manual/semi_auto.",
    )
    parser.add_argument(
        "--combined-score-weight-holdout",
        type=float,
        default=None,
        help="Weight for holdout metric in combined score (default from config, typically 0.4).",
    )
    parser.add_argument(
        "--combined-score-weight-backtest",
        type=float,
        default=None,
        help="Weight for backtest metric in combined score (default from config, typically 0.6).",
    )
    parser.add_argument(
        "--seasonal-feature",
        choices=["none", "season", "month"],
        default=None,
        help="Optional seasonal feature added to the model after feature selection. Default: none.",
    )


def _resolve_reporting_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load merged configuration for a CLI run."""
    return load_config(getattr(args, "config", None))


def _get_nested(config: dict[str, Any], section: str, key: str, fallback: Any) -> Any:
    """Read a nested config key with fallback."""
    return config.get(section, {}).get(key, fallback)


def _resolve_modeling_kwargs(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Resolve modeling and readiness defaults from config plus CLI args."""
    selection_mode = (
        getattr(args, "selection_mode", None) or _get_nested(config, "modeling", "selection_mode", "auto")
    )
    forced_features_raw = getattr(args, "forced_features", None) or []

    # CLI validation: mode/forced consistency
    if selection_mode == "auto" and forced_features_raw:
        LOGGER.error(
            "--selection-mode=auto does not accept --forced-feature. "
            "Use --selection-mode=manual or --selection-mode=semi_auto."
        )
        raise SystemExit(2)
    if selection_mode in ("manual", "semi_auto") and not forced_features_raw:
        LOGGER.error(
            "--selection-mode=%s requires at least one --forced-feature argument.",
            selection_mode,
        )
        raise SystemExit(2)

    return {
        "test_size": args.test_size if args.test_size is not None else _get_nested(config, "modeling", "test_size", 0.2),
        "min_train_size": (
            args.min_train_size if args.min_train_size is not None else _get_nested(config, "modeling", "min_train_size", 20)
        ),
        "min_target_observations": (
            args.min_target_observations
            if args.min_target_observations is not None
            else _get_nested(config, "readiness", "min_target_observations", 30)
        ),
        "min_shared_samples": (
            args.min_shared_samples
            if args.min_shared_samples is not None
            else _get_nested(config, "readiness", "min_shared_samples", 20)
        ),
        "max_missing_ratio": (
            args.max_missing_ratio
            if args.max_missing_ratio is not None
            else _get_nested(config, "readiness", "max_missing_ratio", 0.6)
        ),
        "heavy_censoring_ratio": (
            args.heavy_censoring_ratio
            if args.heavy_censoring_ratio is not None
            else _get_nested(config, "readiness", "heavy_censoring_ratio", 0.5)
        ),
        "min_eligible_predictors": (
            args.min_eligible_predictors
            if args.min_eligible_predictors is not None
            else _get_nested(config, "readiness", "min_eligible_predictors", 2)
        ),
        "min_target_correlation": (
            args.min_target_correlation
            if args.min_target_correlation is not None
            else _get_nested(config, "modeling", "min_target_correlation", 0.3)
        ),
        "significance_alpha": (
            getattr(args, "significance_alpha", None)
            if getattr(args, "significance_alpha", None) is not None
            else _get_nested(config, "modeling", "significance_alpha", 0.05)
        ),
        "max_features": (
            args.max_features if args.max_features is not None else _get_nested(config, "modeling", "max_features", 5)
        ),
        "multicollinearity_threshold": (
            args.multicollinearity_threshold
            if args.multicollinearity_threshold is not None
            else _get_nested(config, "modeling", "multicollinearity_threshold", 0.85)
        ),
        "selection_mode": selection_mode,
        "forced_features": forced_features_raw,
        "combined_score_weight_holdout": (
            getattr(args, "combined_score_weight_holdout", None)
            if getattr(args, "combined_score_weight_holdout", None) is not None
            else _get_nested(config, "modeling", "combined_score_weight_holdout", 0.4)
        ),
        "combined_score_weight_backtest": (
            getattr(args, "combined_score_weight_backtest", None)
            if getattr(args, "combined_score_weight_backtest", None) is not None
            else _get_nested(config, "modeling", "combined_score_weight_backtest", 0.6)
        ),
        "seasonal_feature": (
            getattr(args, "seasonal_feature", None)
            or _get_nested(config, "modeling", "seasonal_feature", "none")
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the project CLI parser."""
    parser = argparse.ArgumentParser(prog="water-analysis")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to write the log file in addition to the console.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize_parser = subparsers.add_parser("normalize", help="Build canonical long format from a raw file.")
    normalize_parser.add_argument("--input", required=True, help="Path to a CSV or XLSX source file.")
    normalize_parser.add_argument("--output", required=True, help="Path to the output CSV file.")
    _add_source_profile_argument(normalize_parser)

    pivot_parser = subparsers.add_parser("pivot", help="Build an analysis pivot from a raw file.")
    pivot_parser.add_argument("--input", required=True, help="Path to a CSV or XLSX source file.")
    pivot_parser.add_argument("--output", required=True, help="Path to the output CSV file.")
    pivot_parser.add_argument(
        "--aggregation-level",
        default="sample_point_level",
        choices=["sample_point_level", "point_type_level", "oktmo_level"],
        help="Aggregation scope for the pivot output.",
    )
    _add_source_profile_argument(pivot_parser)

    profile_parser = subparsers.add_parser("profile", help="Build scope-aware dataset profile tables.")
    profile_parser.add_argument("--input", required=True, help="Path to a CSV or XLSX source file.")
    profile_parser.add_argument("--output-prefix", default=None, help="Optional prefix for exported profile tables.")
    _add_scope_arguments(profile_parser)
    _add_config_argument(profile_parser)
    _add_source_profile_argument(profile_parser)

    correlate_parser = subparsers.add_parser("correlate", help="Run scoped correlation analysis.")
    correlate_parser.add_argument("--input", required=True, help="Path to a CSV or XLSX source file.")
    correlate_parser.add_argument("--output-prefix", default=None, help="Optional prefix for exported correlation tables.")
    correlate_parser.add_argument(
        "--target",
        dest="targets",
        action="append",
        default=None,
        help="Target indicator to analyze. Repeat to analyze multiple targets. Defaults to required targets.",
    )
    correlate_parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        default=None,
        choices=["spearman", "pearson"],
        help="Correlation method. Repeat to run multiple methods. Defaults to spearman.",
    )
    correlate_parser.add_argument("--min-shared-samples", type=int, default=20)
    _add_scope_arguments(correlate_parser)
    _add_config_argument(correlate_parser)
    _add_source_profile_argument(correlate_parser)

    readiness_parser = subparsers.add_parser("readiness", help="Evaluate modeling suitability for scoped targets.")
    readiness_parser.add_argument("--input", required=True, help="Path to a CSV or XLSX source file.")
    readiness_parser.add_argument("--output", default=None, help="Optional CSV path for readiness results.")
    readiness_parser.add_argument(
        "--target",
        dest="targets",
        action="append",
        default=None,
        help="Target indicator to evaluate. Repeat to evaluate multiple targets. Defaults to required targets.",
    )
    readiness_parser.add_argument("--min-target-observations", type=int, default=30)
    readiness_parser.add_argument("--min-shared-samples", type=int, default=20)
    readiness_parser.add_argument("--max-missing-ratio", type=float, default=0.6)
    readiness_parser.add_argument("--heavy-censoring-ratio", type=float, default=0.5)
    readiness_parser.add_argument("--min-eligible-predictors", type=int, default=2)
    _add_scope_arguments(readiness_parser)
    _add_config_argument(readiness_parser)
    _add_source_profile_argument(readiness_parser)

    compare_models_parser = subparsers.add_parser("compare-models", help="Compare baselines and ML models.")
    compare_models_parser.add_argument("--input", required=True, help="Path to a CSV or XLSX source file.")
    compare_models_parser.add_argument("--output-dir", default=None, help="Optional output directory for artifacts.")
    _add_scope_arguments(compare_models_parser)
    _add_modeling_arguments(compare_models_parser)
    _add_config_argument(compare_models_parser)
    _add_source_profile_argument(compare_models_parser)

    train_parser = subparsers.add_parser("train", help="Train one model after honest comparison against baselines.")
    train_parser.add_argument("--input", required=True, help="Path to a CSV or XLSX source file.")
    train_parser.add_argument("--output-dir", required=True, help="Directory for model and evaluation artifacts.")
    _add_scope_arguments(train_parser)
    _add_modeling_arguments(train_parser)
    _add_config_argument(train_parser)
    _add_source_profile_argument(train_parser)

    report_parser = subparsers.add_parser("report", help="Run the end-to-end workflow and build a specialist report bundle.")
    report_parser.add_argument("--input", required=True, help="Path to a CSV or XLSX source file.")
    report_parser.add_argument("--output-dir", default=None, help="Optional report bundle directory.")
    report_parser.add_argument(
        "--seasonality-granularity",
        choices=["season", "month"],
        default=None,
        help="Granularity for the seasonal analysis section (season or month). Default from config.",
    )
    _add_scope_arguments(report_parser)
    _add_modeling_arguments(report_parser)
    _add_config_argument(report_parser)
    _add_source_profile_argument(report_parser)

    estimate_parser = subparsers.add_parser("estimate-missing", help="Estimate missing target values using a saved model package.")
    estimate_parser.add_argument("--input", required=True, help="Path to a new CSV or XLSX source file.")
    estimate_parser.add_argument("--model-package", default=None, help="Directory containing model.joblib and model_card.json.")
    estimate_parser.add_argument("--model-catalog", default=None, help="Optional catalog directory with model packages.")
    estimate_parser.add_argument("--target", default=None, help="Target indicator. Defaults to model_card.json target.")
    estimate_parser.add_argument("--scope", default=None, choices=SCOPE_NAMES, help="Inference scope. Defaults to model_card.json scope.")
    estimate_parser.add_argument("--oktmo", default=None, help="Optional OKTMO filter for inference.")
    estimate_parser.add_argument("--point-type", default=None, help="Optional point type filter for inference.")
    estimate_parser.add_argument("--point-code", default=None, help="Optional full point code for inference.")
    estimate_parser.add_argument("--output-dir", required=True, help="Directory for inference artifacts.")
    estimate_parser.add_argument("--min-observed-features", type=int, default=None)
    estimate_parser.add_argument("--min-feature-coverage", type=float, default=None)
    estimate_parser.add_argument("--allow-scope-fallback", action="store_true")
    estimate_parser.add_argument("--allow-missing-feature-columns", action="store_true")
    estimate_parser.add_argument("--predict-all", action="store_true")
    estimate_parser.add_argument("--format", default="csv", choices=["csv"])
    _add_config_argument(estimate_parser)
    _add_source_profile_argument(estimate_parser)

    manual_parser = subparsers.add_parser(
        "estimate-manual",
        help="Estimate a target value from manually entered indicator values (no input file).",
    )
    manual_parser.add_argument("--model-package", default=None, help="Directory containing model.joblib and model_card.json.")
    manual_parser.add_argument("--model-catalog", default=None, help="Optional catalog directory with model packages.")
    manual_parser.add_argument("--target", default=None, help="Target indicator. Defaults to model_card.json target.")
    manual_parser.add_argument("--scope", default=None, choices=SCOPE_NAMES, help="Scope for catalog selection. Defaults to model_card.json scope.")
    manual_parser.add_argument(
        "--value",
        dest="values",
        action="append",
        default=[],
        metavar="INDICATOR=VALUE",
        help='Indicator value, repeatable. Example: --value "Цветность=12". Censored text like "<0.05" is allowed.',
    )
    manual_parser.add_argument(
        "--date",
        default=None,
        help="Sample date (YYYY-MM-DD or DD.MM.YYYY). Used for seasonal models; defaults to today if omitted.",
    )
    manual_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for the full inference bundle. If omitted, the result is only printed.",
    )
    manual_parser.add_argument("--min-observed-features", type=int, default=1)
    manual_parser.add_argument("--min-feature-coverage", type=float, default=0.0)
    manual_parser.add_argument(
        "--list-features",
        action="store_true",
        help="List the indicator values the selected model expects, then exit.",
    )
    _add_config_argument(manual_parser)

    subparsers.add_parser("ui", help="Launch the Streamlit web interface.")

    return parser


def _cmd_ui() -> int:
    """Launch the Streamlit web interface and print a console hint when ready."""
    import subprocess
    import sys
    import threading
    import time
    import urllib.request

    app_path = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"
    if not app_path.exists():
        print(f"Streamlit app not found: {app_path}", file=sys.stderr)
        return 1

    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(app_path)])

    def _wait_and_print_hint() -> None:
        for _ in range(60):
            time.sleep(0.5)
            try:
                urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=1)
                print("\n  Нажмите Ctrl+C в этом окне, чтобы завершить приложение принудительно.\n")
                return
            except Exception:
                pass

    threading.Thread(target=_wait_and_print_hint, daemon=True).start()

    try:
        process.wait()
    except KeyboardInterrupt:
        pass

    return process.returncode or 0


def _parse_value_pairs(pairs: list[str]) -> dict[str, str]:
    """Parse repeated ``INDICATOR=VALUE`` CLI arguments into a mapping."""
    parsed: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid --value '{item}'. Expected INDICATOR=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --value '{item}': empty indicator name.")
        parsed[key] = value.strip()
    return parsed


def _log_manual_predictions(result: "Any", target: str) -> None:
    """Log a concise, specialist-facing summary of manual estimation results."""
    predictions = result.predictions
    if predictions.empty:
        LOGGER.warning("Оценка не получена. См. диагностику ниже.")
        if not result.diagnostics.empty and "reason" in result.diagnostics.columns:
            for reason in result.diagnostics["reason"].astype(str).tolist():
                LOGGER.warning("  Причина: %s", reason)
        return

    for row in predictions.itertuples(index=False):
        status = getattr(row, "prediction_status", "")
        sample_date = getattr(row, "SampleDate", "")
        if status == "estimated":
            value = getattr(row, "predicted_value", float("nan"))
            lower = getattr(row, "lower_bound", float("nan"))
            upper = getattr(row, "upper_bound", float("nan"))
            coverage = getattr(row, "feature_coverage", float("nan"))
            coverage_pct = 0.0 if pd.isna(coverage) else float(coverage) * 100
            if pd.isna(lower) or pd.isna(upper):
                LOGGER.info(
                    "Оценка «%s» (%s): %.4g (интервал недоступен), покрытие признаков %.0f%%.",
                    target, sample_date, value, coverage_pct,
                )
            else:
                LOGGER.info(
                    "Оценка «%s» (%s): %.4g [%.4g; %.4g], покрытие признаков %.0f%%.",
                    target, sample_date, value, lower, upper, coverage_pct,
                )
        else:
            LOGGER.warning(
                "Строка (%s) не оценена (статус: %s). %s",
                sample_date, status, getattr(row, "warnings", "") or "",
            )


def _cmd_estimate_manual(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Estimate a target value from manually entered indicator values."""
    package = None
    if args.model_package:
        package = load_model_package(args.model_package)
    elif args.model_catalog:
        if not args.target or not args.scope:
            LOGGER.error("--target and --scope are required when selecting from --model-catalog.")
            return 2
        package = select_model_from_catalog(args.model_catalog, target=args.target, scope_name=args.scope)
        if package is None:
            LOGGER.error("No compatible model package found in catalog: %s", args.model_catalog)
            return 2
    else:
        LOGGER.error("Provide either --model-package or --model-catalog.")
        return 2

    card = package.model_card
    expected_features = manual_input_feature_names(card)

    if args.list_features:
        LOGGER.info("Целевой показатель модели: %s", card.target)
        LOGGER.info("Сценарий модели: %s", card.scope_name)
        if expected_features:
            LOGGER.info("Ожидаемые показатели для ручного ввода (%d):", len(expected_features))
            for feature in expected_features:
                LOGGER.info("  - %s", feature)
        else:
            LOGGER.info("Модель использует только сезонные признаки — укажите только --date.")
        if card.seasonal_feature != "none":
            LOGGER.info("Модель использует сезонные признаки — обязательно укажите --date.")
        return 0

    try:
        values = _parse_value_pairs(args.values)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

    expected_set = set(expected_features)
    unknown = [name for name in values if name not in expected_set]
    if unknown:
        LOGGER.warning(
            "Показатели не входят в модель и будут проигнорированы: %s. Ожидаются: %s",
            ", ".join(unknown),
            ", ".join(expected_features) or "(нет; модель сезонная)",
        )
    if not values and expected_features:
        LOGGER.error(
            "Не задано ни одного значения показателя. Используйте --value \"Показатель=Значение\" "
            "(список показателей: --list-features)."
        )
        return 2

    sample_date = args.date or pd.Timestamp.today().strftime("%Y-%m-%d")
    if args.date is None and card.seasonal_feature != "none":
        LOGGER.warning(
            "Дата не указана (--date); для сезонных признаков используется сегодняшняя дата %s.",
            sample_date,
        )

    sample = {"SampleDate": sample_date, **values}
    samples_df = pd.DataFrame([sample])

    result = estimate_manual_values(
        package,
        samples_df,
        min_observed_features=args.min_observed_features,
        min_feature_coverage=args.min_feature_coverage,
    )

    LOGGER.info(
        "Ручная оценка завершена. Строк: %s; успешно оценено: %s; пропущено: %s.",
        result.rows_for_estimation,
        result.predicted_rows,
        result.skipped_rows,
    )
    _log_manual_predictions(result, card.target)

    if args.output_dir:
        saved = save_inference_outputs(
            result,
            args.output_dir,
            model_card_payload=card.payload,
            run_parameters={
                "input": "manual_entry",
                "model_package": str(package.package_dir),
                "model_catalog": args.model_catalog,
                "target": card.target,
                "scope": card.scope_name,
                "date": sample_date,
                "values": values,
                "min_observed_features": args.min_observed_features,
                "min_feature_coverage": args.min_feature_coverage,
            },
        )
        LOGGER.info("Расчёты: %s и %s", saved.get("predictions"), saved.get("predictions_xlsx"))
        LOGGER.info("Расчётные значения (long): %s", saved.get("estimated_values_long"))
        LOGGER.info("Краткий отчёт: %s", saved.get("inference_summary"))

    return 0


def main() -> int:
    """Run the project CLI."""
    args = build_parser().parse_args()

    if args.command == "ui":
        return _cmd_ui()

    if args.command == "estimate-manual":
        _setup_logging(args.log_level, args.log_file)
        return _cmd_estimate_manual(args, _resolve_reporting_config(args))

    # For the report command with an explicit --output-dir, start writing the
    # log file immediately so that source-profile resolution and data-loading
    # warnings are captured.  When --output-dir is omitted (auto-generated from
    # scope) we attach the file handler later, after the output dir is known.
    early_report_log: bool = False
    if args.command == "report" and getattr(args, "output_dir", None) and args.log_file is None:
        _setup_logging(args.log_level, args.log_file, auto_log_dir=Path(args.output_dir))
        early_report_log = True
    else:
        _setup_logging(args.log_level, args.log_file)

    config = _resolve_reporting_config(args)

    input_path = Path(args.input)
    output_path = Path(args.output) if hasattr(args, "output") and args.output else None

    if args.command == "estimate-missing":
        inference_config = config.get("inference", {})
        package = None
        if args.model_package:
            package = load_model_package(args.model_package)
        elif args.model_catalog:
            if not args.target or not args.scope:
                LOGGER.error("--target and --scope are required when selecting from --model-catalog.")
                return 2
            package = select_model_from_catalog(args.model_catalog, target=args.target, scope_name=args.scope)
            if package is None:
                LOGGER.error("No compatible model package found in catalog: %s", args.model_catalog)
                return 2
        else:
            LOGGER.error("Provide either --model-package or --model-catalog.")
            return 2

        pre_profile_est = None
        if getattr(args, "source_profile", "auto") != "auto":
            try:
                pre_profile_est = load_source_profile(args.source_profile)
            except FileNotFoundError:
                available = [p.name for p in list_available_profiles()]
                LOGGER.error(
                    "Unknown source profile '%s'. Available profiles: %s",
                    args.source_profile,
                    available,
                )
                return 2
        raw_df = read_source_table(input_path, source_profile=pre_profile_est)
        source_profile = _resolve_source_profile(args.source_profile, raw_df.columns)

        try:
            result = estimate_missing_values(
                input_path,
                package,
                target=args.target,
                scope_name=args.scope,
                oktmo=args.oktmo,
                point_type=args.point_type,
                point_code=args.point_code,
                min_observed_features=(
                    args.min_observed_features
                    if args.min_observed_features is not None
                    else int(inference_config.get("min_observed_features", 2))
                ),
                min_feature_coverage=(
                    args.min_feature_coverage
                    if args.min_feature_coverage is not None
                    else float(inference_config.get("min_feature_coverage", 0.5))
                ),
                allow_scope_fallback=args.allow_scope_fallback,
                allow_missing_feature_columns=args.allow_missing_feature_columns,
                predict_all=args.predict_all,
                source_profile=source_profile,
            )
        except Exception:
            LOGGER.exception("estimate-missing failed with an unhandled exception.")
            raise

        saved = save_inference_outputs(
            result,
            args.output_dir,
            model_card_payload=package.model_card.payload,
            run_parameters={
                "input": str(input_path),
                "model_package": str(package.package_dir),
                "model_catalog": args.model_catalog,
                "target": args.target or package.model_card.target,
                "scope": args.scope or package.model_card.scope_name,
                "oktmo": args.oktmo,
                "point_type": args.point_type,
                "point_code": args.point_code,
                "min_observed_features": args.min_observed_features,
                "min_feature_coverage": args.min_feature_coverage,
                "allow_scope_fallback": args.allow_scope_fallback,
                "allow_missing_feature_columns": args.allow_missing_feature_columns,
                "predict_all": args.predict_all,
                "format": args.format,
                "source_profile": source_profile.name,
            },
        )
        LOGGER.info(
            "Расчетная оценка прогнозируемых значений завершена. Строк найдено: %s; успешно оценено: %s; пропущено: %s.",
            result.rows_for_estimation,
            result.predicted_rows,
            result.skipped_rows,
        )
        LOGGER.info("Пакет модели: %s", package.package_dir)
        LOGGER.info("Расчеты: %s и %s", saved.get("predictions"), saved.get("predictions_xlsx"))
        LOGGER.info("Диагностика: %s и %s", saved.get("inference_diagnostics"), saved.get("inference_diagnostics_xlsx"))
        LOGGER.info("Краткий отчет: %s", saved.get("inference_summary"))
        return 0

    LOGGER.info("Reading source file: %s", input_path)
    # If the profile is given explicitly, resolve it before reading so that the
    # xlsx reader can use it for sheet/header detection right away.
    pre_profile = None
    if getattr(args, "source_profile", "auto") != "auto":
        try:
            pre_profile = load_source_profile(args.source_profile)
        except FileNotFoundError:
            available = [p.name for p in list_available_profiles()]
            LOGGER.error(
                "Unknown source profile '%s'. Available profiles: %s",
                args.source_profile,
                available,
            )
            return 2
    raw_df = read_source_table(input_path, source_profile=pre_profile)
    source_profile = _resolve_source_profile(args.source_profile, raw_df.columns)
    long_df = build_canonical_long_format(raw_df, source_profile=source_profile)
    LOGGER.info("Normalized %d rows from %s.", len(long_df), input_path)

    if args.command == "normalize":
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        LOGGER.info("Saved canonical long format: %s (%s rows)", output_path, len(long_df))
        return 0

    if args.command == "pivot":
        pivot_output = Path(args.output)
        pivot_df = build_indicator_pivot(long_df, aggregation_level=args.aggregation_level)
        pivot_output.parent.mkdir(parents=True, exist_ok=True)
        pivot_df.to_csv(pivot_output, index=False, encoding="utf-8-sig")
        LOGGER.info(
            "Saved pivot: %s (%s rows, aggregation=%s)",
            pivot_output,
            len(pivot_df),
            args.aggregation_level,
        )
        return 0

    scope_slices = build_scope_slices(
        long_df,
        scope_name=args.scope,
        oktmo=getattr(args, "oktmo", None),
        point_type=getattr(args, "point_type", None),
        point_code=getattr(args, "point_code", None),
    )
    _print_scope_count(scope_slices)
    if not scope_slices:
        LOGGER.warning("No scope slices matched the provided filters.")
        return 0

    if args.command == "profile":
        reports = build_profile_reports(scope_slices)
        if args.output_prefix:
            output_prefix = Path(args.output_prefix)
            summary_df = pd.concat([report.summary_frame() for report in reports], ignore_index=True)
            indicator_df = pd.concat([report.indicator_observations for report in reports], ignore_index=True)
            missingness_df = pd.concat([report.missingness for report in reports], ignore_index=True)
            point_type_df = pd.concat([report.point_type_coverage for report in reports], ignore_index=True)
            constant_df = pd.concat([report.constant_series for report in reports], ignore_index=True)
            cooccurrence_df = pd.concat(
                [
                    _cooccurrence_to_long(report.scope_name, report.scope_id, report.scope_label, report.cooccurrence_matrix)
                    for report in reports
                ],
                ignore_index=True,
            )
            _save_dataframe(summary_df, output_prefix.with_name(f"{output_prefix.name}_summary.csv"))
            _save_json(summary_df.to_dict(orient="records"), output_prefix.with_name(f"{output_prefix.name}_summary.json"))
            _save_dataframe(indicator_df, output_prefix.with_name(f"{output_prefix.name}_indicator_observations.csv"))
            _save_dataframe(missingness_df, output_prefix.with_name(f"{output_prefix.name}_missingness.csv"))
            _save_dataframe(point_type_df, output_prefix.with_name(f"{output_prefix.name}_point_type_coverage.csv"))
            _save_dataframe(constant_df, output_prefix.with_name(f"{output_prefix.name}_constant_series.csv"))
            _save_dataframe(cooccurrence_df, output_prefix.with_name(f"{output_prefix.name}_cooccurrence.csv"))
            LOGGER.info("Saved profile tables with prefix: %s", output_prefix)
        else:
            for report in reports:
                LOGGER.info("Profile %s: %s", report.scope_id, report.summary)
        return 0

    if args.command == "correlate":
        methods = tuple(args.methods) if args.methods else ("spearman",)
        targets = args.targets if args.targets else list(REQUIRED_TARGETS)
        analysis = run_correlation_analysis(
            scope_slices,
            targets=targets,
            methods=methods,
            min_shared_samples=args.min_shared_samples,
        )
        if args.output_prefix:
            output_prefix = Path(args.output_prefix)
            _save_dataframe(analysis.results, output_prefix.with_name(f"{output_prefix.name}_results.csv"))
            _save_dataframe(analysis.diagnostics, output_prefix.with_name(f"{output_prefix.name}_diagnostics.csv"))
            LOGGER.info("Saved correlation outputs with prefix: %s", output_prefix)
        else:
            if analysis.results.empty:
                LOGGER.info("No correlation results produced.")
            else:
                LOGGER.info("Top correlation rows:\n%s", analysis.results.head(10).to_string(index=False))
            if not analysis.diagnostics.empty:
                LOGGER.info("Diagnostics:\n%s", analysis.diagnostics.to_string(index=False))
        return 0

    if args.command == "compare-models":
        modeling_kwargs = _resolve_modeling_kwargs(args, config)
        selected_models = args.models if args.models else config.get("modeling", {}).get("default_models")
        all_comparison_frames: list[pd.DataFrame] = []
        for scope_slice in scope_slices:
            try:
                run = compare_models_in_scope(
                    scope_slice,
                    target=args.target,
                    model_names=selected_models,
                    **modeling_kwargs,
                )
            except ModelingNotAllowedError as error:
                LOGGER.warning("%s", error)
                continue

            scope_df = run.comparison_df.copy()
            scope_df.insert(0, "scope_name", run.scope_name)
            scope_df.insert(1, "scope_id", run.scope_id)
            scope_df.insert(2, "target", run.target)
            all_comparison_frames.append(scope_df)

            if args.output_dir:
                scope_output_dir = Path(args.output_dir) / _safe_scope_dirname(run.scope_id)
                saved = save_model_comparison_run(run, scope_output_dir)
                LOGGER.info("Saved comparison artifacts for %s: %s", run.scope_id, saved)
            else:
                LOGGER.info("Comparison for %s:\n%s", run.scope_id, scope_df.to_string(index=False))

        if all_comparison_frames and not args.output_dir:
            summary_df = pd.concat(all_comparison_frames, ignore_index=True)
            LOGGER.info("Combined comparison summary:\n%s", summary_df.to_string(index=False))
        return 0

    if args.command == "train":
        modeling_kwargs = _resolve_modeling_kwargs(args, config)
        if len(scope_slices) != 1:
            raise ValueError("The train command requires filters that resolve to exactly one scope slice.")
        chosen_model_name = None
        if args.models:
            if len(args.models) > 1:
                raise ValueError("The train command accepts at most one --model value.")
            chosen_model_name = args.models[0]

        try:
            run, chosen_model = train_model_in_scope(
                scope_slices[0],
                target=args.target,
                model_name=chosen_model_name,
                **modeling_kwargs,
            )
        except ModelingNotAllowedError as error:
            LOGGER.error("%s", error)
            return 2
        LOGGER.info(
            "Readiness status: %s. Feature selection chose: %s",
            "allowed",
            getattr(run, "selected_features", "N/A"),
        )
        saved = save_model_comparison_run(run, Path(args.output_dir), chosen_model_name=chosen_model.model_name)
        LOGGER.info("Модель обучена: %s. Победитель сравнения: %s.", chosen_model.model_name, chosen_model.model_name)
        LOGGER.info("Артефакты сохранены: %s", saved)
        if "deployable_model_package" in saved:
            LOGGER.info("Пакет для estimate-missing: %s", saved["deployable_model_package"])
        return 0

    if args.command == "report":
        modeling_kwargs = _resolve_modeling_kwargs(args, config)
        correlation_methods = config.get("correlation", {}).get("methods", ["spearman", "pearson"])
        correlation_min_shared = modeling_kwargs["min_shared_samples"]
        reporting_config = config.get("reporting", {})
        seasonality_cfg = config.get("seasonality", {})
        seasonality_granularity = (
            getattr(args, "seasonality_granularity", None)
            or seasonality_cfg.get("analysis_granularity", "season")
        )
        if len(scope_slices) != 1:
            raise ValueError("The report command requires filters that resolve to exactly one scope slice.")
        scope_slice = scope_slices[0]
        report_output_dir = Path(args.output_dir) if args.output_dir else default_report_output_dir(
            config,
            scope_id=scope_slice.scope_id,
            target=args.target,
        )

        # Attach file logging now only if we couldn't do it early (auto output dir).
        if args.log_file is None and not early_report_log:
            log_path = report_output_dir / "metadata" / "run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )
            logging.getLogger().addHandler(file_handler)
            LOGGER.info("Log file: %s", log_path)

        LOGGER.info("Report output directory: %s", report_output_dir)
        LOGGER.info("Scope: %s (%s rows).", scope_slice.scope_id, len(scope_slice.dataframe))

        run_parameters = {
            "input": str(input_path),
            "scope": args.scope,
            "scope_id": scope_slice.scope_id,
            "target": args.target,
            "oktmo": args.oktmo,
            "point_type": args.point_type,
            "point_code": args.point_code,
            "models": args.models if args.models else config.get("modeling", {}).get("default_models"),
            "source_profile": source_profile.name,
            "seasonal_feature": modeling_kwargs.get("seasonal_feature", "none"),
        }

        try:
            bundle = generate_report_bundle(
                scope_slice,
                target=args.target,
                output_dir=report_output_dir,
                run_parameters=run_parameters,
                correlation_methods=list(correlation_methods),
                correlation_min_shared_samples=correlation_min_shared,
                readiness_kwargs={
                    "min_target_observations": modeling_kwargs["min_target_observations"],
                    "min_shared_samples": modeling_kwargs["min_shared_samples"],
                    "max_missing_ratio": modeling_kwargs["max_missing_ratio"],
                    "heavy_censoring_ratio": modeling_kwargs["heavy_censoring_ratio"],
                    "min_eligible_predictors": modeling_kwargs["min_eligible_predictors"],
                },
                modeling_kwargs={
                    **modeling_kwargs,
                    "model_names": args.models if args.models else config.get("modeling", {}).get("default_models"),
                },
                reporting_kwargs={
                    "plots_dpi": reporting_config.get("plots_dpi", 150),
                    "top_correlations": reporting_config.get("top_correlations", 10),
                    "heatmap_max_features": reporting_config.get("heatmap_max_features", 20),
                },
                seasonality_granularity=seasonality_granularity,
                seasonality_min_group_size=int(seasonality_cfg.get("min_group_size", 5)),
            )
        except Exception:
            LOGGER.exception("report command failed with an unhandled exception.")
            raise

        LOGGER.info("Отчет сформирован: %s", bundle.output_dir)
        LOGGER.info("Краткий отчет для специалиста: %s", bundle.summary_path)
        LOGGER.info("Таблицы CSV/XLSX: %s", bundle.output_dir / "tables")
        LOGGER.info("Графики: %s", bundle.output_dir / "plots")
        if "deployable_model_package" in bundle.generated_files:
            LOGGER.info("Пакет модели для estimate-missing: %s", bundle.generated_files["deployable_model_package"])
        else:
            LOGGER.info("Пакет модели best_model_package не создан; причина указана в specialist_summary.md.")
        return 0

    targets = args.targets if args.targets else list(REQUIRED_TARGETS)
    assessments = assess_readiness(
        scope_slices,
        targets=targets,
        min_target_observations=args.min_target_observations,
        min_shared_samples=args.min_shared_samples,
        max_missing_ratio=args.max_missing_ratio,
        heavy_censoring_ratio=args.heavy_censoring_ratio,
        min_eligible_predictors=args.min_eligible_predictors,
    )
    for assessment in assessments:
        LOGGER.info(
            "Readiness %s / %s: %s — issues: %s",
            assessment.scope_id,
            assessment.target,
            assessment.status,
            assessment.issue_codes,
        )
    readiness_df = pd.DataFrame([assessment.to_record() for assessment in assessments])
    if output_path:
        _save_dataframe(readiness_df, output_path)
        LOGGER.info("Saved readiness report: %s", output_path)
    else:
        LOGGER.info("Readiness results:\n%s", readiness_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
