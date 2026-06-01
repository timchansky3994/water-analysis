"""Orchestration wrappers around the water_analysis Python API.

These functions mirror the logic from cli.py::main for the 'report' and
'estimate-missing' commands so that both CLI and UI share the same flow.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd

from water_analysis.analysis.scopes import ScopeSlice, build_scope_slices
from water_analysis.config import load_config
from water_analysis.inference.engine import estimate_manual_values, estimate_missing_values
from water_analysis.inference.package import LoadedModelPackage, load_model_package
from water_analysis.inference.results import InferenceResult, save_inference_outputs
from water_analysis.io.source_profiles import (
    SourceProfile,
    autodetect_source_profile,
    list_available_profiles,
    load_source_profile,
)
from water_analysis.preprocessing.long_format import build_canonical_long_format, read_source_table
from water_analysis.reporting.bundle import ReportBundle, generate_report_bundle


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPORTS_ROOT = _PROJECT_ROOT / "reports"

LOGGER = logging.getLogger("water_analysis.streamlit.report")


_CYRILLIC_MAP = str.maketrans({
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
    'е': 'e', 'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'i',
    'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
    'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
    'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch',
    'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '',
    'э': 'e', 'ю': 'yu', 'я': 'ya',
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D',
    'Е': 'E', 'Ё': 'Yo', 'Ж': 'Zh', 'З': 'Z', 'И': 'I',
    'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N',
    'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T',
    'У': 'U', 'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch',
    'Ш': 'Sh', 'Щ': 'Sch', 'Ъ': '', 'Ы': 'Y', 'Ь': '',
    'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
})


def _safe_name(text: str) -> str:
    """Transliterate Cyrillic then strip characters unsafe for filesystem paths."""
    text = text.translate(_CYRILLIC_MAP)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


@contextmanager
def _capture_run_log(output_dir: Path, *, level: str = "INFO") -> Iterator[None]:
    """Attach a file handler that writes library logs to ``metadata/run.log``.

    The CLI writes this log via its own logging setup; the UI does not run the
    CLI, so without this the "Лог запуска" tab would always be empty for
    UI-generated bundles. Mirrors cli.py's ``metadata/run.log`` behaviour.

    The handler is attached to the ``water_analysis`` logger, NOT the root
    logger: the Streamlit entrypoint (``app.py``) sets
    ``logging.getLogger("water_analysis").propagate = False`` to keep library
    logs out of the terminal, so records never reach the root logger and a
    root-attached handler would capture nothing. Every library module and this
    module's ``LOGGER`` live under ``water_analysis.*``, so that logger is the
    common ancestor that still sees every record. Its effective level would
    otherwise be ``WARNING`` (inherited from the root), so we lower it for the
    duration to let INFO records through, then restore it.
    """
    log_path = output_dir / "metadata" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    wa_logger = logging.getLogger("water_analysis")
    previous_level = wa_logger.level
    if previous_level == logging.NOTSET or previous_level > handler.level:
        wa_logger.setLevel(handler.level)
    wa_logger.addHandler(handler)
    try:
        yield
    finally:
        handler.flush()
        wa_logger.removeHandler(handler)
        handler.close()
        wa_logger.setLevel(previous_level)


def resolve_source_profile(profile_arg: str, columns: Any) -> SourceProfile:
    """Resolve a profile name / 'auto' to a SourceProfile."""
    if profile_arg == "auto":
        detected = autodetect_source_profile(columns)
        if detected is not None:
            return detected
        return load_source_profile("_default")
    return load_source_profile(profile_arg)


def load_long_format(input_path: Path, source_profile: SourceProfile | None = None) -> pd.DataFrame:
    """Read and normalize a raw source file to canonical long format."""
    raw_df = read_source_table(input_path, source_profile=source_profile)
    return build_canonical_long_format(raw_df, source_profile=source_profile)


def load_raw_and_detect_profile(
    input_path: Path,
    *,
    profile_arg: str = "auto",
) -> tuple[pd.DataFrame, SourceProfile]:
    """Read raw file and autodetect (or load) the source profile.

    Returns (raw_df, profile).  Call build_canonical_long_format(raw_df, source_profile=profile)
    to get the long-format frame.
    """
    pre_profile: SourceProfile | None = None
    if profile_arg != "auto":
        pre_profile = load_source_profile(profile_arg)
    raw_df = read_source_table(input_path, source_profile=pre_profile)
    profile = resolve_source_profile(profile_arg, raw_df.columns)
    return raw_df, profile


def _default_output_dir(config: dict[str, Any], *, scope_id: str, target: str) -> Path:
    reporting = config.get("reporting", {})
    fmt = reporting.get("timestamp_format", "%Y%m%d_%H%M%S")
    root = _PROJECT_ROOT / reporting.get("reports_dir", "reports")
    ts = datetime.now().strftime(fmt)
    return root / f"streamlit_{ts}" / _safe_name(scope_id) / _safe_name(target)


def _get_nested(config: dict[str, Any], section: str, key: str, fallback: Any) -> Any:
    return config.get(section, {}).get(key, fallback)


def build_modeling_kwargs(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build modeling kwargs from config with optional UI overrides."""
    overrides = overrides or {}
    kwargs = {
        "test_size": overrides.get("test_size", _get_nested(config, "modeling", "test_size", 0.2)),
        "min_train_size": overrides.get("min_train_size", _get_nested(config, "modeling", "min_train_size", 20)),
        "min_target_observations": overrides.get("min_target_observations", _get_nested(config, "readiness", "min_target_observations", 30)),
        "min_shared_samples": overrides.get("min_shared_samples", _get_nested(config, "readiness", "min_shared_samples", 20)),
        "max_missing_ratio": overrides.get("max_missing_ratio", _get_nested(config, "readiness", "max_missing_ratio", 0.6)),
        "heavy_censoring_ratio": overrides.get("heavy_censoring_ratio", _get_nested(config, "readiness", "heavy_censoring_ratio", 0.5)),
        "min_eligible_predictors": overrides.get("min_eligible_predictors", _get_nested(config, "readiness", "min_eligible_predictors", 2)),
        "min_target_correlation": overrides.get("min_target_correlation", _get_nested(config, "modeling", "min_target_correlation", 0.3)),
        "significance_alpha": overrides.get("significance_alpha", _get_nested(config, "modeling", "significance_alpha", 0.05)),
        "max_features": overrides.get("max_features", _get_nested(config, "modeling", "max_features", 5)),
        "multicollinearity_threshold": overrides.get("multicollinearity_threshold", _get_nested(config, "modeling", "multicollinearity_threshold", 0.85)),
        "selection_mode": overrides.get("selection_mode", _get_nested(config, "modeling", "selection_mode", "auto")),
        "forced_features": overrides.get("forced_features", []),
        "combined_score_weight_holdout": overrides.get(
            "combined_score_weight_holdout",
            _get_nested(config, "modeling", "combined_score_weight_holdout", 0.4),
        ),
        "combined_score_weight_backtest": overrides.get(
            "combined_score_weight_backtest",
            _get_nested(config, "modeling", "combined_score_weight_backtest", 0.6),
        ),
        "seasonal_feature": overrides.get(
            "seasonal_feature",
            _get_nested(config, "modeling", "seasonal_feature", "none"),
        ),
    }
    return kwargs


def run_full_report(
    input_path: Path,
    *,
    scope_name: str,
    oktmo: str | None,
    point_type: str | None,
    point_code: str | None,
    target: str,
    output_dir: Path | None = None,
    source_profile: SourceProfile | None = None,
    model_names: list[str] | None = None,
    modeling_overrides: dict[str, Any] | None = None,
    on_progress: Callable[[str, float], None] | None = None,
    input_display_name: str | None = None,
    seasonality_granularity: str | None = None,
) -> ReportBundle:
    """Run the full report workflow and return a ReportBundle.

    This replicates cli.py's 'report' command logic so that both CLI and UI
    share the same orchestration path.
    """
    config = load_config()
    modeling_kwargs = build_modeling_kwargs(config, modeling_overrides)

    raw_df = read_source_table(input_path, source_profile=source_profile)
    if source_profile is None:
        source_profile = resolve_source_profile("auto", raw_df.columns)
    long_df = build_canonical_long_format(raw_df, source_profile=source_profile)

    scope_slices = build_scope_slices(
        long_df,
        scope_name=scope_name,
        oktmo=oktmo,
        point_type=point_type,
        point_code=point_code,
    )
    if not scope_slices:
        raise ValueError("Выбранные фильтры не соответствуют ни одному срезу данных.")
    if len(scope_slices) != 1:
        raise ValueError(
            f"Команда report требует ровно один срез, получено {len(scope_slices)}. "
            "Уточните ОКТМО, тип точки или код точки."
        )
    scope_slice: ScopeSlice = scope_slices[0]

    # Fill OKTMO / PointType / PointCode from the resolved scope when not provided by the caller.
    _sel = scope_slice.selector
    if oktmo is None:
        oktmo = _sel.get("OKTMO") or None
    if point_type is None:
        _pt = _sel.get("PointType_Code", "")
        point_type = _pt if _pt and _pt != "10110+10150" else None
    if point_code is None:
        point_code = _sel.get("FullPointCode") or None

    if output_dir is None:
        output_dir = _default_output_dir(config, scope_id=scope_slice.scope_id, target=target)

    correlation_methods = list(config.get("correlation", {}).get("methods", ["spearman", "pearson"]))
    reporting_config = config.get("reporting", {})

    seasonality_cfg = config.get("seasonality", {})
    _seasonality_granularity = seasonality_granularity or seasonality_cfg.get("analysis_granularity", "season")
    _seasonality_min_group = int(seasonality_cfg.get("min_group_size", 5))

    run_parameters: dict[str, Any] = {
        "input": input_display_name or str(input_path),
        "scope": scope_name,
        "scope_id": scope_slice.scope_id,
        "target": target,
        "oktmo": oktmo,
        "point_type": point_type,
        "point_code": point_code,
        "models": model_names or config.get("modeling", {}).get("default_models"),
        "source_profile": source_profile.name,
        "selection_mode": modeling_kwargs.get("selection_mode", "auto"),
        "seasonal_feature": modeling_kwargs.get("seasonal_feature", "none"),
    }

    with _capture_run_log(output_dir):
        LOGGER.info("Запуск отчёта через веб-интерфейс.")
        LOGGER.info("Источник данных: %s", run_parameters["input"])
        LOGGER.info("Профиль источника: %s", source_profile.name)
        LOGGER.info("Сценарий анализа: %s (срез %s).", scope_name, scope_slice.scope_id)
        LOGGER.info("Целевой показатель: %s", target)
        LOGGER.info(
            "Фильтры: ОКТМО=%s, тип точки=%s, код точки=%s.",
            oktmo or "—",
            point_type or "—",
            point_code or "—",
        )
        LOGGER.info("Модели: %s", ", ".join(run_parameters["models"] or []) or "по умолчанию")
        LOGGER.info("Режим отбора признаков: %s.", modeling_kwargs.get("selection_mode", "auto"))
        LOGGER.info("Размер среза: %s строк.", len(scope_slice.dataframe))
        LOGGER.info("Каталог результатов: %s", output_dir)
        bundle = generate_report_bundle(
            scope_slice,
            target=target,
            output_dir=output_dir,
            run_parameters=run_parameters,
            correlation_methods=correlation_methods,
            correlation_min_shared_samples=modeling_kwargs["min_shared_samples"],
            readiness_kwargs={
                "min_target_observations": modeling_kwargs["min_target_observations"],
                "min_shared_samples": modeling_kwargs["min_shared_samples"],
                "max_missing_ratio": modeling_kwargs["max_missing_ratio"],
                "heavy_censoring_ratio": modeling_kwargs["heavy_censoring_ratio"],
                "min_eligible_predictors": modeling_kwargs["min_eligible_predictors"],
            },
            modeling_kwargs={
                **modeling_kwargs,
                "model_names": model_names or config.get("modeling", {}).get("default_models"),
            },
            reporting_kwargs={
                "plots_dpi": reporting_config.get("plots_dpi", 150),
                "top_correlations": reporting_config.get("top_correlations", 10),
                "heatmap_max_features": reporting_config.get("heatmap_max_features", 20),
            },
            seasonality_granularity=_seasonality_granularity,
            seasonality_min_group_size=_seasonality_min_group,
            progress_callback=on_progress,
        )
        status = bundle.readiness_assessment.status if bundle.readiness_assessment else "—"
        LOGGER.info("Отчёт сформирован. Статус пригодности данных: %s.", status)
        return bundle


def run_estimate_missing(
    input_path: Path,
    package_dir: Path,
    *,
    source_profile: SourceProfile | None = None,
    min_observed_features: int = 2,
    min_feature_coverage: float = 0.5,
    allow_missing_feature_columns: bool = False,
    predict_all: bool = False,
    output_dir: Path | None = None,
    on_progress: Callable[[str, float], None] | None = None,
    input_display_name: str | None = None,
) -> tuple[InferenceResult, Path]:
    """Run estimate-missing and save outputs.

    Returns (InferenceResult, output_dir).
    """
    package: LoadedModelPackage = load_model_package(package_dir)

    raw_df = read_source_table(input_path, source_profile=source_profile)
    if source_profile is None:
        source_profile = resolve_source_profile("auto", raw_df.columns)

    result = estimate_missing_values(
        input_path,
        package,
        source_profile=source_profile,
        min_observed_features=min_observed_features,
        min_feature_coverage=min_feature_coverage,
        allow_missing_feature_columns=allow_missing_feature_columns,
        predict_all=predict_all,
        progress_callback=on_progress,
    )

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _REPORTS_ROOT / f"estimate_{ts}"

    save_inference_outputs(
        result,
        output_dir,
        model_card_payload=package.model_card.payload,
        run_parameters={
            "input": input_display_name or str(input_path),
            "model_package": str(package_dir),
            "target": package.model_card.target,
            "scope": package.model_card.scope_name,
            "source_profile": source_profile.name,
            "min_observed_features": min_observed_features,
            "min_feature_coverage": min_feature_coverage,
            "allow_missing_feature_columns": allow_missing_feature_columns,
            "predict_all": predict_all,
        },
    )

    return result, output_dir


def run_estimate_manual(
    package_dir: Path,
    samples: pd.DataFrame,
    *,
    min_observed_features: int = 1,
    min_feature_coverage: float = 0.0,
    output_dir: Path | None = None,
    on_progress: Callable[[str, float], None] | None = None,
) -> tuple[InferenceResult, Path]:
    """Estimate target values from manually entered indicator values and save outputs.

    The manual counterpart of :func:`run_estimate_missing`: it applies the same
    model package to values typed by the specialist (one sample per row of
    ``samples``) and writes the identical inference bundle to disk, so the UI
    can display and offer the results for download exactly like the file path.

    Returns (InferenceResult, output_dir).
    """
    package: LoadedModelPackage = load_model_package(package_dir)

    result = estimate_manual_values(
        package,
        samples,
        min_observed_features=min_observed_features,
        min_feature_coverage=min_feature_coverage,
        progress_callback=on_progress,
    )

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _REPORTS_ROOT / f"estimate_manual_{ts}"

    save_inference_outputs(
        result,
        output_dir,
        model_card_payload=package.model_card.payload,
        run_parameters={
            "input": "manual_entry",
            "model_package": str(package_dir),
            "target": package.model_card.target,
            "scope": package.model_card.scope_name,
            "rows_entered": int(len(samples)),
            "min_observed_features": min_observed_features,
            "min_feature_coverage": min_feature_coverage,
        },
    )

    return result, output_dir
