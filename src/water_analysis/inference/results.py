"""Structured inference outputs and markdown summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from water_analysis.reporting.xlsx_export import save_dataframe_csv_and_xlsx


@dataclass(frozen=True)
class InferenceResult:
    """Dataframes and counters produced by an estimation run."""

    predictions: pd.DataFrame
    estimated_values_long: pd.DataFrame
    diagnostics: pd.DataFrame
    summary: dict[str, Any]

    @property
    def rows_for_estimation(self) -> int:
        """Return number of rows considered for estimation."""
        return int(self.summary.get("rows_for_estimation", 0))

    @property
    def predicted_rows(self) -> int:
        """Return number of rows successfully estimated."""
        return int(self.summary.get("predicted_rows", 0))

    @property
    def skipped_rows(self) -> int:
        """Return number of rows skipped with diagnostics."""
        return int(self.summary.get("skipped_rows", 0))


_REASON_LABEL_RU: dict[str, str] = {
    "skipped_insufficient_features": "Слишком мало заполненных признаков в строке",
    "skipped_low_feature_coverage": "Доля заполненных признаков ниже порога",
    "missing_feature_column": "Признак модели отсутствует в файле",
    "missing_feature_columns_not_allowed": "Отсутствующие признаки не разрешены настройкой",
    "target_mismatch": "Несовпадение целевого показателя",
    "no_model_features_available": "Ни один признак модели не найден в файле",
    "target_listed_as_feature": "Целевой показатель ошибочно указан как признак",
    "empty_scope_after_pivot": "После фильтрации по сценарию данные пусты",
}


def build_inference_summary_markdown(result: InferenceResult) -> str:
    """Render a specialist-facing inference run summary."""
    summary = result.summary
    diagnostics = result.diagnostics
    missing_feature_counts = {}
    if not result.predictions.empty and "missing_features" in result.predictions:
        exploded = (
            result.predictions["missing_features"]
            .fillna("")
            .astype(str)
            .str.split("|")
            .explode()
        )
        exploded = exploded[exploded != ""]
        missing_feature_counts = exploded.value_counts().head(10).to_dict()

    skipped_by_reason = {}
    if not diagnostics.empty and "reason" in diagnostics:
        skipped_by_reason = diagnostics["reason"].value_counts().to_dict()

    lines = [
        "# Отчет о расчетной оценке прогнозируемых значений",
        "",
        "Расчетные значения в этом отчете являются аналитическими оценками модели. Они не являются лабораторными измерениями и не должны заменять исходные результаты анализа.",
        "",
        "## Что было сделано",
        f"- Выполнена расчетная оценка прогнозируемых значений показателя: `{summary.get('target')}`.",
        f"- Сценарий анализа: `{summary.get('scope_name')}`.",
        "",
        "## Использованная модель",
        f"- Пакет модели: `{summary.get('model_package')}`",
        f"- Модель: `{summary.get('model_name')}`",
        "",
        "## Результаты обработки",
        f"- Строк найдено для расчетной оценки: {summary.get('rows_for_estimation', 0)}",
        f"- Успешно оценено: {summary.get('predicted_rows', 0)}",
        f"- Пропущено: {summary.get('skipped_rows', 0)}",
    ]

    if skipped_by_reason:
        lines.extend(["", "## Почему строки могли быть пропущены"])
        for reason, count in skipped_by_reason.items():
            label = _REASON_LABEL_RU.get(str(reason), str(reason))
            lines.append(f"- {label}: {count} строк(и).")

    if missing_feature_counts:
        lines.extend(["", "## Часто отсутствующие признаки"])
        lines.extend(f"- `{feature}`: {count}" for feature, count in missing_feature_counts.items())

    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend(["", "## Предупреждения"])
        lines.extend(f"- {warning}" for warning in warnings)

    lines.extend(
        [
            "",
            "## Файлы результата",
            "- `predictions.csv` и `predictions.xlsx`: строки с расчетными оценками и диагностикой.",
            "- `estimated_values_long.csv` и `estimated_values_long.xlsx`: расчетные значения в long-format для дальнейшего анализа.",
            "- `inference_diagnostics.csv` и `inference_diagnostics.xlsx`: причины пропуска строк и проблемы совместимости.",
            "- `model_card_used.json`: карточка примененной модели.",
            "- `run_parameters.json`: параметры запуска.",
            "",
            "## Важное ограничение",
            "Расчетные значения помогают специалисту оценить отсутствующий показатель по сопутствующим измерениям, но не являются лабораторными измерениями.",
        ]
    )

    return "\n".join(lines).strip() + "\n"


def save_inference_outputs(
    result: InferenceResult,
    output_dir: str | Path,
    *,
    model_card_payload: dict[str, Any],
    run_parameters: dict[str, Any],
) -> dict[str, str]:
    """Save all inference outputs to a reproducible directory."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    predictions_saved = save_dataframe_csv_and_xlsx(result.predictions, root / "predictions.csv", table_name="predictions")
    saved["predictions"] = str(predictions_saved["csv"])
    saved["predictions_xlsx"] = str(predictions_saved["xlsx"])

    estimated_saved = save_dataframe_csv_and_xlsx(
        result.estimated_values_long,
        root / "estimated_values_long.csv",
        table_name="estimated_values_long",
    )
    saved["estimated_values_long"] = str(estimated_saved["csv"])
    saved["estimated_values_long_xlsx"] = str(estimated_saved["xlsx"])

    diagnostics_saved = save_dataframe_csv_and_xlsx(
        result.diagnostics,
        root / "inference_diagnostics.csv",
        table_name="inference_diagnostics",
    )
    saved["inference_diagnostics"] = str(diagnostics_saved["csv"])
    saved["inference_diagnostics_xlsx"] = str(diagnostics_saved["xlsx"])

    summary_path = root / "inference_summary.md"
    summary_path.write_text(build_inference_summary_markdown(result), encoding="utf-8")
    saved["inference_summary"] = str(summary_path)

    model_card_path = root / "model_card_used.json"
    model_card_path.write_text(json.dumps(model_card_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    saved["model_card_used"] = str(model_card_path)

    parameters_path = root / "run_parameters.json"
    parameters_path.write_text(json.dumps(run_parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    saved["run_parameters"] = str(parameters_path)

    return saved
