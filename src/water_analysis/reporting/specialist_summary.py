"""Human-readable specialist-facing summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from water_analysis.analysis.correlations import CorrelationAnalysis
from water_analysis.analysis.feature_selection import diagnose_no_predictors
from water_analysis.modeling.trainer import ModelComparisonRun
from water_analysis.profiling.passport import ProfileReport
from water_analysis.profiling.readiness import ReadinessAssessment

if TYPE_CHECKING:
    from water_analysis.analysis.seasonality import SeasonalityAnalysis


READINESS_STATUS_RU = {
    "suitable": "данные пригодны для моделирования",
    "weakly_suitable": "данные ограниченно пригодны, результаты нужно трактовать осторожно",
    "unsuitable": "данные не пригодны для обучения модели в этом срезе",
}

READINESS_REASON_RU = {
    "target_unavailable": "целевой показатель отсутствует или не имеет числовых значений",
    "too_few_observations": "слишком мало наблюдений целевого показателя",
    "limited_observations": "наблюдений меньше рекомендуемого уровня",
    "target_constant": "целевой показатель постоянный",
    "target_near_constant": "целевой показатель почти не меняется",
    "extreme_missingness": "слишком высокая доля пропусков",
    "high_missingness": "высокая доля пропусков",
    "heavy_censoring": "много цензурированных значений, например ниже предела обнаружения",
    "low_shared_measurements": "мало совместных измерений целевого и сопутствующих показателей",
    "weak_predictor_availability": "мало доступных сопутствующих показателей",
}


@dataclass(frozen=True)
class SpecialistSummaryInput:
    """Inputs needed for rendering a specialist-facing summary."""

    run_parameters: dict[str, Any]
    profile_report: ProfileReport
    readiness_assessment: ReadinessAssessment
    correlation_analysis: CorrelationAnalysis
    model_run: ModelComparisonRun | None
    deployable_model_package: str | None = None
    seasonality_analysis: "SeasonalityAnalysis | None" = None


def _format_value(value: object) -> str:
    """Format a value for markdown."""
    if value is None:
        return "нет данных"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value) if value else "нет данных"
    if pd.isna(value):
        return "нет данных"
    return str(value)


def _format_run_parameters(run_parameters: dict[str, Any]) -> str:
    """Render run parameters for a specialist."""
    if not run_parameters:
        return "Параметры запуска не сохранены."
    labels = {
        "input": "Входной файл",
        "scope": "Сценарий анализа",
        "scope_id": "Идентификатор среза",
        "target": "Целевой показатель",
        "oktmo": "ОКТМО",
        "point_type": "Тип точки",
        "point_code": "Полный код точки",
        "models": "Проверенные ML-модели",
        "selection_mode": "Режим отбора предикторов",
    }
    return "\n".join(f"- {labels.get(key, key)}: `{_format_value(value)}`" for key, value in run_parameters.items())


def _format_issues(readiness_assessment: ReadinessAssessment) -> str:
    """Render readiness issues as Russian markdown."""
    if not readiness_assessment.issues:
        return "- Критичных ограничений не выявлено."
    lines = []
    for issue in readiness_assessment.issues:
        explanation = READINESS_REASON_RU.get(issue.code, issue.message)
        severity = "критично" if issue.severity == "critical" else "предупреждение"
        lines.append(f"- {severity}: {explanation} (`{issue.code}`). {issue.message}")
    return "\n".join(lines)


def _format_correlations(correlation_analysis: CorrelationAnalysis, target: str, limit: int) -> str:
    """Render top correlations as markdown lines."""
    if correlation_analysis.results.empty:
        return "Подходящие пары показателей для расчета корреляции не найдены."

    target_results = correlation_analysis.results[correlation_analysis.results["target"] == target].copy()
    if target_results.empty:
        return "Для выбранного целевого показателя корреляции не рассчитаны."

    top_rows = (
        target_results.assign(corr_abs=target_results["corr"].abs())
        .sort_values(["corr_abs", "n_shared"], ascending=[False, False])
        .head(limit)
    )
    lines = []
    for row in top_rows.itertuples(index=False):
        lines.append(
            f"- `{row.feature}`: метод `{row.method}`, корреляция={row.corr:.3f}, "
            f"совместных измерений={row.n_shared}, p-value={row.p_value:.3g}"
        )
    return "\n".join(lines)


def _format_seasonality_section(analysis: "SeasonalityAnalysis | None") -> str:
    """Render the seasonality diagnostic section."""
    if analysis is None:
        return "Сезонный анализ не выполнялся."

    lines: list[str] = []

    test = analysis.pattern_test
    if test.get("test") == "kruskal_wallis":
        p_val = test.get("p_value", float("nan"))
        stat = test.get("statistic", float("nan"))
        n_groups = test.get("groups_used", "?")
        if analysis.seasonal_pattern_detected:
            lines.append(
                f"**Вывод:** сезонный паттерн **обнаружен** "
                f"(критерий Краскела–Уоллиса, p={p_val:.3g}, статистика={stat:.2f}, "
                f"групп с достаточными данными: {n_groups})."
            )
        else:
            lines.append(
                f"**Вывод:** статистически значимый сезонный паттерн **не обнаружен** "
                f"(p={p_val:.3g}, статистика={stat:.2f}, групп: {n_groups}). "
                f"Это может означать отсутствие сезонности или недостаточно данных для её выявления."
            )
    else:
        reason = test.get("reason", "неизвестно")
        lines.append(f"**Вывод:** тест на сезонность пропущен — {reason}.")
        lines.append("Вывод о сезонности для данного среза ненадёжен — слишком мало наблюдений по группам.")

    granularity_label = "сезону" if analysis.granularity == "season" else "месяцу"

    if not analysis.group_stats.empty and "median" in analysis.group_stats.columns:
        lines.append("")
        lines.append(f"Распределение по {granularity_label}:")
        stats = analysis.group_stats.sort_values("median", ascending=False)
        top = stats.iloc[0]
        bot = stats.iloc[-1]
        lines.append(
            f"- Наиболее высокие значения: группа **{top['group']}** (медиана {top['median']:.3g}, n={int(top['n_observations'])})."
        )
        lines.append(
            f"- Наиболее низкие значения: группа **{bot['group']}** (медиана {bot['median']:.3g}, n={int(bot['n_observations'])})."
        )

    if not analysis.per_season_correlations.empty:
        lines.append("")
        lines.append("Корреляции внутри групп (сильнейшие связи):")
        top_corr = (
            analysis.per_season_correlations
            .assign(_abs=analysis.per_season_correlations["corr"].abs())
            .sort_values("_abs", ascending=False)
            .head(5)
        )
        for row in top_corr.itertuples(index=False):
            lines.append(
                f"- Группа **{row.group}**, показатель `{row.feature}`: "
                f"ρ={row.corr:.3f}, n={row.n_shared}"
            )

    if analysis.diagnostics:
        lines.append("")
        lines.append("Диагностика:")
        for msg in analysis.diagnostics[:5]:
            lines.append(f"- {msg}")

    return "\n".join(lines)


_SELECTION_MODE_RU = {
    "auto": "автоматический (система выбирает по корреляции и покрытию)",
    "manual": "ручной (специалист задал список показателей явно)",
    "semi_auto": "полу-автоматический (специалист задал обязательные показатели, остальные добавлены автоматически)",
}


_SELECTION_EXCLUSION_RU = {
    "too_few_shared_samples": "слишком мало совместных измерений с целевым показателем",
    "constant_feature": "показатель постоянен (нет вариации) на совместных измерениях",
    "below_min_correlation": "корреляция с целевым показателем ниже требуемого порога",
    "not_significant": "связь с целевым показателем статистически незначима",
    "multicollinear_with": "дублирует уже отобранный показатель (мультиколлинеарность)",
    "budget_exhausted": "исключён из-за ограничения на число предикторов",
    "not_in_manual_list": "не входит в заданный вручную список показателей",
    "insufficient_data": "недостаточно данных для оценки связи с целевым показателем",
    "feature_absent": "показатель отсутствует в данных этого среза",
}


def _selection_reason_ru(base_code: str) -> str:
    """Render a base feature-selection exclusion code as Russian text."""
    return _SELECTION_EXCLUSION_RU.get(base_code, base_code)


def _explain_no_ml_model(model_run: ModelComparisonRun) -> list[str]:
    """Explain why no ML model was built, as explicitly as readiness reasons.

    Triggered when the slice is suitable enough to attempt modeling, but feature
    selection produced no usable predictors, so every ML model was skipped.
    """
    diagnosis = diagnose_no_predictors(model_run.feature_selection)

    if diagnosis.total_candidates == 0:
        return [
            "- ML-модель не была построена: в этом срезе нет сопутствующих показателей, "
            "которые можно было бы использовать как предикторы.",
        ]

    mode = diagnosis.selection_mode
    if mode in ("manual", "semi_auto") and diagnosis.forced_details:
        lines = [
            "- ML-модель не была построена: показатели, заданные для модели, "
            "нельзя использовать как предикторы:",
        ]
        for feature, base_code, n_shared in diagnosis.forced_details:
            lines.append(
                f"    - «{feature}»: {_selection_reason_ru(base_code)} "
                f"(совместных измерений с целевым: {n_shared})."
            )
        if mode == "semi_auto":
            lines.append("    - Автоматический отбор также не нашёл подходящих показателей.")
        lines.append(
            "  Что можно сделать: выбрать показатели с бо́льшим числом совместных измерений "
            "с целевым или переключиться на автоматический отбор."
        )
        return lines

    # auto mode (or semi_auto without forced details): aggregate the reasons.
    lines = [
        f"- ML-модель не была построена: ни один из {diagnosis.total_candidates} сопутствующих "
        "показателей не прошёл автоматический отбор предикторов. Причины:",
    ]
    for base_code, count in diagnosis.reason_counts:
        lines.append(f"    - показателей: {count} — {_selection_reason_ru(base_code)}.")
    if diagnosis.best_abs_correlation is not None:
        lines.append(
            f"  Наибольшая по модулю корреляция показателя с целевым составила "
            f"ρ≈{diagnosis.best_abs_correlation:.2f} — этого недостаточно для надёжной модели."
        )
    lines.append(
        "  Что можно сделать: ослабить порог корреляции/значимости в настройках отбора, "
        "задать показатели вручную или выбрать другой целевой показатель либо срез."
    )
    return lines


def _format_stability_warning(model_name: str, holdout_rmse: float, backtest_rmse: float, ratio: float) -> str:
    """Render a per-model stability warning line."""
    if ratio >= 1.5:
        return (
            f"⚠️ Модель показывает признаки нестабильности: на финальной проверочной части "
            f"RMSE={holdout_rmse:.3f}, но на скользящих временных окнах средний RMSE={backtest_rmse:.3f} "
            f"(отношение {ratio:.2f}×). Это может означать переобучение на хвосте обучающих данных или дрейф. "
            f"Применяйте оценки модели с осторожностью."
        )
    if ratio <= 0.7:
        return (
            f"ℹ️ На финальной проверочной части RMSE={holdout_rmse:.3f} оказалась значительно хуже "
            f"среднего по скользящим окнам RMSE={backtest_rmse:.3f}. Возможно, последний проверочный "
            f"период был нетипичным. Backtest-результат вероятно ближе к реальной ожидаемой точности."
        )
    return ""


def _format_run_level_stability(model_run: ModelComparisonRun) -> str:
    """Render a run-level stability section if most models are unstable."""
    fitted = [r for r in model_run.results if r.status == "fitted" and r.stability_ratio is not None]
    if not fitted:
        return ""
    n = len(fitted)
    high_ratio = sum(1 for r in fitted if r.stability_ratio >= 1.5)  # type: ignore[operator]
    low_ratio = sum(1 for r in fitted if r.stability_ratio <= 0.7)  # type: ignore[operator]
    threshold = 2 / 3
    if n >= 3 and high_ratio / n >= threshold:
        return (
            "\n\n**Стабильность сравнения:** "
            "⚠️ У большинства моделей метрики на финальной проверке значительно хуже backtest. "
            "Это может означать, что финальная часть данных содержит аномалию или сдвиг распределения. "
            "Доверяйте больше backtest-результатам."
        )
    if n >= 3 and low_ratio / n >= threshold:
        return (
            "\n\n**Стабильность сравнения:** "
            "⚠️ У большинства моделей метрики на финальной проверке значительно лучше backtest. "
            "Финальный проверочный период оказался необычно простым; "
            "реальная ожидаемая точность ближе к backtest-средним."
        )
    return ""


def _format_modeling_section(model_run: ModelComparisonRun | None) -> str:
    """Render the modeling section of the summary."""
    if model_run is None:
        return (
            "Моделирование не выполнялось или не дало применимой ML-модели. "
            "В отчете сохранены паспорт данных, проверка пригодности и корреляции."
        )

    best_baseline = model_run.get_best_baseline_result()
    best_ml = model_run.get_best_ml_result()
    selected_features = ", ".join(model_run.selected_features) if model_run.selected_features else "не выбраны"
    seasonal_features = getattr(model_run, "seasonal_features", ())
    if seasonal_features:
        selected_features += f" + сезонные признаки: {', '.join(seasonal_features)}"

    sel_mode = getattr(model_run, "selection_mode", "auto")
    forced = getattr(model_run, "forced_features", ())
    mode_label = _SELECTION_MODE_RU.get(sel_mode, sel_mode)

    lines = [
        f"- Режим отбора предикторов: `{sel_mode}` — {mode_label}.",
    ]
    if forced:
        lines.append(f"- Показатели, заданные специалистом: {', '.join(forced)}.")
    lines += [
        f"- Выбранные сопутствующие показатели: {selected_features}",
        f"- Лучшая базовая модель: `{best_baseline.model_name}`" if best_baseline else "- Лучшая базовая модель: нет данных",
    ]

    if best_ml is None:
        lines.extend(_explain_no_ml_model(model_run))
        return "\n".join(lines)

    lines.append(f"- Лучшая ML-модель: `{best_ml.model_name}`")
    lines.append(
        f"- Метрики на проверочной части: MAE={best_ml.metrics.get('mae', float('nan')):.3f}, "
        f"RMSE={best_ml.metrics.get('rmse', float('nan')):.3f}, "
        f"R2={best_ml.metrics.get('r2', float('nan')):.3f}, "
        f"SMAPE={best_ml.metrics.get('smape', float('nan')):.3f}"
    )

    # Combined score and selection method
    if best_ml.combined_score_used_fallback:
        lines.append(
            "- Backtest-метрики недоступны (мало данных для скользящей проверки), "
            "выбор сделан только по holdout RMSE."
        )
    else:
        lines.append("- Модель выбрана по комбинированному скору RMSE с весами holdout 40% / backtest 60%.")
    if best_ml.combined_score is not None:
        lines.append(f"- Комбинированный скор (holdout 40% + backtest 60%): {best_ml.combined_score:.3f}")
    if best_ml.stability_ratio is not None:
        lines.append(f"- Стабильность (backtest/holdout): {best_ml.stability_ratio:.2f}")

    if best_baseline is not None and best_ml.beats_best_baseline:
        lines.append("- ML-модель лучше базовой модели по комбинированному скору.")
    elif best_baseline is not None:
        lines.append(
            "- Важно: ML-модель не лучше базовой модели по комбинированному скору. "
            "Такой результат нельзя считать уверенным преимуществом машинного обучения."
        )

    # Per-model stability warnings
    if best_ml.stability_ratio is not None:
        holdout_rmse = best_ml.metrics.get("rmse", 0.0)
        backtest_rmse = best_ml.backtest_metrics.get("rmse", 0.0)
        warning = _format_stability_warning(best_ml.model_name, holdout_rmse, backtest_rmse, best_ml.stability_ratio)
        if warning:
            lines.append(f"- {warning}")

    if not best_ml.interpretability_df.empty:
        top_interpretation = " | ".join(
            f"{row.feature}={row.importance:.4f}"
            for row in best_ml.interpretability_df.itertuples(index=False)
        )
        lines.append(f"- Признаки модели (по убыванию значимости): {top_interpretation}")

    # Run-level stability section
    run_level = _format_run_level_stability(model_run)
    if run_level:
        lines.append(run_level)

    return "\n".join(lines)


def _format_estimation_section(model_run: ModelComparisonRun | None, package_path: str | None) -> str:
    """Render deployable-model guidance for missing-value estimation."""
    if model_run is None:
        return (
            "Пакет `models/best_model_package` не создан, потому что моделирование не было выполнено "
            "или срез данных признан непригодным."
        )

    best_ml = model_run.get_best_ml_result()
    if best_ml is None or not best_ml.feature_names:
        return (
            "Пакет `models/best_model_package` не создан: подходящую ML-модель построить не удалось. "
            "Причина указана в разделе «Сравнение моделей»."
        )

    lines = [
        f"- Пакет модели: `{package_path or 'не создан'}`",
        f"- Требуемые сопутствующие показатели: {', '.join(best_ml.feature_names)}",
        "- Расчетные значения являются аналитическими оценками, а не лабораторными измерениями.",
    ]
    if not best_ml.beats_best_baseline:
        lines.append("- Предупреждение: ML-модель хуже или не лучше базовой модели; применяйте оценки только как слабую подсказку.")
    if package_path:
        lines.append(
            "- Пример запуска: "
            f"`python -m water_analysis.cli estimate-missing --input data/new_measurements.csv --model-package {package_path} --output-dir reports/estimate_run`"
        )
    return "\n".join(lines)


def _format_where_to_look(package_path: str | None) -> str:
    """Render result locations."""
    lines = [
        "- Основной текстовый отчет: `summary/specialist_summary.md`.",
        "- Таблицы CSV для машинной обработки: `tables/*.csv`.",
        "- Таблицы XLSX для просмотра в Excel: `tables/*.xlsx`.",
        "- Графики: `plots/*.png`.",
        "- Метаданные запуска: `metadata/*.json`.",
    ]
    if package_path:
        lines.append(f"- Пакет модели для `estimate-missing`: `{package_path}`.")
    return "\n".join(lines)


def build_specialist_summary(
    summary_input: SpecialistSummaryInput,
    *,
    top_correlations: int = 10,
) -> str:
    """Build a specialist-facing markdown summary."""
    readiness = summary_input.readiness_assessment
    profile = summary_input.profile_report
    diagnostics = summary_input.correlation_analysis.diagnostics
    unavailable = diagnostics[diagnostics["status"] == "target_unavailable"] if not diagnostics.empty else pd.DataFrame()

    lines = [
        "# Отчет по анализу качества питьевой воды",
        "",
        "## 1. Параметры анализа",
        _format_run_parameters(summary_input.run_parameters),
        "",
        "## 2. Краткая характеристика данных",
        f"- Сценарий анализа: `{profile.scope_name}`.",
        f"- Срез: {profile.scope_label} (`{profile.scope_id}`).",
        f"- Целевой показатель: `{readiness.target}`.",
        f"- Записей журнала (табличных строк): {profile.summary['record_count']}.",
        f"- Фактов взятия проб (исследований): {profile.summary['sample_event_count']}.",
        f"- Уникальных ОКТМО: {profile.summary['unique_oktmo_count']}.",
        f"- Уникальных точек отбора: {profile.summary['unique_point_count']}.",
        f"- Период наблюдений: {_format_value(profile.summary['observation_start'])} - {_format_value(profile.summary['observation_end'])}.",
        f"- Число показателей: {profile.summary['indicator_count']}.",
        "",
        "## 3. Пригодность данных к моделированию",
        f"- Статус: `{readiness.status}` — {READINESS_STATUS_RU.get(readiness.status, readiness.status)}.",
        f"- Наблюдений целевого показателя: {readiness.target_observation_count}.",
        f"- Доля пропусков целевого показателя: {readiness.target_missing_ratio:.3f}.",
        f"- Доля цензурированных значений целевого показателя: {readiness.target_censored_ratio:.3f}.",
        f"- Доступных сопутствующих показателей: {readiness.eligible_predictor_count}.",
        "",
        "Причины и ограничения:",
        _format_issues(readiness),
        "",
        "## 4. Найденные зависимости",
        _format_correlations(summary_input.correlation_analysis, readiness.target, top_correlations),
        "",
        "## 5. Сезонность",
        _format_seasonality_section(summary_input.seasonality_analysis),
        "",
        "## 6. Сравнение моделей",
        _format_modeling_section(summary_input.model_run),
        "",
        "## 7. Возможность расчетной оценки прогнозируемых значений",
        _format_estimation_section(summary_input.model_run, summary_input.deployable_model_package),
        "",
        "## 8. Ограничения интерпретации",
        "- Система является инструментом поддержки решения специалиста, а не заменой лабораторных исследований.",
        "- Если ML-модель не лучше базовой модели, ее результаты нельзя считать надежным улучшением.",
        "- Расчетные значения из `estimate-missing` нельзя записывать как лабораторно измеренные.",
    ]

    if readiness.status == "unsuitable":
        lines.append("- Для этого среза обучение модели не рекомендуется.")
    if not unavailable.empty:
        lines.append("- Целевой показатель отсутствует в исходных данных для выбранного среза; это диагностическое состояние.")

    lines.extend(
        [
            "",
            "## 9. Где смотреть результаты",
            _format_where_to_look(summary_input.deployable_model_package),
        ]
    )

    return "\n".join(lines).strip() + "\n"
