"""Plot generation for specialist-facing report bundles."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from water_analysis.modeling.trainer import ModelComparisonRun

if TYPE_CHECKING:
    from water_analysis.analysis.seasonality import SeasonalityAnalysis

_LOGGER = logging.getLogger(__name__)

plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def _clean_label(value: object) -> str:
    """Remove control characters from plot labels."""
    text = str(value)
    return "".join(character for character in text if ord(character) >= 32 and ord(character) not in range(127, 160))


def _save_figure(path: str | Path, *, dpi: int) -> str:
    """Save the current figure and close it."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return str(output_path)


def _placeholder_plot(path: str | Path, title: str, message: str, *, dpi: int) -> str:
    """Create a simple explanatory placeholder plot."""
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    return _save_figure(path, dpi=dpi)


def plot_predicted_vs_actual(run: ModelComparisonRun | None, path: str | Path, *, dpi: int = 150) -> str:
    """Plot predicted versus actual values for the preferred report model."""
    if run is None:
        return _placeholder_plot(path, "Факт и расчет", "Моделирование для этого отчета не выполнялось.", dpi=dpi)

    preferred = run.get_preferred_result_for_reporting()
    if preferred is None:
        return _placeholder_plot(path, "Факт и расчет", "Нет обученной модели для отображения.", dpi=dpi)

    predictions = run.holdout_predictions_df[run.holdout_predictions_df["model_name"] == preferred.model_name]
    if predictions.empty:
        return _placeholder_plot(path, "Факт и расчет", "Нет расчетов на проверочной части.", dpi=dpi)

    plt.figure(figsize=(6, 6))
    plt.scatter(predictions["actual"], predictions["predicted"], alpha=0.7, edgecolors="black")
    min_value = min(predictions["actual"].min(), predictions["predicted"].min())
    max_value = max(predictions["actual"].max(), predictions["predicted"].max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="red")
    plt.xlabel("Фактическое значение")
    plt.ylabel("Расчетное значение")
    plt.title(f"Фактические и расчетные значения: {preferred.model_name}")
    plt.grid(alpha=0.3)
    return _save_figure(path, dpi=dpi)


def plot_residuals(run: ModelComparisonRun | None, path: str | Path, *, dpi: int = 150) -> str:
    """Plot residuals for the preferred report model."""
    if run is None:
        return _placeholder_plot(path, "Остатки модели", "Моделирование для этого отчета не выполнялось.", dpi=dpi)

    preferred = run.get_preferred_result_for_reporting()
    if preferred is None:
        return _placeholder_plot(path, "Остатки модели", "Нет обученной модели для отображения.", dpi=dpi)

    predictions = run.holdout_predictions_df[run.holdout_predictions_df["model_name"] == preferred.model_name].copy()
    if predictions.empty:
        return _placeholder_plot(path, "Остатки модели", "Нет расчетов на проверочной части.", dpi=dpi)
    predictions["residual"] = predictions["actual"] - predictions["predicted"]

    plt.figure(figsize=(7, 4))
    plt.scatter(predictions["predicted"], predictions["residual"], alpha=0.7, edgecolors="black")
    plt.axhline(0.0, linestyle="--", color="red")
    plt.xlabel("Расчетное значение")
    plt.ylabel("Остаток модели")
    plt.title(f"Остатки модели: {preferred.model_name}")
    plt.grid(alpha=0.3)
    return _save_figure(path, dpi=dpi)


def plot_backtest(run: ModelComparisonRun | None, path: str | Path, *, dpi: int = 150) -> str:
    """Plot holdout, backtest and combined RMSE side by side for every fitted model."""
    if run is None or run.comparison_df.empty:
        return _placeholder_plot(path, "Сравнение моделей", "Метрики недоступны.", dpi=dpi)

    df = run.comparison_df[run.comparison_df["status"] == "fitted"].copy()
    if df.empty or "holdout_rmse" not in df.columns:
        return _placeholder_plot(path, "Сравнение моделей", "Нет подходящих результатов.", dpi=dpi)

    has_backtest = "backtest_rmse" in df.columns and df["backtest_rmse"].notna().any()
    has_combined = "combined_score" in df.columns and df["combined_score"].notna().any()
    if not has_combined:
        _LOGGER.warning("combined_score column absent or all-NaN in comparison_df; falling back to two-bar plot.")

    # Sort by combined_score ascending; fallback to holdout_rmse
    if has_combined:
        df = df.sort_values("combined_score", na_position="last").reset_index(drop=True)
    else:
        df = df.sort_values("holdout_rmse").reset_index(drop=True)

    best_result = run.get_best_ml_result()
    best_name = best_result.model_name if best_result else None

    n = len(df)
    model_names = list(df["model_name"])

    if has_backtest and has_combined:
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 5))
        xs = list(range(n))

        ax.bar(
            [i - width for i in xs],
            df["holdout_rmse"],
            width=width,
            label="Holdout RMSE (финальный тест)",
            color="#E8A838",
        )
        ax.bar(
            xs,
            df["backtest_rmse"].fillna(0),
            width=width,
            label="Backtest RMSE (среднее по отрезкам)",
            color="#4C78A8",
        )
        # Draw combined_score only for rows where it is not NaN
        combined_mask = df["combined_score"].notna()
        xs_comb = [i + width for i, has_val in enumerate(combined_mask) if has_val]
        vals_comb = df.loc[combined_mask, "combined_score"].tolist()
        if xs_comb:
            ax.bar(
                xs_comb,
                vals_comb,
                width=width,
                label="Комбинированный скор (выбор модели)",
                color="#72B7B2",
            )

        ax.set_xticks(xs)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_title("Сравнение моделей: holdout, backtest и комбинированный RMSE")

        # Star over combined_score bar of best model (or holdout bar if combined is NaN)
        if best_name and best_name in model_names:
            idx = model_names.index(best_name)
            combined_val = df.iloc[idx]["combined_score"] if "combined_score" in df.columns else float("nan")
            if pd.notna(combined_val):
                bar_x = idx + width
                ann_val = float(combined_val)
            else:
                bar_x = idx - width
                ann_val = float(df.iloc[idx]["holdout_rmse"])
            ax.annotate(
                "★ выбранная",
                xy=(bar_x, ann_val),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#1A6B5A",
            )

    elif has_backtest:
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))
        xs = list(range(n))
        ax.bar(
            [i - width / 2 for i in xs],
            df["holdout_rmse"],
            width=width,
            label="Holdout RMSE (финальный тест)",
            color="#E8A838",
        )
        ax.bar(
            [i + width / 2 for i in xs],
            df["backtest_rmse"].fillna(0),
            width=width,
            label="Backtest RMSE (среднее по отрезкам)",
            color="#4C78A8",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_title("Сравнение моделей: holdout, backtest и комбинированный RMSE")

        if best_name and best_name in model_names:
            idx = model_names.index(best_name)
            holdout_val = float(df.iloc[idx]["holdout_rmse"])
            ax.annotate(
                "★ выбранная",
                xy=(idx - width / 2, holdout_val),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#B54E12",
            )

    else:
        fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))
        xs = list(range(n))
        ax.bar(xs, df["holdout_rmse"], color="#E8A838", label="Holdout RMSE (финальный тест)")
        ax.set_xticks(xs)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_title("Сравнение моделей: holdout RMSE")

        if best_name and best_name in model_names:
            idx = model_names.index(best_name)
            holdout_val = float(df.iloc[idx]["holdout_rmse"])
            ax.annotate(
                "★ выбранная",
                xy=(idx, holdout_val),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#B54E12",
            )

    ax.set_ylabel("RMSE")
    ax.set_xlabel("Модель")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save_figure(path, dpi=dpi)


def plot_correlation_heatmap(
    correlation_results: pd.DataFrame,
    path: str | Path,
    *,
    target: str,
    max_features: int = 20,
    dpi: int = 150,
) -> str:
    """Plot a heatmap from existing correlation results."""
    clean_target = _clean_label(target)
    if correlation_results.empty or "target" not in correlation_results.columns:
        return _placeholder_plot(path, "Карта корреляций", f"Нет результатов корреляции для показателя '{clean_target}'.", dpi=dpi)

    scoped = correlation_results[correlation_results["target"] == target].copy()
    if scoped.empty:
        return _placeholder_plot(path, "Карта корреляций", f"Нет результатов корреляции для показателя '{clean_target}'.", dpi=dpi)

    top_features = (
        scoped.assign(corr_abs=scoped["corr"].abs())
        .sort_values(["corr_abs", "n_shared"], ascending=[False, False])["feature"]
        .drop_duplicates()
        .head(max_features)
        .tolist()
    )
    heatmap_df = scoped[scoped["feature"].isin(top_features)].pivot_table(
        index="method",
        columns="feature",
        values="corr",
        aggfunc="first",
    )
    if heatmap_df.empty:
        return _placeholder_plot(path, "Карта корреляций", "Нет значений для построения карты корреляций.", dpi=dpi)

    plt.figure(figsize=(max(6, 0.6 * len(heatmap_df.columns)), 3 + 0.4 * len(heatmap_df.index)))
    plt.imshow(heatmap_df.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Корреляция")
    plt.xticks(range(len(heatmap_df.columns)), [_clean_label(value) for value in heatmap_df.columns], rotation=45, ha="right")
    plt.yticks(range(len(heatmap_df.index)), [_clean_label(value) for value in heatmap_df.index])
    plt.title(f"Карта корреляций для показателя: {clean_target}")
    return _save_figure(path, dpi=dpi)


def plot_seasonal_profile(
    seasonality_analysis: "SeasonalityAnalysis",
    path: str | Path,
    *,
    dpi: int = 150,
) -> str:
    """Plot target distribution by season or month (boxplot or bar+errorbar)."""
    stats = seasonality_analysis.group_stats
    target = _clean_label(seasonality_analysis.target)

    if stats.empty or "median" not in stats.columns:
        return _placeholder_plot(
            path,
            "Сезонный профиль",
            "Недостаточно данных для построения сезонного профиля.",
            dpi=dpi,
        )

    groups = [_clean_label(str(g)) for g in stats["group"].tolist()]
    medians = stats["median"].tolist()
    q25 = stats["q25"].tolist()
    q75 = stats["q75"].tolist()
    ns = stats["n_observations"].tolist()

    errors_lo = [max(0.0, m - lo) for m, lo in zip(medians, q25)]
    errors_hi = [max(0.0, hi - m) for m, hi in zip(medians, q75)]

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 5))
    xs = list(range(len(groups)))
    ax.bar(xs, medians, color="#4C78A8", alpha=0.7, label="Медиана")
    ax.errorbar(xs, medians, yerr=[errors_lo, errors_hi], fmt="none", color="black", capsize=5, label="IQR (Q25–Q75)")

    for i, (x, n) in enumerate(zip(xs, ns)):
        ax.annotate(f"n={n}", xy=(x, 0), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels(groups, rotation=20 if len(groups) > 6 else 0, ha="right" if len(groups) > 6 else "center")
    granularity_label = "месяцам" if seasonality_analysis.granularity == "month" else "сезонам"
    ax.set_title(f"Распределение «{target}» по {granularity_label}")
    ax.set_ylabel(target)
    ax.set_xlabel("Сезон" if seasonality_analysis.granularity == "season" else "Месяц")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    detected = seasonality_analysis.seasonal_pattern_detected
    p_info = ""
    if seasonality_analysis.pattern_test.get("test") == "kruskal_wallis":
        p_val = seasonality_analysis.pattern_test.get("p_value", float("nan"))
        p_info = f" (p={p_val:.3g})"
    status_text = f"Сезонный паттерн {'обнаружен' if detected else 'не обнаружен'}{p_info}"
    fig.text(0.5, -0.02, status_text, ha="center", fontsize=9, style="italic")

    return _save_figure(path, dpi=dpi)


def plot_cooccurrence_heatmap(
    cooccurrence_matrix: pd.DataFrame,
    path: str | Path,
    *,
    max_features: int = 20,
    dpi: int = 150,
) -> str:
    """Plot a heatmap from an existing co-occurrence matrix."""
    if cooccurrence_matrix.empty:
        return _placeholder_plot(path, "Совместные измерения", "Матрица совместных измерений недоступна.", dpi=dpi)

    diagonal = pd.Series(cooccurrence_matrix.values.diagonal(), index=cooccurrence_matrix.index)
    top_features = diagonal.sort_values(ascending=False).head(max_features).index.tolist()
    matrix = cooccurrence_matrix.loc[top_features, top_features]

    plt.figure(figsize=(max(6, 0.5 * len(matrix.columns)), max(5, 0.4 * len(matrix.index))))
    plt.imshow(matrix.values, aspect="auto", cmap="Blues")
    plt.colorbar(label="Совместных измерений")
    plt.xticks(range(len(matrix.columns)), [_clean_label(value) for value in matrix.columns], rotation=45, ha="right")
    plt.yticks(range(len(matrix.index)), [_clean_label(value) for value in matrix.index])
    plt.title("Карта совместных измерений показателей")
    return _save_figure(path, dpi=dpi)
