import pandas as pd

from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.modeling.trainer import compare_models_in_scope
from water_analysis.preprocessing.long_format import build_canonical_long_format


def test_ml_model_beats_baseline_when_signal_exists() -> None:
    raw_rows: list[dict[str, str]] = []
    dates = [f"{day:02d}.01.2018" for day in range(1, 13)]
    target_values = [float(day) for day in range(1, 13)]
    feature_values = [value * 2.0 for value in target_values]

    for sample_date, target_value, feature_value in zip(dates, target_values, feature_values, strict=True):
        for indicator, value in {
            "Жесткость общая": str(target_value).replace(".", ","),
            "Цветность": str(feature_value).replace(".", ","),
            "Мутность (по формазину)": str(feature_value / 2).replace(".", ","),
        }.items():
            raw_rows.append(
                {
                    "Дата проведения исследования": sample_date,
                    "Тип точки": "Точка на распределительной сети",
                    "Код точки": "00000000001.10110.0010",
                    "Гигиенический показатель": indicator,
                    "Норматив": "",
                    "Строчное значение": "",
                    "Результат исследования": value,
                    "Нижний предел обнаружения": "",
                    "Верхний предел обнаружения": "",
                    "Ошибка метода определения": "",
                    "Нормативная документация": "",
                    "Цели исследований": "",
                    "Соотв. ПДК": "",
                }
            )

    long_df = build_canonical_long_format(pd.DataFrame(raw_rows))
    scope_slice = build_scope_slices(long_df, scope_name="global")[0]
    run = compare_models_in_scope(
        scope_slice,
        target="Жесткость общая",
        model_names=["bayesian_ridge"],
        test_size=0.25,
        min_train_size=6,
        min_target_observations=6,
        min_shared_samples=6,
        min_eligible_predictors=1,
        min_target_correlation=0.2,
        max_features=2,
    )

    bayes_row = run.comparison_df[run.comparison_df["model_name"] == "bayesian_ridge"].iloc[0]
    assert bool(bayes_row["beats_best_baseline"]) is True
    assert "ml_beats_baseline_combined" in str(bayes_row["comparison_note"])
