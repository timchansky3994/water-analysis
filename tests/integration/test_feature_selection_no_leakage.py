import pandas as pd

from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.modeling.trainer import compare_models_in_scope
from water_analysis.preprocessing.long_format import build_canonical_long_format


def test_feature_selection_uses_only_training_fold_information() -> None:
    raw_rows: list[dict[str, str]] = []
    dates = [f"{day:02d}.01.2018" for day in range(1, 11)]
    target_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 50.0, 60.0]
    feature_good = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    feature_leak = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 50.0, 60.0]

    for sample_date, target_value, good_value, leak_value in zip(
        dates,
        target_values,
        feature_good,
        feature_leak,
        strict=True,
    ):
        for indicator, value in {
            "Жесткость общая": target_value,
            "Цветность": good_value,
            "Мутность (по формазину)": leak_value,
        }.items():
            raw_rows.append(
                {
                    "Дата проведения исследования": sample_date,
                    "Тип точки": "Точка на распределительной сети",
                    "Код точки": "00000000001.10110.0010",
                    "Гигиенический показатель": indicator,
                    "Норматив": "",
                    "Строчное значение": "",
                    "Результат исследования": str(value).replace(".", ","),
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
        test_size=0.2,
        min_train_size=6,
        min_target_observations=6,
        min_shared_samples=6,
        min_eligible_predictors=1,
        min_target_correlation=0.3,
        max_features=2,
    )

    assert "Цветность" in run.selected_features
    assert "Мутность (по формазину)" not in run.selected_features
