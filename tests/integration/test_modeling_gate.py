import pandas as pd
import pytest

from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.modeling.trainer import ModelingNotAllowedError, compare_models_in_scope
from water_analysis.preprocessing.long_format import build_canonical_long_format


def test_training_blocked_when_readiness_is_unsuitable() -> None:
    raw_rows: list[dict[str, str]] = []
    dates = ["09.01.2018", "10.01.2018", "11.01.2018", "12.01.2018", "13.01.2018", "14.01.2018"]
    for sample_date in dates:
        for indicator, value in {
            "Жесткость общая": "5,0",
            "Цветность": "1,0",
            "Мутность (по формазину)": "2,0",
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

    with pytest.raises(ModelingNotAllowedError):
        compare_models_in_scope(
            scope_slice,
            target="Жесткость общая",
            model_names=["bayesian_ridge"],
            min_target_observations=4,
            min_shared_samples=4,
            min_eligible_predictors=1,
        )
