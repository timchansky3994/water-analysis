import pandas as pd

from water_analysis.preprocessing.long_format import build_canonical_long_format
from water_analysis.preprocessing.pivot_builder import build_indicator_pivot


def test_default_pivot_does_not_merge_distinct_sample_points() -> None:
    raw_df = pd.DataFrame(
        {
            "Дата проведения исследования": ["09.01.2018", "09.01.2018"],
            "Тип точки": ["Точка на распределительной сети", "Точка на распределительной сети"],
            "Код точки": ["00000000001.10110.0010", "00000000001.10110.0011"],
            "Гигиенический показатель": ["Жесткость общая", "Жесткость общая"],
            "Норматив": ["", ""],
            "Строчное значение": ["", ""],
            "Результат исследования": ["1,0", "3,0"],
            "Нижний предел обнаружения": ["", ""],
            "Верхний предел обнаружения": ["", ""],
            "Ошибка метода определения": ["", ""],
            "Нормативная документация": ["", ""],
            "Цели исследований": ["", ""],
            "Соотв. ПДК": ["", ""],
        }
    )

    long_df = build_canonical_long_format(raw_df)

    sample_point_pivot = build_indicator_pivot(long_df)
    point_type_pivot = build_indicator_pivot(long_df, aggregation_level="point_type_level")

    assert len(sample_point_pivot) == 2
    assert sorted(sample_point_pivot["Жесткость общая"].tolist()) == [1.0, 3.0]
    assert len(point_type_pivot) == 1
    assert point_type_pivot.iloc[0]["Жесткость общая"] == 2.0
