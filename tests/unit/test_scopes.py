import pandas as pd

from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.preprocessing.long_format import build_canonical_long_format


def _build_scope_raw_dataframe() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    dates = ["09.01.2018", "10.01.2018"]
    measurements = [
        ("00000000001.10110.0010", "Точка на распределительной сети", "Жесткость общая", ["1,0", "2,0"]),
        ("00000000001.10150.0011", "Точка водоснабжения", "Жесткость общая", ["1,5", "2,5"]),
        ("00000000001.10320.0012", "Подземный источник", "Жесткость общая", ["3,0", "3,5"]),
        ("00000000002.10110.0001", "Точка на распределительной сети", "Жесткость общая", ["4,0", "5,0"]),
    ]

    for date_index, sample_date in enumerate(dates):
        for code, point_type_name, indicator, values in measurements:
            rows.append(
                {
                    "Дата проведения исследования": sample_date,
                    "Тип точки": point_type_name,
                    "Код точки": code,
                    "Гигиенический показатель": indicator,
                    "Норматив": "",
                    "Строчное значение": "",
                    "Результат исследования": values[date_index],
                    "Нижний предел обнаружения": "",
                    "Верхний предел обнаружения": "",
                    "Ошибка метода определения": "",
                    "Нормативная документация": "",
                    "Цели исследований": "",
                    "Соотв. ПДК": "",
                }
            )

    return pd.DataFrame(rows)


def test_build_scope_slices_supports_all_required_scopes() -> None:
    long_df = build_canonical_long_format(_build_scope_raw_dataframe())

    global_slices = build_scope_slices(long_df, scope_name="global")
    oktmo_slices = build_scope_slices(long_df, scope_name="oktmo")
    oktmo_type_slices = build_scope_slices(long_df, scope_name="oktmo_point_type")
    drinking_slices = build_scope_slices(long_df, scope_name="drinking_water_combined")
    point_slices = build_scope_slices(long_df, scope_name="point")

    assert len(global_slices) == 1
    assert len(oktmo_slices) == 2
    assert len(oktmo_type_slices) == 4
    assert len(drinking_slices) == 1
    assert len(point_slices) == 4
    assert set(drinking_slices[0].dataframe["PointType_Code"].astype(str).unique()) == {"10110", "10150"}


def test_build_scope_slices_respects_filters() -> None:
    long_df = build_canonical_long_format(_build_scope_raw_dataframe())

    oktmo_slice = build_scope_slices(long_df, scope_name="oktmo", oktmo="00000000001")
    point_slice = build_scope_slices(long_df, scope_name="point", point_code="00000000001.10110.0010")

    assert len(oktmo_slice) == 1
    assert oktmo_slice[0].selector["OKTMO"] == "00000000001"
    assert len(point_slice) == 1
    assert point_slice[0].selector["FullPointCode"] == "00000000001.10110.0010"
