from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_linear_raw_dataframe(
    *,
    dates: list[str],
    target_values: list[float],
    feature_map: dict[str, list[float]],
    point_code: str = "00000000001.10110.0010",
    point_type_name: str = "Точка на распределительной сети",
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for index, sample_date in enumerate(dates):
        values = {"Жесткость общая": target_values[index], **{name: series[index] for name, series in feature_map.items()}}
        for indicator, value in values.items():
            rows.append(
                {
                    "Дата проведения исследования": sample_date,
                    "Тип точки": point_type_name,
                    "Код точки": point_code,
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
    return pd.DataFrame(rows)


def build_secondary_raw_dataframe(
    *,
    dates: list[str],
    target_values: list[float],
    feature_map: dict[str, list[float]],
    point_code: str = "00000000001.10110.0010",
    point_type_name: str = "Точка на распределительной сети",
) -> pd.DataFrame:
    """Build a synthetic raw DataFrame in the secondary regional export format.

    Omits optional columns absent in that format: Строчное значение,
    Верхний предел обнаружения, Цели исследований, Соотв. ПДК.
    """
    rows: list[dict[str, str]] = []
    for index, sample_date in enumerate(dates):
        values = {"Жесткость общая": target_values[index], **{name: series[index] for name, series in feature_map.items()}}
        for indicator, value in values.items():
            rows.append(
                {
                    "Дата": sample_date,
                    "Тип точки": point_type_name,
                    "Код точки": point_code,
                    "Гигиенический показатель": indicator,
                    "Норматив": "",
                    "Результат исследования": str(value).replace(".", ","),
                    "НПО": "",
                    "Ошибка метода определения": "",
                    "Нормативная документация": "",
                }
            )
    return pd.DataFrame(rows)


def write_raw_csv(dataframe: pd.DataFrame, path: Path) -> Path:
    dataframe.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
    return path
