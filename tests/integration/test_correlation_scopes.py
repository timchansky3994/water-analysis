import pandas as pd

from water_analysis.analysis.correlations import run_correlation_analysis
from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.preprocessing.long_format import build_canonical_long_format


def _build_correlation_raw_dataframe() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    dates = ["09.01.2018", "10.01.2018", "11.01.2018", "12.01.2018"]
    series_map = {
        ("00000000001.10110.0010", "Точка на распределительной сети"): {
            "Жесткость общая": ["1,0", "2,0", "3,0", "4,0"],
            "Цветность": ["2,0", "4,0", "6,0", "8,0"],
            "Мутность (по формазину)": ["1,0", "1,5", "2,0", "2,5"],
        },
        ("00000000001.10150.0011", "Точка водоснабжения"): {
            "Жесткость общая": ["2,0", "3,0", "4,0", "5,0"],
            "Цветность": ["4,0", "6,0", "8,0", "10,0"],
            "Мутность (по формазину)": ["1,2", "1,7", "2,2", "2,7"],
        },
        ("00000000002.10110.0001", "Точка на распределительной сети"): {
            "Жесткость общая": ["3,0", "4,0", "5,0", "6,0"],
            "Цветность": ["6,0", "8,0", "10,0", "12,0"],
            "Мутность (по формазину)": ["1,4", "1,9", "2,4", "2,9"],
        },
        ("00000000001.10320.0012", "Подземный источник"): {
            "Жесткость общая": ["10,0", "10,0", "10,0", "10,0"],
            "Цветность": ["1,0", "1,0", "1,0", "1,0"],
            "Мутность (по формазину)": ["0,1", "0,1", "0,1", "0,1"],
        },
    }

    for sample_date_index, sample_date in enumerate(dates):
        for (code, point_type_name), indicators in series_map.items():
            for indicator, values in indicators.items():
                rows.append(
                    {
                        "Дата проведения исследования": sample_date,
                        "Тип точки": point_type_name,
                        "Код точки": code,
                        "Гигиенический показатель": indicator,
                        "Норматив": "",
                        "Строчное значение": "",
                        "Результат исследования": values[sample_date_index],
                        "Нижний предел обнаружения": "",
                        "Верхний предел обнаружения": "",
                        "Ошибка метода определения": "",
                        "Нормативная документация": "",
                        "Цели исследований": "",
                        "Соотв. ПДК": "",
                    }
                )

    return pd.DataFrame(rows)


def test_correlation_engine_supports_global_oktmo_and_drinking_scopes() -> None:
    long_df = build_canonical_long_format(_build_correlation_raw_dataframe())

    global_analysis = run_correlation_analysis(
        build_scope_slices(long_df, scope_name="global"),
        targets=["Жесткость общая"],
        methods=["spearman", "pearson"],
        min_shared_samples=4,
    )
    oktmo_analysis = run_correlation_analysis(
        build_scope_slices(long_df, scope_name="oktmo"),
        targets=["Жесткость общая"],
        methods=["spearman"],
        min_shared_samples=4,
    )
    drinking_analysis = run_correlation_analysis(
        build_scope_slices(long_df, scope_name="drinking_water_combined"),
        targets=["Жесткость общая"],
        methods=["spearman"],
        min_shared_samples=4,
    )

    assert not global_analysis.results.empty
    assert set(global_analysis.results["method"]) == {"spearman", "pearson"}
    assert "Цветность" in set(global_analysis.results["feature"])
    assert set(oktmo_analysis.results["scope_name"]) == {"oktmo"}
    assert set(drinking_analysis.results["scope_name"]) == {"drinking_water_combined"}
    assert set(drinking_analysis.results["PointType_Code"]) == {"10110+10150"}
    assert (drinking_analysis.results["n_shared"] >= 4).all()
