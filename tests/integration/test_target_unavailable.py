import pandas as pd

from water_analysis.analysis.correlations import run_correlation_analysis
from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.preprocessing.long_format import build_canonical_long_format


def test_correlation_engine_reports_target_unavailable() -> None:
    raw_df = pd.DataFrame(
        {
            "Дата проведения исследования": ["09.01.2018", "09.01.2018", "10.01.2018", "10.01.2018"],
            "Тип точки": ["Точка на распределительной сети"] * 4,
            "Код точки": ["00000000001.10110.0010"] * 4,
            "Гигиенический показатель": [
                "Жесткость общая",
                "Цветность",
                "Жесткость общая",
                "Цветность",
            ],
            "Норматив": [""] * 4,
            "Строчное значение": [""] * 4,
            "Результат исследования": ["1,0", "2,0", "2,0", "4,0"],
            "Нижний предел обнаружения": [""] * 4,
            "Верхний предел обнаружения": [""] * 4,
            "Ошибка метода определения": [""] * 4,
            "Нормативная документация": [""] * 4,
            "Цели исследований": [""] * 4,
            "Соотв. ПДК": [""] * 4,
        }
    )

    long_df = build_canonical_long_format(raw_df)
    analysis = run_correlation_analysis(
        build_scope_slices(long_df, scope_name="global"),
        targets=["Химическое потребление кислорода (ХПК)"],
        methods=["spearman"],
        min_shared_samples=2,
    )

    assert analysis.results.empty
    assert not analysis.diagnostics.empty
    assert set(analysis.diagnostics["status"]) == {"target_unavailable"}
