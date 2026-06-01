import pandas as pd

from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.preprocessing.long_format import build_canonical_long_format
from water_analysis.profiling.readiness import assess_readiness


def _build_readiness_raw_dataframe(*, constant_target: bool = False) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    dates = ["09.01.2018", "10.01.2018", "11.01.2018", "12.01.2018", "13.01.2018", "14.01.2018"]
    code = "00000000001.10110.0010"
    point_type_name = "Точка на распределительной сети"
    hardness_values = ["2,0"] * 6 if constant_target else ["1,0", "2,0", "3,0", "4,0", "5,0", "6,0"]
    color_values = ["2,0", "4,0", "6,0", "8,0", "10,0", "12,0"]
    turbidity_values = ["1,0", "1,5", "2,0", "2,5", "3,0", "3,5"]

    indicators = [
        ("Жесткость общая", hardness_values),
        ("Цветность", color_values),
        ("Мутность (по формазину)", turbidity_values),
    ]

    for date_index, sample_date in enumerate(dates):
        for indicator, values in indicators:
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


def test_readiness_marks_suitable_target_when_data_is_usable() -> None:
    long_df = build_canonical_long_format(_build_readiness_raw_dataframe())
    scope_slice = build_scope_slices(long_df, scope_name="global")

    assessments = assess_readiness(
        scope_slice,
        targets=["Жесткость общая"],
        min_target_observations=4,
        min_shared_samples=4,
        min_eligible_predictors=2,
    )

    assert len(assessments) == 1
    assert assessments[0].status == "suitable"
    assert assessments[0].eligible_predictor_count >= 2


def test_readiness_marks_target_unavailable_explicitly() -> None:
    long_df = build_canonical_long_format(_build_readiness_raw_dataframe())
    scope_slice = build_scope_slices(long_df, scope_name="global")

    assessment = assess_readiness(scope_slice, targets=["Химическое потребление кислорода (ХПК)"])[0]

    assert assessment.status == "unsuitable"
    assert any(issue.code == "target_unavailable" for issue in assessment.issues)


def test_readiness_rejects_constant_target() -> None:
    long_df = build_canonical_long_format(_build_readiness_raw_dataframe(constant_target=True))
    scope_slice = build_scope_slices(long_df, scope_name="global")

    assessment = assess_readiness(
        scope_slice,
        targets=["Жесткость общая"],
        min_target_observations=4,
        min_shared_samples=4,
        min_eligible_predictors=2,
    )[0]

    assert assessment.status == "unsuitable"
    assert any(issue.code == "target_constant" for issue in assessment.issues)
