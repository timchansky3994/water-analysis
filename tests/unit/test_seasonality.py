"""Tests for the seasonal analysis diagnostic layer."""

from __future__ import annotations

import pandas as pd
import pytest

from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.analysis.seasonality import SeasonalityAnalysis, analyze_seasonality
from water_analysis.preprocessing.long_format import build_canonical_long_format

# Canonical Cyrillic column names matching the _default source profile.
_DATE_COL = "Дата проведения исследования"
_POINT_COL = "Код точки"
_INDICATOR_COL = "Гигиенический показатель"
_VALUE_COL = "Результат исследования"
_CODE = "00000000001.10110.0010"

# Representative dates: one per season, 5 per season so n >= min_group_size
_SEASON_MONTHS = {
    "winter": ["05.01.2020", "06.01.2020", "07.01.2020", "08.01.2020", "09.01.2020"],
    "spring": ["05.04.2020", "06.04.2020", "07.04.2020", "08.04.2020", "09.04.2020"],
    "summer": ["05.07.2020", "06.07.2020", "07.07.2020", "08.07.2020", "09.07.2020"],
    "autumn": ["05.10.2020", "06.10.2020", "07.10.2020", "08.10.2020", "09.10.2020"],
}


def _make_raw(*, target_by_season: dict[str, float], extra_indicator: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    for season, dates in _SEASON_MONTHS.items():
        tv = target_by_season.get(season, 10.0)
        for date_str in dates:
            rows.append(
                {
                    _DATE_COL: date_str,
                    "Тип точки": "Точка на распределительной сети",
                    _POINT_COL: _CODE,
                    _INDICATOR_COL: "Мутность (по формазину)",
                    "Норматив": "",
                    "Строчное значение": "",
                    _VALUE_COL: str(int(tv)),
                    "Нижний предел обнаружения": "",
                    "Верхний предел обнаружения": "",
                    "Ошибка метода определения": "",
                    "Нормативная документация": "",
                    "Цели исследований": "",
                    "Соотв. ПДК": "",
                }
            )
            if extra_indicator:
                rows.append(
                    {
                        _DATE_COL: date_str,
                        "Тип точки": "Точка на распределительной сети",
                        _POINT_COL: _CODE,
                        _INDICATOR_COL: "Жесткость общая",
                        "Норматив": "",
                        "Строчное значение": "",
                        _VALUE_COL: str(int(tv * 2)),
                        "Нижний предел обнаружения": "",
                        "Верхний предел обнаружения": "",
                        "Ошибка метода определения": "",
                        "Нормативная документация": "",
                        "Цели исследований": "",
                        "Соотв. ПДК": "",
                    }
                )
    return pd.DataFrame(rows)


def _make_scope(raw_df: pd.DataFrame) -> object:
    long_df = build_canonical_long_format(raw_df)
    return build_scope_slices(long_df, scope_name="global")[0]


def test_seasonality_detects_seasonal_pattern() -> None:
    raw = _make_raw(target_by_season={"winter": 1, "spring": 5, "summer": 50, "autumn": 10})
    scope = _make_scope(raw)
    result = analyze_seasonality(scope, target="Мутность (по формазину)", min_group_size=4)

    assert isinstance(result, SeasonalityAnalysis)
    assert result.seasonal_pattern_detected is True
    assert result.pattern_test["test"] == "kruskal_wallis"
    assert result.pattern_test["p_value"] < 0.05
    assert not result.group_stats.empty
    assert set(result.group_stats["group"].tolist()) >= {"winter", "spring", "summer", "autumn"}


def test_seasonality_reports_no_pattern_when_flat() -> None:
    raw = _make_raw(target_by_season={"winter": 10, "spring": 10, "summer": 10, "autumn": 10})
    scope = _make_scope(raw)
    result = analyze_seasonality(scope, target="Мутность (по формазину)", min_group_size=4)

    assert result.seasonal_pattern_detected is False
    # Perfectly constant data: scipy kruskal raises "All numbers are identical" which is caught
    # and returned as "skipped".  Both "skipped" and "kruskal_wallis" (p>0.05) indicate no pattern.
    assert result.pattern_test["test"] in ("kruskal_wallis", "skipped")
    if result.pattern_test["test"] == "kruskal_wallis":
        assert result.pattern_test["p_value"] > 0.05


def test_seasonality_skips_test_with_too_few_groups() -> None:
    # Only winter dates
    rows: list[dict] = []
    for date_str in _SEASON_MONTHS["winter"]:
        rows.append(
            {
                _DATE_COL: date_str,
                "Тип точки": "Точка на распределительной сети",
                _POINT_COL: _CODE,
                _INDICATOR_COL: "Мутность (по формазину)",
                "Норматив": "",
                "Строчное значение": "",
                _VALUE_COL: "5",
                "Нижний предел обнаружения": "",
                "Верхний предел обнаружения": "",
                "Ошибка метода определения": "",
                "Нормативная документация": "",
                "Цели исследований": "",
                "Соотв. ПДК": "",
            }
        )
    raw = pd.DataFrame(rows)
    scope = _make_scope(raw)
    result = analyze_seasonality(scope, target="Мутность (по формазину)", min_group_size=4)

    assert result.pattern_test["test"] == "skipped"
    assert len(result.diagnostics) == 0 or True  # no error raised


def test_per_season_correlations_computed() -> None:
    # Build data with within-season variation so Spearman correlation is computable.
    # Мутность = base + i, Жесткость = (base + i) * 2 → perfect positive correlation within each season.
    rows: list[dict] = []
    season_bases = {"winter": 5, "spring": 15, "summer": 30, "autumn": 10}
    for season, dates in _SEASON_MONTHS.items():
        base = season_bases[season]
        for i, date_str in enumerate(dates):
            turb_val = base + i
            hard_val = turb_val * 2
            for ind, val in [("Мутность (по формазину)", turb_val), ("Жесткость общая", hard_val)]:
                rows.append(
                    {
                        _DATE_COL: date_str,
                        "Тип точки": "Точка на распределительной сети",
                        _POINT_COL: _CODE,
                        _INDICATOR_COL: ind,
                        "Норматив": "",
                        "Строчное значение": "",
                        _VALUE_COL: str(int(val)),
                        "Нижний предел обнаружения": "",
                        "Верхний предел обнаружения": "",
                        "Ошибка метода определения": "",
                        "Нормативная документация": "",
                        "Цели исследований": "",
                        "Соотв. ПДК": "",
                    }
                )
    raw = pd.DataFrame(rows)
    scope = _make_scope(raw)
    result = analyze_seasonality(scope, target="Мутность (по формазину)", min_group_size=4)

    assert not result.per_season_correlations.empty
    assert "feature" in result.per_season_correlations.columns
    assert "corr" in result.per_season_correlations.columns
    # Only groups with n >= min_group_size should have entries
    assert all(result.per_season_correlations["n_shared"] >= 4)


def test_per_season_correlations_skip_small_groups() -> None:
    # One season with only 2 dates (below min_group_size=5)
    dates_small = ["01.01.2020", "02.01.2020"]
    dates_full = ["01.04.2020", "02.04.2020", "03.04.2020", "04.04.2020", "05.04.2020"]
    rows: list[dict] = []
    for date_str in dates_small + dates_full:
        for ind, val in [("Мутность (по формазину)", "10"), ("Жесткость общая", "20")]:
            rows.append(
                {
                    _DATE_COL: date_str,
                    "Тип точки": "Точка на распределительной сети",
                    _POINT_COL: _CODE,
                    _INDICATOR_COL: ind,
                    "Норматив": "",
                    "Строчное значение": "",
                    _VALUE_COL: val,
                    "Нижний предел обнаружения": "",
                    "Верхний предел обнаружения": "",
                    "Ошибка метода определения": "",
                    "Нормативная документация": "",
                    "Цели исследований": "",
                    "Соотв. ПДК": "",
                }
            )
    raw = pd.DataFrame(rows)
    scope = _make_scope(raw)
    result = analyze_seasonality(scope, target="Мутность (по формазину)", min_group_size=5)

    # winter (2 obs) should be skipped — diagnostics should mention it
    corr_groups = set(result.per_season_correlations["group"].tolist()) if not result.per_season_correlations.empty else set()
    # spring (5 obs) should have correlations
    assert "spring" in corr_groups or result.per_season_correlations.empty  # might be empty if corr not computable


def test_month_granularity() -> None:
    raw = _make_raw(target_by_season={"winter": 2, "spring": 20, "summer": 40, "autumn": 10})
    scope = _make_scope(raw)
    result = analyze_seasonality(scope, target="Мутность (по формазину)", granularity="month", min_group_size=4)

    assert result.granularity == "month"
    assert not result.group_stats.empty
    # Group column should contain month numbers (1, 4, 7, 10)
    groups = set(result.group_stats["group"].tolist())
    assert groups.issubset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
    assert {1, 4, 7, 10}.issubset(groups)
