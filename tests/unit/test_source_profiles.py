"""Unit tests for source profile loading and schema resolution."""

from __future__ import annotations

import logging

import pytest

from water_analysis.io.schemas import REQUIRED_RAW_FIELDS, resolve_raw_schema
from water_analysis.io.source_profiles import (
    autodetect_source_profile,
    list_available_profiles,
    load_source_profile,
)


def test_load_default_profile_succeeds() -> None:
    profile = load_source_profile("_default")
    assert profile.name == "default"
    assert "SampleDate" in profile.column_aliases
    assert "FullPointCode" in profile.column_aliases
    assert "Indicator" in profile.column_aliases
    assert "ResultValueText" in profile.column_aliases


def test_load_secondary_profile_succeeds() -> None:
    profile = load_source_profile("secondary")
    assert profile.name == "secondary"
    assert "SampleDate" in profile.column_aliases
    assert "FullPointCode" in profile.column_aliases
    assert "Indicator" in profile.column_aliases
    assert "ResultValueText" in profile.column_aliases
    # These optional fields should be absent from the secondary profile aliases
    assert "RawValueText" not in profile.column_aliases
    assert "DetectionLimitUpperText" not in profile.column_aliases
    assert "ResearchGoal" not in profile.column_aliases
    assert "ComplianceText" not in profile.column_aliases


def test_list_available_profiles_returns_both() -> None:
    profiles = list_available_profiles()
    names = {p.name for p in profiles}
    assert "default" in names
    assert "secondary" in names


def test_autodetect_picks_secondary_for_secondary_columns() -> None:
    secondary_columns = [
        "Дата",
        "Тип точки",
        "Код точки",
        "Гигиенический показатель",
        "Норматив",
        "Результат исследования",
        "НПО",
        "Ошибка метода определения",
        "Нормативная документация",
    ]
    profile = autodetect_source_profile(secondary_columns)
    assert profile is not None
    assert profile.name == "secondary"


def test_autodetect_picks_default_for_default_columns() -> None:
    default_columns = [
        "Дата проведения исследования",
        "Тип точки",
        "Код точки",
        "Гигиенический показатель",
        "Норматив",
        "Строчное значение",
        "Результат исследования",
        "Нижний предел обнаружения",
        "Верхний предел обнаружения",
        "Ошибка метода определения",
        "Нормативная документация",
        "Цели исследований",
        "Соотв. ПДК",
    ]
    profile = autodetect_source_profile(default_columns)
    assert profile is not None
    assert profile.name == "default"


def test_autodetect_returns_none_for_unknown_columns() -> None:
    unknown_columns = ["Column_A", "Column_B", "Column_C"]
    profile = autodetect_source_profile(unknown_columns)
    assert profile is None


def test_resolve_schema_fails_clearly_when_required_field_missing() -> None:
    # FullPointCode is absent — should raise ValueError with useful message
    columns = [
        "Дата проведения исследования",
        "Тип точки",
        # "Код точки" is intentionally missing
        "Гигиенический показатель",
        "Результат исследования",
    ]
    with pytest.raises(ValueError) as exc_info:
        resolve_raw_schema(columns)
    error_msg = str(exc_info.value)
    assert "FullPointCode" in error_msg
    # The error message should list available file columns so the user can debug
    assert "Дата проведения исследования" in error_msg or "File columns" in error_msg


def test_resolve_schema_warns_for_missing_optional_fields(caplog: pytest.LogCaptureFixture) -> None:
    # Provide only the four required fields; all optional fields are absent
    columns = [
        "Дата проведения исследования",
        "Код точки",
        "Гигиенический показатель",
        "Результат исследования",
    ]
    with caplog.at_level(logging.WARNING, logger="water_analysis.io.schemas"):
        result = resolve_raw_schema(columns)

    # Required fields must be resolved
    assert result["SampleDate"] is not None
    assert result["FullPointCode"] is not None
    assert result["Indicator"] is not None
    assert result["ResultValueText"] is not None

    # Optional fields that are in the default profile but absent from the file must be None
    assert result["RawValueText"] is None
    assert result["DetectionLimitLowerText"] is None

    # A WARNING must have been emitted about missing optional fields
    assert any("Optional columns" in record.message for record in caplog.records)


def test_normalize_warns_for_absent_optional_fields(caplog: pytest.LogCaptureFixture) -> None:
    """normalize_raw_measurements emits a WARNING for optional fields not in the source file."""
    import pandas as pd

    from water_analysis.preprocessing.raw_normalizer import normalize_raw_measurements
    from water_analysis.io.source_profiles import load_source_profile

    secondary_profile = load_source_profile("secondary")
    raw_df = pd.DataFrame(
        [
            {
                "Дата": "01.01.2020",
                "Тип точки": "Водопроводная сеть",
                "Код точки": "00000000001.10110.0010",
                "Гигиенический показатель": "Цветность",
                "Норматив": "",
                "Результат исследования": "10",
                "НПО": "",
                "Ошибка метода определения": "",
                "Нормативная документация": "",
            }
        ]
    )

    with caplog.at_level(logging.WARNING, logger="water_analysis.preprocessing.raw_normalizer"):
        normalize_raw_measurements(raw_df, source_profile=secondary_profile)

    # Fields absent from the secondary format (RawValueText, DetectionLimitUpperText, etc.)
    # must trigger a WARNING.
    assert any("Optional source fields absent" in record.message for record in caplog.records)


def test_resolve_schema_required_fields_constant_is_correct() -> None:
    assert REQUIRED_RAW_FIELDS == {"SampleDate", "FullPointCode", "Indicator", "ResultValueText"}
