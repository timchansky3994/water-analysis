"""Schemas and dataset contracts for raw and canonical tables."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from water_analysis.io.source_profiles import SourceProfile

LOGGER = logging.getLogger(__name__)

# Canonical names for the four fields that every source format must supply.
REQUIRED_RAW_FIELDS: frozenset[str] = frozenset({"SampleDate", "FullPointCode", "Indicator", "ResultValueText"})

# Legacy constant kept for backwards compatibility with existing callers.
# New code should load a SourceProfile instead.
RAW_SCHEMA_ALIASES: dict[str, tuple[str, ...]] = {
    "SampleDate": ("Дата проведения исследования", "Date"),
    "PointType_Name": ("Тип точки", "PointType_Name"),
    "FullPointCode": ("Код точки", "Code"),
    "Indicator": ("Гигиенический показатель", "Indicator"),
    "Normative": ("Норматив", "Normative"),
    "RawValueText": ("Строчное значение", "Value_Text"),
    "ResultValueText": ("Результат исследования", "Value"),
    "DetectionLimitLowerText": ("Нижний предел обнаружения", "DetectionLimitLower"),
    "DetectionLimitUpperText": ("Верхний предел обнаружения", "DetectionLimitUpper"),
    "MethodErrorText": ("Ошибка метода определения", "MethodError"),
    "NormativeDocument": ("Нормативная документация", "NormativeDocument"),
    "ResearchGoal": ("Цели исследований", "ResearchGoal"),
    "ComplianceText": ("Соотв. ПДК", "Compliance"),
}

REQUIRED_TARGETS: tuple[str, ...] = (
    "Жесткость общая",
    "Цветность",
    "Мутность (по формазину)",
    "Перманганатная окисляемость",
    "Химическое потребление кислорода (ХПК)",
)

DRINKING_WATER_POINT_TYPES: frozenset[str] = frozenset({"10110", "10150"})

CANONICAL_LONG_COLUMNS: tuple[str, ...] = (
    "SampleDate",
    "FullPointCode",
    "OKTMO",
    "PointType_Code",
    "PointNumber",
    "PointType_Name",
    "Indicator",
    "Normative",
    "RawValueText",
    "ResultValueText",
    "Value_Approx",
    "Value_ParseStatus",
    "CensoringType",
    "CensoringQualifier",
    "CensoringLowerBound",
    "CensoringUpperBound",
    "DetectionLimitLowerText",
    "DetectionLimitLower",
    "DetectionLimitUpperText",
    "DetectionLimitUpper",
    "MethodErrorText",
    "NormativeDocument",
    "ResearchGoal",
    "ComplianceText",
    "IsCensored",
    "HasDetectionLimitMetadata",
    "ParseIssue",
    "year",
    "month",
    "quarter",
    "season",
    "drinking_water",
)

SCOPE_NAMES: tuple[str, ...] = (
    "global",
    "oktmo",
    "oktmo_point_type",
    "drinking_water_combined",
    "point",
)

PIVOT_AGGREGATION_LEVELS: tuple[str, ...] = (
    "sample_point_level",
    "point_type_level",
    "oktmo_level",
)

LONG_FORMAT_METADATA_COLUMNS: tuple[str, ...] = (
    "SampleDate",
    "FullPointCode",
    "OKTMO",
    "PointType_Code",
    "PointNumber",
    "PointType_Name",
    "Normative",
    "RawValueText",
    "ResultValueText",
    "Value_Approx",
    "Value_ParseStatus",
    "CensoringType",
    "CensoringQualifier",
    "CensoringLowerBound",
    "CensoringUpperBound",
    "DetectionLimitLowerText",
    "DetectionLimitLower",
    "DetectionLimitUpperText",
    "DetectionLimitUpper",
    "MethodErrorText",
    "NormativeDocument",
    "ResearchGoal",
    "ComplianceText",
    "IsCensored",
    "HasDetectionLimitMetadata",
    "ParseIssue",
    "year",
    "month",
    "quarter",
    "season",
    "drinking_water",
)


@dataclass(frozen=True)
class RequiredTargetDiagnostic:
    """Coverage status for a mandatory target indicator."""

    indicator: str
    status: str
    observed_rows: int


def resolve_raw_schema(
    columns: Sequence[str],
    profile: "SourceProfile | None" = None,
) -> dict[str, str | None]:
    """Resolve raw dataset column names to canonical input names.

    Returns a mapping from canonical field names to raw column names.
    Required fields missing from the file raise ValueError with a clear message.
    Optional fields missing from the file map to None and emit a WARNING.
    """
    if profile is None:
        from water_analysis.io.source_profiles import load_source_profile

        profile = load_source_profile("_default")

    column_lookup = {col.strip(): col for col in columns}
    resolved: dict[str, str | None] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []

    for canonical_name, aliases in profile.column_aliases.items():
        source_name = next((alias for alias in aliases if alias in column_lookup), None)
        if source_name is not None:
            resolved[canonical_name] = column_lookup[source_name]
        else:
            resolved[canonical_name] = None
            if canonical_name in REQUIRED_RAW_FIELDS:
                missing_required.append(canonical_name)
            else:
                missing_optional.append(canonical_name)

    if missing_required:
        available_columns = list(columns)
        raise ValueError(
            f"Missing required raw columns: {', '.join(missing_required)}. "
            f"File columns: {available_columns}"
        )

    if missing_optional:
        LOGGER.warning(
            "Optional columns absent in source file (will be filled with empty strings): %s",
            ", ".join(missing_optional),
        )

    return resolved


def build_required_target_diagnostics(indicators: Iterable[str]) -> list[RequiredTargetDiagnostic]:
    """Build explicit present/missing diagnostics for mandatory targets."""
    counts = Counter(indicator for indicator in indicators if indicator)
    diagnostics: list[RequiredTargetDiagnostic] = []

    for target in REQUIRED_TARGETS:
        observed_rows = counts.get(target, 0)
        diagnostics.append(
            RequiredTargetDiagnostic(
                indicator=target,
                status="present" if observed_rows > 0 else "missing",
                observed_rows=observed_rows,
            )
        )

    return diagnostics
