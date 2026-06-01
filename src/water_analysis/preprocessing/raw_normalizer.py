"""Raw-table normalization into a canonical long representation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from water_analysis.domain.point_code import parse_point_code
from water_analysis.io.schemas import DRINKING_WATER_POINT_TYPES, resolve_raw_schema
from water_analysis.preprocessing.censoring import parse_censored_value

if TYPE_CHECKING:
    from water_analysis.io.source_profiles import SourceProfile

LOGGER = logging.getLogger(__name__)

# Full set of optional source-field keys.  Any field absent from the resolved
# schema (either because the profile doesn't define it or the file doesn't have
# it) is filled with empty strings and logged once at WARNING level.
_ALL_OPTIONAL_SOURCE_FIELDS: frozenset[str] = frozenset(
    {
        "PointType_Name",
        "Normative",
        "RawValueText",
        "DetectionLimitLowerText",
        "DetectionLimitUpperText",
        "MethodErrorText",
        "NormativeDocument",
        "ResearchGoal",
        "ComplianceText",
    }
)


def _normalize_text(value: Any) -> str:
    """Normalize a text cell while preserving empties."""
    if value is None:
        return ""
    return str(value).strip()


def _derive_season(month_value: int | None) -> str | None:
    """Map a month number to a season label."""
    if month_value is None or pd.isna(month_value):
        return None
    if month_value in (12, 1, 2):
        return "winter"
    if month_value in (3, 4, 5):
        return "spring"
    if month_value in (6, 7, 8):
        return "summer"
    return "autumn"


def _parse_sample_dates(series: pd.Series) -> pd.Series:
    """Parse common CSV and Excel date representations without assuming one export format."""
    parsed = pd.to_datetime(series, format="%d.%m.%Y", errors="coerce")
    missing = parsed.isna() & series.astype(str).str.strip().ne("")
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    missing = parsed.isna() & series.astype(str).str.strip().ne("")
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna() & series.astype(str).str.strip().ne("")
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], errors="coerce", dayfirst=True)
    return parsed


def _col(raw_df: pd.DataFrame, schema: dict[str, str | None], key: str) -> pd.Series:
    """Return the mapped column as normalized text, or an empty-string series if absent."""
    col_name = schema.get(key)
    if col_name is None:
        return pd.Series("", index=raw_df.index)
    return raw_df[col_name].map(_normalize_text)


def normalize_raw_measurements(
    raw_df: pd.DataFrame,
    source_profile: "SourceProfile | None" = None,
) -> pd.DataFrame:
    """Normalize a raw laboratory export into canonical long-format rows."""
    schema = resolve_raw_schema(raw_df.columns, profile=source_profile)

    absent_optional = sorted(f for f in _ALL_OPTIONAL_SOURCE_FIELDS if schema.get(f) is None)
    if absent_optional:
        LOGGER.warning(
            "Optional source fields absent (filled with empty strings): %s",
            ", ".join(absent_optional),
        )

    normalized = pd.DataFrame(
        {
            "SampleDateRaw": _col(raw_df, schema, "SampleDate"),
            "PointType_Name": _col(raw_df, schema, "PointType_Name"),
            "FullPointCode": _col(raw_df, schema, "FullPointCode"),
            "Indicator": _col(raw_df, schema, "Indicator"),
            "Normative": _col(raw_df, schema, "Normative"),
            "RawValueText": _col(raw_df, schema, "RawValueText"),
            "ResultValueText": _col(raw_df, schema, "ResultValueText"),
            "DetectionLimitLowerText": _col(raw_df, schema, "DetectionLimitLowerText"),
            "DetectionLimitUpperText": _col(raw_df, schema, "DetectionLimitUpperText"),
            "MethodErrorText": _col(raw_df, schema, "MethodErrorText"),
            "NormativeDocument": _col(raw_df, schema, "NormativeDocument"),
            "ResearchGoal": _col(raw_df, schema, "ResearchGoal"),
            "ComplianceText": _col(raw_df, schema, "ComplianceText"),
        }
    )

    point_codes = normalized["FullPointCode"].map(parse_point_code)
    normalized["OKTMO"] = point_codes.map(lambda code: code.oktmo if code else None)
    normalized["PointType_Code"] = point_codes.map(lambda code: code.point_type_code if code else None)
    normalized["PointNumber"] = point_codes.map(lambda code: code.point_number if code else None)

    normalized["SampleDate"] = _parse_sample_dates(normalized["SampleDateRaw"])

    parsed_measurements = normalized.apply(
        lambda row: parse_censored_value(
            row["ResultValueText"] or row["RawValueText"],
            detection_limit_lower=row["DetectionLimitLowerText"],
            detection_limit_upper=row["DetectionLimitUpperText"],
        ).to_record(),
        axis=1,
        result_type="expand",
    )
    parsed_measurements = parsed_measurements.rename(
        columns={
            "numeric_approx": "Value_Approx",
            "parse_status": "Value_ParseStatus",
            "censoring_type": "CensoringType",
            "censoring_qualifier": "CensoringQualifier",
            "censoring_lower_bound": "CensoringLowerBound",
            "censoring_upper_bound": "CensoringUpperBound",
            "detection_limit_lower": "DetectionLimitLower",
            "detection_limit_upper": "DetectionLimitUpper",
            "is_censored": "IsCensored",
            "has_detection_limit_metadata": "HasDetectionLimitMetadata",
            "parse_issue": "ParseIssue",
        }
    )
    normalized = pd.concat([normalized, parsed_measurements.drop(columns=["raw_value"])], axis=1)

    normalized["year"] = normalized["SampleDate"].dt.year.astype("Int64")
    normalized["month"] = normalized["SampleDate"].dt.month.astype("Int64")
    normalized["quarter"] = normalized["SampleDate"].dt.quarter.astype("Int64")
    normalized["season"] = normalized["month"].map(_derive_season)
    normalized["drinking_water"] = normalized["PointType_Code"].isin(DRINKING_WATER_POINT_TYPES)

    return normalized
