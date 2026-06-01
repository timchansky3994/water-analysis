"""Parsing utilities for censored and thresholded laboratory values."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

MISSING_TOKENS = {"", "-", "—", "nan", "none", "null"}
LEFT_QUALIFIERS = ("<=", "<", "≤", "менее", "меньше")
RIGHT_QUALIFIERS = (">=", ">", "≥", "более", "больше")
INTERVAL_PATTERN = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*[-–]\s*([+-]?\d+(?:[.,]\d+)?)\s*$")
NUMERIC_PATTERN = re.compile(r"^[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True)
class CensoringParseResult:
    """Structured parsing result for a single measurement value."""

    raw_value: str
    numeric_approx: float | None
    parse_status: str
    censoring_type: str
    censoring_qualifier: str | None
    censoring_lower_bound: float | None
    censoring_upper_bound: float | None
    detection_limit_lower: float | None
    detection_limit_upper: float | None
    is_censored: bool
    has_detection_limit_metadata: bool
    parse_issue: str | None

    def to_record(self) -> dict[str, object]:
        """Convert the result to a flat dictionary for tabular use."""
        return asdict(self)


def parse_decimal_token(raw_value: str | None) -> float | None:
    """Parse a decimal token while accepting comma decimals."""
    if raw_value is None:
        return None

    cleaned = str(raw_value).strip().replace(" ", "")
    if not cleaned or cleaned.lower() in MISSING_TOKENS:
        return None

    cleaned = cleaned.replace(",", ".")
    if not NUMERIC_PATTERN.match(cleaned):
        return None

    value = float(cleaned)
    if math.isnan(value):
        return None
    return value


def parse_censored_value(
    raw_value: str | None,
    *,
    detection_limit_lower: str | None = None,
    detection_limit_upper: str | None = None,
) -> CensoringParseResult:
    """Parse an observed value and preserve censoring metadata."""
    raw_text = "" if raw_value is None else str(raw_value).strip()
    lower_limit_value = parse_decimal_token(detection_limit_lower)
    upper_limit_value = parse_decimal_token(detection_limit_upper)
    has_limit_metadata = lower_limit_value is not None or upper_limit_value is not None

    if not raw_text or raw_text.lower() in MISSING_TOKENS:
        return CensoringParseResult(
            raw_value=raw_text,
            numeric_approx=None,
            parse_status="missing",
            censoring_type="missing",
            censoring_qualifier=None,
            censoring_lower_bound=None,
            censoring_upper_bound=None,
            detection_limit_lower=lower_limit_value,
            detection_limit_upper=upper_limit_value,
            is_censored=False,
            has_detection_limit_metadata=has_limit_metadata,
            parse_issue=None,
        )

    numeric_exact = parse_decimal_token(raw_text)
    if numeric_exact is not None:
        return CensoringParseResult(
            raw_value=raw_text,
            numeric_approx=numeric_exact,
            parse_status="parsed",
            censoring_type="exact",
            censoring_qualifier=None,
            censoring_lower_bound=numeric_exact,
            censoring_upper_bound=numeric_exact,
            detection_limit_lower=lower_limit_value,
            detection_limit_upper=upper_limit_value,
            is_censored=False,
            has_detection_limit_metadata=has_limit_metadata,
            parse_issue=None,
        )

    lowered_text = raw_text.lower()
    for qualifier in LEFT_QUALIFIERS:
        if lowered_text.startswith(qualifier):
            bound = parse_decimal_token(raw_text[len(qualifier) :])
            if bound is None:
                break
            return CensoringParseResult(
                raw_value=raw_text,
                numeric_approx=bound / 2.0,
                parse_status="parsed",
                censoring_type="left_censored",
                censoring_qualifier=qualifier,
                censoring_lower_bound=0.0,
                censoring_upper_bound=bound,
                detection_limit_lower=lower_limit_value,
                detection_limit_upper=upper_limit_value,
                is_censored=True,
                has_detection_limit_metadata=has_limit_metadata,
                parse_issue=None,
            )

    for qualifier in RIGHT_QUALIFIERS:
        if lowered_text.startswith(qualifier):
            bound = parse_decimal_token(raw_text[len(qualifier) :])
            if bound is None:
                break
            return CensoringParseResult(
                raw_value=raw_text,
                numeric_approx=bound,
                parse_status="parsed",
                censoring_type="right_censored",
                censoring_qualifier=qualifier,
                censoring_lower_bound=bound,
                censoring_upper_bound=None,
                detection_limit_lower=lower_limit_value,
                detection_limit_upper=upper_limit_value,
                is_censored=True,
                has_detection_limit_metadata=has_limit_metadata,
                parse_issue=None,
            )

    interval_match = INTERVAL_PATTERN.match(raw_text)
    if interval_match:
        lower_bound = parse_decimal_token(interval_match.group(1))
        upper_bound = parse_decimal_token(interval_match.group(2))
        if lower_bound is not None and upper_bound is not None:
            return CensoringParseResult(
                raw_value=raw_text,
                numeric_approx=(lower_bound + upper_bound) / 2.0,
                parse_status="parsed",
                censoring_type="interval",
                censoring_qualifier="interval",
                censoring_lower_bound=lower_bound,
                censoring_upper_bound=upper_bound,
                detection_limit_lower=lower_limit_value,
                detection_limit_upper=upper_limit_value,
                is_censored=True,
                has_detection_limit_metadata=has_limit_metadata,
                parse_issue=None,
            )

    return CensoringParseResult(
        raw_value=raw_text,
        numeric_approx=None,
        parse_status="unparsed",
        censoring_type="unparsed",
        censoring_qualifier=None,
        censoring_lower_bound=None,
        censoring_upper_bound=None,
        detection_limit_lower=lower_limit_value,
        detection_limit_upper=upper_limit_value,
        is_censored=False,
        has_detection_limit_metadata=has_limit_metadata,
        parse_issue="unsupported_value_format",
    )
