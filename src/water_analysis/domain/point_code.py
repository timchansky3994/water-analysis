"""Sampling point code parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PointCode:
    """Structured representation of a sampling point identifier."""

    full_point_code: str
    oktmo: str
    point_type_code: str
    point_number: str


def parse_point_code(raw_code: str | None) -> PointCode | None:
    """Parse a full sampling point code into its component parts."""
    if raw_code is None:
        return None

    cleaned = str(raw_code).strip()
    if not cleaned:
        return None

    parts = [part.strip() for part in cleaned.split(".")]
    if len(parts) != 3 or any(not part for part in parts):
        return None

    return PointCode(
        full_point_code=".".join(parts),
        oktmo=parts[0],
        point_type_code=parts[1],
        point_number=parts[2],
    )
