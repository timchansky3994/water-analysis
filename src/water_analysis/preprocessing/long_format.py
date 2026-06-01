"""Canonical long-format ingestion entry points."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from water_analysis.io.csv_reader import read_csv_table
from water_analysis.io.schemas import CANONICAL_LONG_COLUMNS
from water_analysis.io.xlsx_reader import read_xlsx_table
from water_analysis.preprocessing.raw_normalizer import normalize_raw_measurements

if TYPE_CHECKING:
    from water_analysis.io.source_profiles import SourceProfile


def read_source_table(
    path: str | Path,
    *,
    nrows: int | None = None,
    source_profile: "SourceProfile | None" = None,
) -> pd.DataFrame:
    """Read a source table from CSV or XLSX.

    The source_profile parameter is accepted for API consistency but does not
    affect file reading — pass it to build_canonical_long_format instead.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return read_csv_table(file_path, nrows=nrows)
    if suffix in {".xlsx", ".xlsm"}:
        return read_xlsx_table(file_path, nrows=nrows, source_profile=source_profile)
    raise ValueError(f"Unsupported source format: {file_path.suffix}")


def build_canonical_long_format(
    raw_df: pd.DataFrame,
    source_profile: "SourceProfile | None" = None,
) -> pd.DataFrame:
    """Build a canonical long-format dataframe with stable column ordering."""
    normalized = normalize_raw_measurements(raw_df, source_profile=source_profile)
    available_columns = [column for column in CANONICAL_LONG_COLUMNS if column in normalized.columns]
    return normalized.loc[:, available_columns].copy()
