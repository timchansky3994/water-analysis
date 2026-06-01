"""XLSX reader for raw laboratory exports."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from water_analysis.io.schemas import resolve_raw_schema

if TYPE_CHECKING:
    from water_analysis.io.source_profiles import SourceProfile


def _clean_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Drop Excel-only empty rows/columns and normalize header text."""
    cleaned = dataframe.dropna(axis=0, how="all").dropna(axis=1, how="all").copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    cleaned = cleaned.loc[:, [column and not column.startswith("Unnamed:") for column in cleaned.columns]]
    return cleaned.fillna("")


def _has_raw_schema(dataframe: pd.DataFrame, profile: "SourceProfile | None" = None) -> bool:
    """Return whether a dataframe has recognizable raw export columns.

    When a profile is given, only that profile is checked.
    When profile is None, autodetection across all available profiles is used
    so that regional formats (secondary etc.) are recognized without a prior hint.
    """
    if profile is not None:
        try:
            resolve_raw_schema(dataframe.columns, profile=profile)
            return True
        except ValueError:
            return False

    from water_analysis.io.source_profiles import autodetect_source_profile

    return autodetect_source_profile(dataframe.columns) is not None


def _read_sheet_with_header_detection(
    file_path: Path,
    sheet_name: int | str,
    *,
    nrows: int | None,
    profile: "SourceProfile | None" = None,
) -> pd.DataFrame:
    """Read one sheet, scanning for a shifted header row when needed."""
    direct = _clean_table(
        pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            dtype=str,
            keep_default_na=False,
            nrows=nrows,
        )
    )
    if _has_raw_schema(direct, profile):
        return direct

    preview = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        dtype=str,
        keep_default_na=False,
        header=None,
        nrows=25,
    )
    for row_index in range(len(preview)):
        candidate_columns = [str(value).strip() for value in preview.iloc[row_index].tolist()]
        candidate = pd.DataFrame(columns=candidate_columns)
        if _has_raw_schema(candidate, profile):
            return _clean_table(
                pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    dtype=str,
                    keep_default_na=False,
                    header=row_index,
                    nrows=nrows,
                )
            )
    return direct


def read_xlsx_table(
    path: str | Path,
    *,
    sheet_name: int | str = 0,
    nrows: int | None = None,
    source_profile: "SourceProfile | None" = None,
) -> pd.DataFrame:
    """Read an Excel file as a string-preserving raw table."""
    file_path = Path(path)
    # Use context manager so the file handle is closed before any subsequent unlink on Windows.
    with pd.ExcelFile(file_path) as excel_file:
        requested_sheets: list[int | str] = (
            list(excel_file.sheet_names) if sheet_name == 0 else [sheet_name]
        )
        available = ", ".join(str(name) for name in excel_file.sheet_names)

    for current_sheet in requested_sheets:
        table = _read_sheet_with_header_detection(file_path, current_sheet, nrows=nrows, profile=source_profile)
        if _has_raw_schema(table, source_profile):
            return table

    raise ValueError(
        "Excel file was read, but required laboratory export columns were not recognized. "
        f"Available sheets: {available}. Check that the sheet contains the raw table header."
    )
