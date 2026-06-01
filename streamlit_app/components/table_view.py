"""Shared helpers for displaying and downloading result tables in the UI.

Tables are persisted in two forms: a machine-readable CSV with stable English
column names, and a human-readable XLSX with Russian headers. To keep what the
user sees on screen as close as possible to what lands in the files, the UI
shows tables with the same Russian column labels used in the XLSX export — only
the headers are translated, the data itself is never altered.

Download is offered in both formats (CSV and XLSX) side by side.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from water_analysis.reporting.xlsx_export import dataframe_to_xlsx_bytes, humanize_columns

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Numbers whose magnitude falls outside this range are shown on screen in
# scientific notation, so that e.g. 0.0000000001 or 1200000 do not stretch the
# column with a long run of zeros. This affects only the on-screen rendering;
# downloaded CSV/XLSX tables keep the original values untouched.
_SMALL_MAGNITUDE = 1e-4
_LARGE_MAGNITUDE = 1e6


def _format_number(value: object) -> str:
    """Render one numeric cell, using scientific notation for extreme values."""
    if value is None or pd.isna(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == 0:
        return "0"
    magnitude = abs(number)
    if magnitude < _SMALL_MAGNITUDE or magnitude >= _LARGE_MAGNITUDE:
        return f"{number:.3e}"
    return f"{number:.6g}"


def render_table(dataframe: pd.DataFrame, *, max_rows: int | None = None, **kwargs) -> None:
    """Display a dataframe with human-readable Russian column headers.

    Float columns are rendered with scientific notation for very small/large
    magnitudes (display only — the underlying data and downloads are unchanged).
    """
    display = humanize_columns(dataframe)
    if max_rows is not None:
        display = display.head(max_rows)
    kwargs.setdefault("width", "stretch")

    float_columns = list(display.select_dtypes(include="floating").columns)
    if float_columns:
        styled = display.style.format(
            {column: _format_number for column in float_columns}
        )
        st.dataframe(styled, **kwargs)
    else:
        st.dataframe(display, **kwargs)


def download_buttons(
    dataframe: pd.DataFrame,
    *,
    file_stem: str,
    table_name: str,
    key: str,
) -> None:
    """Render side-by-side CSV and XLSX download buttons for an in-memory table.

    ``table_name`` selects the XLSX sheet title (see ``SHEET_NAMES_RU``).
    """
    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            f"Скачать {file_stem}.csv",
            data=dataframe.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{file_stem}.csv",
            mime="text/csv",
            key=f"{key}_csv",
        )
    with col_xlsx:
        st.download_button(
            f"Скачать {file_stem}.xlsx",
            data=dataframe_to_xlsx_bytes(dataframe, table_name=table_name),
            file_name=f"{file_stem}.xlsx",
            mime=_XLSX_MIME,
            key=f"{key}_xlsx",
        )


def download_buttons_from_files(*, csv_path: Path, xlsx_path: Path, key: str) -> None:
    """Render CSV and XLSX download buttons backed by files already on disk."""
    col_csv, col_xlsx = st.columns(2)
    if csv_path.exists():
        with col_csv:
            st.download_button(
                f"Скачать {csv_path.name}",
                data=csv_path.read_bytes(),
                file_name=csv_path.name,
                mime="text/csv",
                key=f"{key}_csv",
            )
    if xlsx_path.exists():
        with col_xlsx:
            st.download_button(
                f"Скачать {xlsx_path.name}",
                data=xlsx_path.read_bytes(),
                file_name=xlsx_path.name,
                mime=_XLSX_MIME,
                key=f"{key}_xlsx",
            )
