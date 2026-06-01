"""CSV reader for raw laboratory exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from water_analysis.io.sniffing import sniff_csv_options


def read_csv_table(
    path: str | Path,
    *,
    encoding: str | None = None,
    delimiter: str | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read a CSV file as a string-preserving raw table."""
    file_path = Path(path)
    sniffed = sniff_csv_options(file_path)
    selected_encoding = encoding or sniffed.encoding
    selected_delimiter = delimiter or sniffed.delimiter

    dataframe = pd.read_csv(
        file_path,
        sep=selected_delimiter,
        encoding=selected_encoding,
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )
    dataframe.columns = [column.strip() for column in dataframe.columns]
    return dataframe
