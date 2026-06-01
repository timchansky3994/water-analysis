"""Input readers and schema helpers."""

from water_analysis.io.csv_reader import read_csv_table
from water_analysis.io.xlsx_reader import read_xlsx_table

__all__ = ["read_csv_table", "read_xlsx_table"]
