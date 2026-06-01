"""Safe pivot builder with explicit aggregation levels."""

from __future__ import annotations

import pandas as pd

from water_analysis.io.schemas import PIVOT_AGGREGATION_LEVELS

SAMPLE_POINT_INDEX = [
    "SampleDate",
    "FullPointCode",
    "OKTMO",
    "PointType_Code",
    "PointNumber",
    "year",
    "month",
    "quarter",
    "season",
    "drinking_water",
]

POINT_TYPE_INDEX = [
    "SampleDate",
    "OKTMO",
    "PointType_Code",
    "year",
    "month",
    "quarter",
    "season",
    "drinking_water",
]

OKTMO_INDEX = [
    "SampleDate",
    "OKTMO",
    "year",
    "month",
    "quarter",
    "season",
]

INDEX_BY_LEVEL = {
    "sample_point_level": SAMPLE_POINT_INDEX,
    "point_type_level": POINT_TYPE_INDEX,
    "oktmo_level": OKTMO_INDEX,
}


def build_indicator_pivot(
    long_df: pd.DataFrame,
    *,
    aggregation_level: str = "sample_point_level",
    value_column: str = "Value_Approx",
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Build a wide indicator pivot at an explicit aggregation scope."""
    if aggregation_level not in PIVOT_AGGREGATION_LEVELS:
        raise ValueError(f"Unsupported aggregation level: {aggregation_level}")

    index_columns = INDEX_BY_LEVEL[aggregation_level]
    required_columns = set(index_columns + ["Indicator", value_column])
    missing = required_columns.difference(long_df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required long-format columns for pivot: {missing_text}")

    usable = long_df.dropna(subset=["SampleDate", "Indicator"]).copy()
    pivot = usable.pivot_table(
        index=index_columns,
        columns="Indicator",
        values=value_column,
        aggfunc=aggfunc,
        observed=False,
    ).reset_index()
    pivot.columns.name = None
    return pivot
