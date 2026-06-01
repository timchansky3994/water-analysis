import pandas as pd

from water_analysis.profiling.passport import _build_constant_series_table


def test_constant_series_table_empty_when_pivot_has_no_indicator_columns() -> None:
    pivot_df = pd.DataFrame(columns=["SampleDate", "FullPointCode", "OKTMO", "PointType_Code", "PointNumber"])

    result = _build_constant_series_table(pivot_df, near_constant_threshold=0.95)

    assert result.empty
    assert list(result.columns) == ["Indicator", "status", "n_observations", "n_unique", "dominant_share"]
