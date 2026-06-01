from pathlib import Path

from water_analysis.io.schemas import REQUIRED_RAW_FIELDS
from water_analysis.preprocessing.long_format import build_canonical_long_format, read_source_table


DATASET_PATH = Path("data/raw/main.csv")

# Required fields in the canonical long-format output (derived from REQUIRED_RAW_FIELDS
# plus the point-code sub-components always produced from FullPointCode).
REQUIRED_LONG_COLUMNS: frozenset[str] = frozenset(
    {"SampleDate", "FullPointCode", "OKTMO", "PointType_Code", "PointNumber", "Indicator", "ResultValueText"}
)


def test_primary_dataset_contract_is_preserved() -> None:
    raw_df = read_source_table(DATASET_PATH)
    long_df = build_canonical_long_format(raw_df)

    assert len(raw_df) > 100_000
    assert len(long_df) == len(raw_df)

    # Required columns must always be present — optional ones may be absent for some formats.
    assert REQUIRED_LONG_COLUMNS.issubset(long_df.columns)
    assert long_df["FullPointCode"].notna().all()
    assert long_df["OKTMO"].notna().all()
    assert long_df["PointType_Code"].notna().all()
    assert long_df["PointNumber"].notna().all()
    assert long_df["SampleDate"].notna().sum() > 100_000
    assert long_df["FullPointCode"].nunique() > 1_000
