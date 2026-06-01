from pathlib import Path

from water_analysis.io.schemas import REQUIRED_TARGETS, build_required_target_diagnostics
from water_analysis.preprocessing.long_format import build_canonical_long_format, read_source_table


DATASET_PATH = Path("data/raw/main.csv")


def test_required_targets_have_explicit_diagnostics() -> None:
    long_df = build_canonical_long_format(read_source_table(DATASET_PATH))
    diagnostics = build_required_target_diagnostics(long_df["Indicator"])

    assert [item.indicator for item in diagnostics] == list(REQUIRED_TARGETS)
    assert all(item.status in {"present", "missing"} for item in diagnostics)

    counts_by_indicator = long_df["Indicator"].value_counts()
    for item in diagnostics:
        expected_count = int(counts_by_indicator.get(item.indicator, 0))
        assert item.observed_rows == expected_count
        assert item.status == ("present" if expected_count > 0 else "missing")
