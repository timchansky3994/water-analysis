"""Tests: scope_picker indicator helpers used to populate target/predictor pickers."""

from __future__ import annotations

from pathlib import Path

import pytest

from water_analysis.io.schemas import REQUIRED_TARGETS
from water_analysis.preprocessing.long_format import (
    build_canonical_long_format,
    read_source_table,
)
from streamlit_app.components.scope_picker import (
    _default_target_index,
    available_indicators,
)

from tests.helpers import build_linear_raw_dataframe, write_raw_csv


@pytest.fixture()
def long_df(tmp_path: Path):
    raw_df = build_linear_raw_dataframe(
        dates=[f"{day:02d}.01.2018" for day in range(1, 6)],
        target_values=[float(day) for day in range(1, 6)],
        feature_map={
            REQUIRED_TARGETS[1]: [float(day * 2) for day in range(1, 6)],
            "Железо общее": [float(day) / 2 for day in range(1, 6)],
        },
    )
    csv_path = write_raw_csv(raw_df, tmp_path / "source.csv")
    return build_canonical_long_format(read_source_table(csv_path))


def test_available_indicators_returns_sorted_indicators_from_data(long_df) -> None:
    indicators = available_indicators(long_df)
    expected = {"Жесткость общая", REQUIRED_TARGETS[1], "Железо общее"}
    assert set(indicators) == expected
    assert indicators == sorted(indicators)


def test_available_indicators_excludes_named_indicator(long_df) -> None:
    indicators = available_indicators(long_df, exclude="Жесткость общая")
    assert "Жесткость общая" not in indicators
    assert REQUIRED_TARGETS[1] in indicators


def test_default_target_index_prefers_first_required_target_present(long_df) -> None:
    indicators = available_indicators(long_df)
    # "Жесткость общая" is REQUIRED_TARGETS[0] and present, so it must be default.
    assert indicators[_default_target_index(indicators)] == "Жесткость общая"


def test_default_target_index_falls_back_to_zero_without_required_targets() -> None:
    indicators = ["Железо общее", "Алюминий"]
    assert _default_target_index(indicators) == 0
