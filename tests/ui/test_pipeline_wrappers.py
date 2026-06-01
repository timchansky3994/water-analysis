"""Tests: pipeline.py wrappers call the Python API correctly."""

from __future__ import annotations

from pathlib import Path

import pytest

from water_analysis.io.schemas import REQUIRED_TARGETS
from streamlit_app.services.pipeline import run_full_report

from tests.helpers import build_linear_raw_dataframe, write_raw_csv


@pytest.fixture()
def csv_with_enough_data(tmp_path: Path) -> Path:
    raw_df = build_linear_raw_dataframe(
        dates=[f"{day:02d}.01.2018" for day in range(1, 16)],
        target_values=[float(day) for day in range(1, 16)],
        feature_map={
            REQUIRED_TARGETS[1]: [float(day * 2) for day in range(1, 16)],
            REQUIRED_TARGETS[2]: [float(day) / 2 for day in range(1, 16)],
        },
    )
    csv_path = tmp_path / "source.csv"
    write_raw_csv(raw_df, csv_path)
    return csv_path


def test_run_full_report_returns_bundle(csv_with_enough_data: Path, tmp_path: Path) -> None:
    bundle = run_full_report(
        csv_with_enough_data,
        scope_name="global",
        oktmo=None,
        point_type=None,
        point_code=None,
        target=REQUIRED_TARGETS[0],
        output_dir=tmp_path / "out",
    )
    assert bundle.output_dir.exists()
    assert bundle.summary_path.exists()
    assert bundle.readiness_assessment is not None


def test_run_full_report_creates_expected_files(csv_with_enough_data: Path, tmp_path: Path) -> None:
    bundle = run_full_report(
        csv_with_enough_data,
        scope_name="global",
        oktmo=None,
        point_type=None,
        point_code=None,
        target=REQUIRED_TARGETS[0],
        output_dir=tmp_path / "out2",
    )
    assert (bundle.output_dir / "tables").exists()
    assert (bundle.output_dir / "plots").exists()
    assert (bundle.output_dir / "metadata" / "run_parameters.json").exists()


def test_run_full_report_writes_run_log(csv_with_enough_data: Path, tmp_path: Path) -> None:
    bundle = run_full_report(
        csv_with_enough_data,
        scope_name="global",
        oktmo=None,
        point_type=None,
        point_code=None,
        target=REQUIRED_TARGETS[0],
        output_dir=tmp_path / "out_log",
    )
    log_path = bundle.output_dir / "metadata" / "run.log"
    assert log_path.exists(), "run.log should be written for UI-generated bundles"
    assert log_path.read_text(encoding="utf-8").strip(), "run.log should not be empty"


def test_run_full_report_writes_run_log_when_water_analysis_logger_isolated(
    csv_with_enough_data: Path, tmp_path: Path
) -> None:
    """Reproduce the running Streamlit app's logging setup.

    app.py routes the ``water_analysis`` logger to a NullHandler and sets
    ``propagate = False`` so library logs don't reach the terminal. A run.log
    handler attached to the root logger would then capture nothing. This test
    pins that the run log is still written (and non-empty) under that setup.
    """
    import logging

    wa_logger = logging.getLogger("water_analysis")
    saved_handlers = list(wa_logger.handlers)
    saved_propagate = wa_logger.propagate
    saved_level = wa_logger.level
    try:
        # Mirror streamlit_app/app.py exactly.
        wa_logger.handlers = [logging.NullHandler()]
        wa_logger.propagate = False
        wa_logger.setLevel(logging.NOTSET)

        bundle = run_full_report(
            csv_with_enough_data,
            scope_name="global",
            oktmo=None,
            point_type=None,
            point_code=None,
            target=REQUIRED_TARGETS[0],
            output_dir=tmp_path / "out_log_isolated",
        )
        log_path = bundle.output_dir / "metadata" / "run.log"
        assert log_path.exists(), "run.log should be written even with propagate=False"
        assert log_path.read_text(encoding="utf-8").strip(), "run.log should not be empty"
    finally:
        wa_logger.handlers = saved_handlers
        wa_logger.propagate = saved_propagate
        wa_logger.setLevel(saved_level)


def test_run_full_report_calls_progress_callback(csv_with_enough_data: Path, tmp_path: Path) -> None:
    calls: list[tuple[str, float]] = []

    def _progress(stage: str, fraction: float) -> None:
        calls.append((stage, fraction))

    run_full_report(
        csv_with_enough_data,
        scope_name="global",
        oktmo=None,
        point_type=None,
        point_code=None,
        target=REQUIRED_TARGETS[0],
        output_dir=tmp_path / "out3",
        on_progress=_progress,
    )

    assert len(calls) >= 3
    assert calls[-1][0] == "Готово"
    assert calls[-1][1] == 1.0


def test_run_full_report_raises_on_no_scope_match(csv_with_enough_data: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="не соответствуют"):
        run_full_report(
            csv_with_enough_data,
            scope_name="oktmo",
            oktmo="99999999999",  # non-existent OKTMO
            point_type=None,
            point_code=None,
            target=REQUIRED_TARGETS[0],
            output_dir=tmp_path / "out4",
        )
