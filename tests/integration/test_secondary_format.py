"""Integration test: secondary regional export format with optional columns absent."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.helpers import build_secondary_raw_dataframe, write_raw_csv


def test_report_cli_secondary_format_explicit_profile(tmp_path: Path) -> None:
    """report command with --source-profile secondary succeeds on secondary-format CSV."""
    raw_df = build_secondary_raw_dataframe(
        dates=[f"{day:02d}.01.2018" for day in range(1, 11)],
        target_values=[float(day) for day in range(1, 11)],
        feature_map={
            "Цветность": [float(day * 2) for day in range(1, 11)],
            "Мутность (по формазину)": [float(day) / 2 for day in range(1, 11)],
        },
    )
    input_path = write_raw_csv(raw_df, tmp_path / "secondary.csv")
    output_dir = tmp_path / "bundle"

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["PYTHONIOENCODING"] = "utf-8"
    command = [
        sys.executable,
        "-m",
        "water_analysis.cli",
        "report",
        "--input",
        str(input_path),
        "--scope",
        "global",
        "--target",
        "Жесткость общая",
        "--output-dir",
        str(output_dir),
        "--model",
        "bayesian_ridge",
        "--min-train-size",
        "4",
        "--min-target-observations",
        "4",
        "--min-shared-samples",
        "4",
        "--min-eligible-predictors",
        "1",
        "--source-profile",
        "secondary",
    ]
    result = subprocess.run(command, cwd=Path.cwd(), env=env, capture_output=True, text=True, encoding="utf-8")

    assert result.returncode == 0, result.stderr
    assert (output_dir / "summary" / "specialist_summary.md").exists()
    summary_text = (output_dir / "summary" / "specialist_summary.md").read_text(encoding="utf-8")
    assert "Параметры анализа" in summary_text

    # Log file must exist and contain a warning about missing optional columns
    log_path = output_dir / "metadata" / "run.log"
    assert log_path.exists(), "run.log should be written automatically in report output dir"
    log_text = log_path.read_text(encoding="utf-8")
    assert "Optional source fields absent" in log_text


def test_report_cli_secondary_format_autodetect(tmp_path: Path) -> None:
    """report command with --source-profile auto detects the secondary profile."""
    raw_df = build_secondary_raw_dataframe(
        dates=[f"{day:02d}.01.2018" for day in range(1, 11)],
        target_values=[float(day) for day in range(1, 11)],
        feature_map={
            "Цветность": [float(day * 2) for day in range(1, 11)],
            "Мутность (по формазину)": [float(day) / 2 for day in range(1, 11)],
        },
    )
    input_path = write_raw_csv(raw_df, tmp_path / "secondary_auto.csv")
    output_dir = tmp_path / "bundle_auto"

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["PYTHONIOENCODING"] = "utf-8"
    command = [
        sys.executable,
        "-m",
        "water_analysis.cli",
        "report",
        "--input",
        str(input_path),
        "--scope",
        "global",
        "--target",
        "Жесткость общая",
        "--output-dir",
        str(output_dir),
        "--model",
        "bayesian_ridge",
        "--min-train-size",
        "4",
        "--min-target-observations",
        "4",
        "--min-shared-samples",
        "4",
        "--min-eligible-predictors",
        "1",
        # --source-profile defaults to "auto"
    ]
    result = subprocess.run(command, cwd=Path.cwd(), env=env, capture_output=True, text=True, encoding="utf-8")

    assert result.returncode == 0, result.stderr
    assert (output_dir / "summary" / "specialist_summary.md").exists()
