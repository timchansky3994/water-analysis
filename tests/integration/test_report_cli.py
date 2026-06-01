import os
import subprocess
import sys
from pathlib import Path

from tests.helpers import build_linear_raw_dataframe, write_raw_csv


def test_report_cli_generates_bundle(tmp_path: Path) -> None:
    raw_df = build_linear_raw_dataframe(
        dates=[f"{day:02d}.01.2018" for day in range(1, 11)],
        target_values=[float(day) for day in range(1, 11)],
        feature_map={
            "Цветность": [float(day * 2) for day in range(1, 11)],
            "Мутность (по формазину)": [float(day) / 2 for day in range(1, 11)],
        },
    )
    input_path = write_raw_csv(raw_df, tmp_path / "demo.csv")
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
    ]
    result = subprocess.run(command, cwd=Path.cwd(), env=env, capture_output=True, text=True, encoding="utf-8")

    assert result.returncode == 0, result.stderr
    assert (output_dir / "summary" / "specialist_summary.md").exists()
    assert (output_dir / "tables" / "comparison_summary.csv").exists()
    assert (output_dir / "tables" / "comparison_summary.xlsx").exists()
    assert (output_dir / "plots" / "correlation_heatmap.png").exists()
    assert (output_dir / "models" / "best_model_package").exists()
    assert not (output_dir / "models" / "model_package").exists()

    summary_text = (output_dir / "summary" / "specialist_summary.md").read_text(encoding="utf-8")
    assert "Параметры анализа" in summary_text
    assert "CSV" in summary_text
    assert "XLSX" in summary_text
