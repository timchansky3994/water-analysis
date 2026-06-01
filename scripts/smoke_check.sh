#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="src"

python -m water_analysis.cli --help >/dev/null
python -m water_analysis.cli report --help >/dev/null
python -m water_analysis.cli estimate-missing --help >/dev/null
python -m pytest tests -q
python -m water_analysis.cli report \
  --input data/raw/main.csv \
  --scope drinking_water_combined \
  --oktmo 14640101001 \
  --target "Жесткость общая" \
  --output-dir reports/smoke_check

echo "Smoke-check completed. Report bundle: reports/smoke_check"
