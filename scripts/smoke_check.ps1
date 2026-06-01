$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:PYTHONPATH = "src"
$python = ".\.venv\Scripts\python.exe"

& $python -m water_analysis.cli --help | Out-Null
& $python -m water_analysis.cli report --help | Out-Null
& $python -m water_analysis.cli estimate-missing --help | Out-Null
& $python -m pytest tests -q
& $python -m water_analysis.cli report `
  --input data/raw/main.csv `
  --scope drinking_water_combined `
  --oktmo 14640101001 `
  --target "Жесткость общая" `
  --output-dir reports\smoke_check

Write-Host "Smoke-check completed. Report bundle: reports\smoke_check"
