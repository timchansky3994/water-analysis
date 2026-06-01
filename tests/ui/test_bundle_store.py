"""Tests: bundle_store discovers bundles and model packages correctly."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from streamlit_app.services.bundle_store import (
    BundleEntry,
    ModelPackageEntry,
    delete_bundle,
    list_model_packages,
    list_report_bundles,
)


def _make_bundle(root: Path, name: str, scope: str = "global", target: str = "Жесткость общая") -> Path:
    bundle_dir = root / name
    (bundle_dir / "summary").mkdir(parents=True)
    (bundle_dir / "metadata").mkdir(parents=True)
    (bundle_dir / "summary" / "specialist_summary.md").write_text("# Test", encoding="utf-8")
    params = {"scope": scope, "scope_id": f"{scope}", "target": target}
    (bundle_dir / "metadata" / "run_parameters.json").write_text(
        json.dumps(params, ensure_ascii=False), encoding="utf-8"
    )
    readiness = {"status": "suitable"}
    (bundle_dir / "metadata" / "readiness.json").write_text(
        json.dumps(readiness, ensure_ascii=False), encoding="utf-8"
    )
    return bundle_dir


def _make_model_package(root: Path, bundle_name: str, target: str = "Жесткость общая") -> Path:
    pkg_dir = root / bundle_name / "models" / "best_model_package"
    pkg_dir.mkdir(parents=True)
    card = {
        "target": target,
        "scope_name": "global",
        "model_name": "bayesian_ridge",
        "created_at": "2026-01-01T12:00:00+00:00",
        "ml_beats_baseline": True,
    }
    (pkg_dir / "model_card.json").write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
    return pkg_dir


def test_list_report_bundles_empty(tmp_path: Path) -> None:
    bundles = list_report_bundles(tmp_path)
    assert bundles == []


def test_list_report_bundles_finds_bundle(tmp_path: Path) -> None:
    _make_bundle(tmp_path, "bundle_1")
    bundles = list_report_bundles(tmp_path)
    assert len(bundles) == 1
    assert isinstance(bundles[0], BundleEntry)
    assert bundles[0].target == "Жесткость общая"
    assert bundles[0].scope == "global"


def test_list_report_bundles_finds_multiple(tmp_path: Path) -> None:
    _make_bundle(tmp_path, "bundle_a", target="Жесткость общая")
    _make_bundle(tmp_path, "bundle_b", target="Цветность")
    bundles = list_report_bundles(tmp_path)
    assert len(bundles) == 2
    targets = {b.target for b in bundles}
    assert "Жесткость общая" in targets
    assert "Цветность" in targets


def test_list_report_bundles_no_summary_not_included(tmp_path: Path) -> None:
    bogus = tmp_path / "not_a_bundle"
    bogus.mkdir()
    (bogus / "metadata").mkdir()
    bundles = list_report_bundles(tmp_path)
    assert bundles == []


def test_list_model_packages_empty(tmp_path: Path) -> None:
    packages = list_model_packages(tmp_path)
    assert packages == []


def test_list_model_packages_finds_package(tmp_path: Path) -> None:
    _make_model_package(tmp_path, "bundle_1")
    packages = list_model_packages(tmp_path)
    assert len(packages) == 1
    pkg = packages[0]
    assert isinstance(pkg, ModelPackageEntry)
    assert pkg.target == "Жесткость общая"
    assert pkg.model_name == "bayesian_ridge"
    assert pkg.ml_beats_baseline is True


def test_delete_bundle_removes_directory(tmp_path: Path) -> None:
    bundle_dir = _make_bundle(tmp_path, "to_delete")
    assert bundle_dir.exists()
    delete_bundle(bundle_dir)
    assert not bundle_dir.exists()


def test_delete_bundle_nonexistent_does_not_raise(tmp_path: Path) -> None:
    nonexistent = tmp_path / "ghost"
    delete_bundle(nonexistent)


def test_bundle_display_label(tmp_path: Path) -> None:
    _make_bundle(tmp_path, "bundle_label")
    bundles = list_report_bundles(tmp_path)
    label = bundles[0].display_label()
    assert "Жесткость общая" in label
    assert "global" in label
