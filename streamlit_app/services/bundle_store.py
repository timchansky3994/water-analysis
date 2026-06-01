"""Discover and manage report bundles and model packages on disk."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPORTS_ROOT = _PROJECT_ROOT / "reports"


@dataclass
class BundleEntry:
    """Metadata about one report bundle directory."""

    path: Path
    created_at: datetime | None
    scope: str
    target: str
    status: str
    scope_id: str

    def display_label(self) -> str:
        ts = self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "дата неизвестна"
        scope_part = self.scope_id.replace(":", " / ") if self.scope_id else self.scope
        return f"{ts} · {scope_part} · {self.target}"


def _scope_detail(scope_name: str, scope_selectors: dict[str, Any]) -> str:
    """Format scope name + selectors into a compact human-readable label."""
    parts = []
    for key in ("OKTMO", "PointType_Code", "FullPointCode"):
        value = scope_selectors.get(key, "")
        if value and str(value) not in ("", "10110+10150"):
            parts.append(str(value))
    if parts:
        return f"{scope_name} ({' / '.join(parts)})"
    return scope_name


@dataclass
class ModelPackageEntry:
    """Metadata about one deployable model package."""

    path: Path
    target: str
    scope_name: str
    model_name: str
    created_at: str
    ml_beats_baseline: bool
    payload: dict[str, Any] = field(default_factory=dict)

    def display_label(self) -> str:
        try:
            dt = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
            ts = dt.strftime("%Y-%m-%d")
        except Exception:
            ts = self.created_at[:10] if self.created_at else "?"
        scope_part = _scope_detail(self.scope_name, self.payload.get("scope_selectors", {}))
        return f"{self.target} · {scope_part} · {self.model_name} · {ts}"


def _load_json_safe(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_created_at(params: dict[str, Any], bundle_path: Path) -> datetime | None:
    for key in ("created_at", "timestamp"):
        raw = params.get(key)
        if raw:
            try:
                return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except Exception:
                pass
    try:
        return datetime.fromtimestamp(bundle_path.stat().st_mtime)
    except Exception:
        return None


def list_report_bundles(reports_root: Path | None = None) -> list[BundleEntry]:
    """Recursively scan reports_root for valid report bundles.

    A bundle is a directory containing summary/specialist_summary.md.
    """
    root = reports_root or _REPORTS_ROOT
    if not root.exists():
        return []

    entries: list[BundleEntry] = []
    for summary_path in sorted(root.rglob("summary/specialist_summary.md")):
        bundle_dir = summary_path.parent.parent
        params = _load_json_safe(bundle_dir / "metadata" / "run_parameters.json")
        readiness = _load_json_safe(bundle_dir / "metadata" / "readiness.json")

        entry = BundleEntry(
            path=bundle_dir,
            created_at=_parse_created_at(params, bundle_dir),
            scope=params.get("scope", ""),
            target=params.get("target", ""),
            status=readiness.get("status", ""),
            scope_id=params.get("scope_id", ""),
        )
        entries.append(entry)

    entries.sort(key=lambda e: e.created_at or datetime.min, reverse=True)
    return entries


def list_model_packages(reports_root: Path | None = None) -> list[ModelPackageEntry]:
    """Scan reports_root for deployable model packages (best_model_package/)."""
    root = reports_root or _REPORTS_ROOT
    if not root.exists():
        return []

    entries: list[ModelPackageEntry] = []
    for card_path in sorted(root.rglob("best_model_package/model_card.json")):
        package_dir = card_path.parent
        card = _load_json_safe(card_path)
        if not card:
            continue
        entry = ModelPackageEntry(
            path=package_dir,
            target=card.get("target", ""),
            scope_name=card.get("scope_name", ""),
            model_name=card.get("model_name", ""),
            created_at=card.get("created_at", ""),
            ml_beats_baseline=bool(card.get("ml_beats_baseline", False)),
            payload=card,
        )
        entries.append(entry)

    entries.sort(key=lambda e: e.created_at, reverse=True)
    return entries


def delete_bundle(bundle_dir: Path) -> None:
    """Permanently delete a report bundle directory."""
    if bundle_dir.exists() and bundle_dir.is_dir():
        shutil.rmtree(bundle_dir)
