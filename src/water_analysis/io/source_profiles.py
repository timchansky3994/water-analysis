"""Source profile loading and auto-detection for regional export formats."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

LOGGER = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILES_DIR = _PROJECT_ROOT / "configs" / "source_profiles"

REQUIRED_RAW_FIELDS: frozenset[str] = frozenset({"SampleDate", "FullPointCode", "Indicator", "ResultValueText"})


@dataclass(frozen=True)
class SourceProfile:
    """Mapping from raw export column names to canonical field names."""

    name: str
    description: str
    column_aliases: dict[str, tuple[str, ...]]


def _load_yaml_profile(path: Path) -> SourceProfile:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    aliases: dict[str, tuple[str, ...]] = {
        field: tuple(names) for field, names in data.get("column_aliases", {}).items()
    }
    return SourceProfile(
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        column_aliases=aliases,
    )


def load_source_profile(name_or_path: str) -> SourceProfile:
    """Load a source profile by name (from configs/source_profiles/) or by full path."""
    candidate = Path(name_or_path)
    if candidate.exists():
        return _load_yaml_profile(candidate)

    named_path = DEFAULT_PROFILES_DIR / f"{name_or_path}.yaml"
    if named_path.exists():
        return _load_yaml_profile(named_path)

    # Fallback: profile files may be named with a leading underscore (e.g. "_default.yaml")
    # but expose a clean name ("default") in their YAML content.
    prefixed_path = DEFAULT_PROFILES_DIR / f"_{name_or_path}.yaml"
    if prefixed_path.exists():
        return _load_yaml_profile(prefixed_path)

    raise FileNotFoundError(
        f"Source profile '{name_or_path}' not found. "
        f"Looked at '{candidate}' and '{named_path}'. "
        f"Available built-in profiles: {[p.stem for p in DEFAULT_PROFILES_DIR.glob('*.yaml')]}"
        if DEFAULT_PROFILES_DIR.exists()
        else f"Source profile '{name_or_path}' not found and profile directory '{DEFAULT_PROFILES_DIR}' does not exist."
    )


def list_available_profiles() -> list[SourceProfile]:
    """Return all profiles found in the default profiles directory."""
    if not DEFAULT_PROFILES_DIR.exists():
        return []
    profiles: list[SourceProfile] = []
    for yaml_path in sorted(DEFAULT_PROFILES_DIR.glob("*.yaml")):
        try:
            profiles.append(_load_yaml_profile(yaml_path))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not load profile '%s': %s", yaml_path, exc)
    return profiles


def autodetect_source_profile(columns: Iterable[str]) -> SourceProfile | None:
    """Pick the best-matching profile for the given set of column names.

    Returns the profile that covers all required fields with the most optional
    field matches, or None if no profile covers all required fields.
    """
    column_set = {col.strip() for col in columns}
    profiles = list_available_profiles()

    best_profile: SourceProfile | None = None
    best_required = -1
    best_optional = -1

    for profile in profiles:
        required_hits = 0
        optional_hits = 0
        for field, aliases in profile.column_aliases.items():
            matched = any(alias in column_set for alias in aliases)
            if field in REQUIRED_RAW_FIELDS:
                if matched:
                    required_hits += 1
            elif matched:
                optional_hits += 1

        if required_hits < len(REQUIRED_RAW_FIELDS):
            continue

        if required_hits > best_required or (required_hits == best_required and optional_hits > best_optional):
            best_profile = profile
            best_required = required_hits
            best_optional = optional_hits

    return best_profile
