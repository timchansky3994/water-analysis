"""Compatibility checks for applying model packages to new data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from water_analysis.analysis.scopes import ScopeSlice, build_scope_slices
from water_analysis.inference.model_card import ModelCard
from water_analysis.inference.package import LoadedModelPackage, load_model_package


@dataclass(frozen=True)
class CompatibilityResult:
    """Outcome of model-package compatibility checks."""

    compatible: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    available_features: tuple[str, ...]
    missing_features: tuple[str, ...]


def check_feature_compatibility(
    card: ModelCard,
    pivot_df: pd.DataFrame,
    *,
    skip_features: frozenset[str] | None = None,
) -> CompatibilityResult:
    """Check whether a pivot has the model features needed for inference.

    Features listed in `skip_features` are excluded from the check — they are
    expected to be built from the date column (e.g. seasonal features) rather
    than read from the input file.
    """
    if card.target in card.feature_names:
        return CompatibilityResult(
            compatible=False,
            reasons=("target_listed_as_feature",),
            warnings=tuple(),
            available_features=tuple(),
            missing_features=card.feature_names,
        )

    skip = skip_features or frozenset()
    indicator_features = tuple(f for f in card.feature_names if f not in skip)

    if not indicator_features:
        # Only seasonal (date-derived) features — always compatible
        return CompatibilityResult(
            compatible=True,
            reasons=tuple(),
            warnings=tuple(),
            available_features=tuple(),
            missing_features=tuple(),
        )

    available = tuple(f for f in indicator_features if f in pivot_df.columns)
    missing = tuple(f for f in indicator_features if f not in pivot_df.columns)
    if not available:
        return CompatibilityResult(
            compatible=False,
            reasons=("no_model_features_available",),
            warnings=tuple(),
            available_features=available,
            missing_features=missing,
        )
    warnings = tuple(f"missing_feature_column:{feature}" for feature in missing)
    return CompatibilityResult(
        compatible=True,
        reasons=tuple(),
        warnings=warnings,
        available_features=available,
        missing_features=missing,
    )


def scope_from_card_or_cli(
    long_df: pd.DataFrame,
    card: ModelCard,
    *,
    scope_name: str | None = None,
    oktmo: str | None = None,
    point_type: str | None = None,
    point_code: str | None = None,
    allow_scope_fallback: bool = False,
) -> tuple[ScopeSlice | None, tuple[str, ...]]:
    """Resolve and validate the inference scope against the model card."""
    requested_scope = scope_name or card.scope_name
    selectors = card.scope_selectors
    requested_oktmo = oktmo if oktmo is not None else selectors.get("OKTMO")
    requested_point_type = point_type if point_type is not None else selectors.get("PointType_Code")
    requested_point_code = point_code if point_code is not None else selectors.get("FullPointCode")
    cli_selector_requested = any(value is not None for value in (oktmo, point_type, point_code))

    if requested_point_type == "10110+10150":
        requested_point_type = None

    reasons: list[str] = []
    if requested_scope == "global" and cli_selector_requested and not allow_scope_fallback:
        reasons.append("local_selector_with_global_scope_requires_allow_scope_fallback")
    if requested_scope != card.scope_name:
        if not allow_scope_fallback:
            reasons.append(f"incompatible_scope:{requested_scope}:model_scope:{card.scope_name}")
        elif card.scope_name != "global":
            reasons.append(f"unsupported_scope_fallback_from:{card.scope_name}:to:{requested_scope}")

    if not allow_scope_fallback:
        if "OKTMO" in selectors and requested_oktmo is not None and str(requested_oktmo) != selectors["OKTMO"]:
            reasons.append("incompatible_oktmo")
        if "FullPointCode" in selectors and requested_point_code is not None and str(requested_point_code) != selectors["FullPointCode"]:
            reasons.append("incompatible_point_code")
        card_point_type = selectors.get("PointType_Code")
        if card_point_type and card_point_type != "10110+10150" and requested_point_type is not None and str(requested_point_type) != card_point_type:
            reasons.append("incompatible_point_type")

    if reasons:
        return None, tuple(reasons)

    slices = build_scope_slices(
        long_df,
        scope_name=requested_scope,
        oktmo=requested_oktmo,
        point_type=requested_point_type,
        point_code=requested_point_code,
    )
    if len(slices) != 1:
        return None, (f"scope_resolved_to_{len(slices)}_slices",)
    return slices[0], tuple()


def select_model_from_catalog(
    catalog_dir: str | Path,
    *,
    target: str,
    scope_name: str,
) -> LoadedModelPackage | None:
    """Select the first compatible package in a simple directory catalog."""
    root = Path(catalog_dir)
    for card_path in sorted(root.glob("**/model_card.json")):
        package = load_model_package(card_path.parent)
        card = package.model_card
        if card.target == target and card.scope_name == scope_name:
            return package
    return None
