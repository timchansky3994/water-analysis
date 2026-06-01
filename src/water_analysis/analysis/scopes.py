"""Scope builders for analytical slices."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from water_analysis.io.schemas import DRINKING_WATER_POINT_TYPES, SCOPE_NAMES


@dataclass(frozen=True)
class ScopeSlice:
    """A concrete analytical slice of the canonical long-format dataset."""

    scope_name: str
    scope_id: str
    scope_label: str
    selector: dict[str, str]
    dataframe: pd.DataFrame

    def metadata_record(self) -> dict[str, str]:
        """Return flat scope metadata without the dataframe payload."""
        return {
            "scope_name": self.scope_name,
            "scope_id": self.scope_id,
            "scope_label": self.scope_label,
            **self.selector,
        }


def _make_scope_slice(
    *,
    scope_name: str,
    scope_id: str,
    scope_label: str,
    selector: dict[str, str],
    dataframe: pd.DataFrame,
) -> ScopeSlice:
    """Build a scope slice while normalizing the frame copy."""
    return ScopeSlice(
        scope_name=scope_name,
        scope_id=scope_id,
        scope_label=scope_label,
        selector=selector,
        dataframe=dataframe.copy(),
    )


def _sorted_unique(series: pd.Series) -> list[str]:
    """Return sorted non-empty unique values as strings."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    return sorted(values.unique().tolist())


def build_scope_slices(
    long_df: pd.DataFrame,
    *,
    scope_name: str,
    oktmo: str | None = None,
    point_type: str | None = None,
    point_code: str | None = None,
) -> list[ScopeSlice]:
    """Build analytical slices for a requested scope."""
    if scope_name not in SCOPE_NAMES:
        raise ValueError(f"Unsupported scope: {scope_name}")

    if scope_name == "global":
        return [
            _make_scope_slice(
                scope_name="global",
                scope_id="global",
                scope_label="Global dataset",
                selector={},
                dataframe=long_df,
            )
        ]

    if scope_name == "oktmo":
        oktmo_values = [str(oktmo)] if oktmo else _sorted_unique(long_df["OKTMO"])
        slices: list[ScopeSlice] = []
        for current_oktmo in oktmo_values:
            scoped = long_df[long_df["OKTMO"].astype(str) == current_oktmo]
            if scoped.empty:
                continue
            slices.append(
                _make_scope_slice(
                    scope_name="oktmo",
                    scope_id=f"oktmo:{current_oktmo}",
                    scope_label=f"OKTMO {current_oktmo}",
                    selector={"OKTMO": current_oktmo},
                    dataframe=scoped,
                )
            )
        return slices

    if scope_name == "oktmo_point_type":
        scoped_df = long_df
        if oktmo:
            scoped_df = scoped_df[scoped_df["OKTMO"].astype(str) == str(oktmo)]
        if point_type:
            scoped_df = scoped_df[scoped_df["PointType_Code"].astype(str) == str(point_type)]

        pairs = (
            scoped_df[["OKTMO", "PointType_Code"]]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values(["OKTMO", "PointType_Code"])
        )

        slices = []
        for _, pair in pairs.iterrows():
            current_oktmo = pair["OKTMO"]
            current_type = pair["PointType_Code"]
            scoped = scoped_df[
                (scoped_df["OKTMO"].astype(str) == current_oktmo)
                & (scoped_df["PointType_Code"].astype(str) == current_type)
            ]
            if scoped.empty:
                continue
            slices.append(
                _make_scope_slice(
                    scope_name="oktmo_point_type",
                    scope_id=f"oktmo_point_type:{current_oktmo}:{current_type}",
                    scope_label=f"OKTMO {current_oktmo} / point type {current_type}",
                    selector={"OKTMO": current_oktmo, "PointType_Code": current_type},
                    dataframe=scoped,
                )
            )
        return slices

    if scope_name == "drinking_water_combined":
        scoped_df = long_df[long_df["PointType_Code"].isin(DRINKING_WATER_POINT_TYPES)]
        if oktmo:
            scoped_df = scoped_df[scoped_df["OKTMO"].astype(str) == str(oktmo)]
        if scoped_df.empty:
            return []

        selector = {"PointType_Code": "10110+10150"}
        scope_id = "drinking_water_combined"
        scope_label = "Combined drinking water (10110 + 10150)"
        if oktmo:
            selector["OKTMO"] = str(oktmo)
            scope_id = f"{scope_id}:{oktmo}"
            scope_label = f"{scope_label} / OKTMO {oktmo}"

        return [
            _make_scope_slice(
                scope_name="drinking_water_combined",
                scope_id=scope_id,
                scope_label=scope_label,
                selector=selector,
                dataframe=scoped_df,
            )
        ]

    point_values = [str(point_code)] if point_code else _sorted_unique(long_df["FullPointCode"])
    slices = []
    for current_point_code in point_values:
        scoped = long_df[long_df["FullPointCode"].astype(str) == current_point_code]
        if scoped.empty:
            continue
        oktmo_value = scoped["OKTMO"].dropna().astype(str).iloc[0] if scoped["OKTMO"].notna().any() else ""
        point_type_value = (
            scoped["PointType_Code"].dropna().astype(str).iloc[0] if scoped["PointType_Code"].notna().any() else ""
        )
        slices.append(
            _make_scope_slice(
                scope_name="point",
                scope_id=f"point:{current_point_code}",
                scope_label=f"Point {current_point_code}",
                selector={
                    "FullPointCode": current_point_code,
                    "OKTMO": oktmo_value,
                    "PointType_Code": point_type_value,
                },
                dataframe=scoped,
            )
        )
    return slices
