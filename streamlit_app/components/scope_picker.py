"""Scope selector component for the Streamlit UI."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from water_analysis.io.schemas import REQUIRED_TARGETS

SCOPE_DISPLAY_NAMES: dict[str, str] = {
    "global": "Весь набор данных",
    "oktmo": "По ОКТМО",
    "oktmo_point_type": "По ОКТМО + типу точки",
    "drinking_water_combined": "По ОКТМО + Питьевая вода (10110 + 10150)",
    "point": "Отдельная точка отбора",
}

POINT_TYPE_NAMES: dict[str, str] = {
    "10110": "10110 — точка на распределительной сети",
    "10150": "10150 — точка водопровода",
    "10310": "10310 — точка водоисточника (поверхностный)",
    "10320": "10320 — точка водоисточника (подземный)",
}


def _sorted_unique(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).str.strip().unique().tolist())


def available_indicators(long_df: pd.DataFrame, *, exclude: str | None = None) -> list[str]:
    """Return the indicators actually present in the loaded data, sorted.

    The list is built from the sample-point pivot, i.e. the same source that
    populates the predictor picker, so target and predictor selectors stay
    consistent with what the modeling pipeline will actually see. Falls back to
    the unique ``Indicator`` values if the pivot cannot be built.
    """
    indicators: list[str] = []
    try:
        from water_analysis.analysis.feature_selection import indicator_columns
        from water_analysis.preprocessing.pivot_builder import build_indicator_pivot

        pivot = build_indicator_pivot(long_df, aggregation_level="sample_point_level")
        indicators = list(indicator_columns(pivot))
    except Exception:
        if "Indicator" in long_df.columns:
            indicators = long_df["Indicator"].dropna().astype(str).str.strip().unique().tolist()
    if exclude is not None:
        indicators = [name for name in indicators if name != exclude]
    return sorted(indicators)


def _default_target_index(indicators: list[str]) -> int:
    """Pick a sensible default target: the first standard target present."""
    for preferred in REQUIRED_TARGETS:
        if preferred in indicators:
            return indicators.index(preferred)
    return 0


def render_scope_picker(
    long_df: pd.DataFrame,
    *,
    key_prefix: str = "scope",
    show_target: bool = True,
) -> dict[str, Any]:
    """Render scope / OKTMO / point_type / point_code / target selectors.

    Returns a dict with keys: scope_name, oktmo, point_type, point_code, target.
    When show_target=False, 'target' is None in the returned dict.
    """
    scope_names = list(SCOPE_DISPLAY_NAMES.keys())
    scope_labels = [SCOPE_DISPLAY_NAMES[s] for s in scope_names]

    selected_scope_idx = st.selectbox(
        "Сценарий анализа",
        options=range(len(scope_names)),
        format_func=lambda i: scope_labels[i],
        key=f"{key_prefix}_scope",
    )
    scope_name = scope_names[selected_scope_idx]

    oktmo: str | None = None
    point_type: str | None = None
    point_code: str | None = None

    if scope_name in ("oktmo", "oktmo_point_type", "drinking_water_combined"):
        oktmo_values = _sorted_unique(long_df["OKTMO"]) if "OKTMO" in long_df.columns else []
        if oktmo_values:
            any_oktmo = "— любой —"
            opts = [any_oktmo] + oktmo_values
            selected = st.selectbox(
                "ОКТМО",
                options=opts,
                key=f"{key_prefix}_oktmo",
            )
            oktmo = None if selected == any_oktmo else selected

    if scope_name in ("oktmo_point_type",):
        pt_values = _sorted_unique(long_df["PointType_Code"]) if "PointType_Code" in long_df.columns else []
        if pt_values:
            any_pt = "— любой —"
            opts = [any_pt] + pt_values
            selected = st.selectbox(
                "Тип точки",
                options=opts,
                format_func=lambda v: POINT_TYPE_NAMES.get(v, v),
                key=f"{key_prefix}_point_type",
            )
            point_type = None if selected == any_pt else selected

    if scope_name == "point":
        # Cascading OKTMO → PointType_Code filters to narrow down FullPointCode list.
        any_oktmo = "— любой —"
        oktmo_opts = _sorted_unique(long_df["OKTMO"]) if "OKTMO" in long_df.columns else []
        _filter_oktmo: str | None = None
        if oktmo_opts:
            sel_o = st.selectbox(
                "ОКТМО (фильтр, необязательно)",
                options=[any_oktmo] + oktmo_opts,
                key=f"{key_prefix}_point_filter_oktmo",
            )
            _filter_oktmo = None if sel_o == any_oktmo else sel_o

        _pc_src = long_df
        if _filter_oktmo and "OKTMO" in long_df.columns:
            _pc_src = _pc_src[_pc_src["OKTMO"].astype(str) == _filter_oktmo]

        any_pt = "— любой —"
        pt_opts = _sorted_unique(_pc_src["PointType_Code"]) if "PointType_Code" in _pc_src.columns else []
        _filter_pt: str | None = None
        if pt_opts:
            sel_pt = st.selectbox(
                "Тип точки (фильтр, необязательно)",
                options=[any_pt] + pt_opts,
                format_func=lambda v: POINT_TYPE_NAMES.get(v, v),
                key=f"{key_prefix}_point_filter_pt",
            )
            _filter_pt = None if sel_pt == any_pt else sel_pt

        if _filter_pt and "PointType_Code" in _pc_src.columns:
            _pc_src = _pc_src[_pc_src["PointType_Code"].astype(str) == _filter_pt]

        pc_values = _sorted_unique(_pc_src["FullPointCode"]) if "FullPointCode" in _pc_src.columns else []
        if pc_values:
            any_pc = "— любой —"
            selected = st.selectbox(
                "Полный код точки (FullPointCode)",
                options=[any_pc] + pc_values,
                key=f"{key_prefix}_point_code",
            )
            point_code = None if selected == any_pc else selected

    target: str | None = None
    if show_target:
        indicators = available_indicators(long_df)
        if indicators:
            target = st.selectbox(
                "Целевой показатель",
                options=indicators,
                index=_default_target_index(indicators),
                key=f"{key_prefix}_target",
                help="Список сформирован из показателей загруженного файла. Начните вводить название для поиска.",
            )
        else:
            # No indicators detected (e.g. the pivot could not be built) — fall
            # back to the standard targets so the page still works.
            target = st.selectbox(
                "Целевой показатель",
                options=list(REQUIRED_TARGETS),
                key=f"{key_prefix}_target",
            )

    return {
        "scope_name": scope_name,
        "oktmo": oktmo,
        "point_type": point_type,
        "point_code": point_code,
        "target": target,
    }
