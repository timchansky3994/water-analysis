"""Page 5: Quick correlation search without model training."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import streamlit as st

from streamlit_app.components.data_loader import render_file_uploader
from streamlit_app.components.scope_picker import available_indicators, render_scope_picker
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.table_view import download_buttons, render_table

st.title("Анализ корреляций")
st.caption("Найдите статистические зависимости между показателями без обучения модели.")
render_sidebar()

# --- Step 1: file upload ---
st.header("1. Загрузка файла")
temp_path, raw_df, long_df, profile = render_file_uploader(key="corr_file")

if long_df is None:
    st.info("Загрузите файл, чтобы продолжить.")
    st.stop()

# --- Step 2: parameters ---
st.header("2. Параметры анализа")

from water_analysis.io.schemas import REQUIRED_TARGETS
from water_analysis.analysis.scopes import build_scope_slices
from water_analysis.analysis.correlations import run_correlation_analysis
from water_analysis.reporting.plots import plot_correlation_heatmap

scope_kwargs_corr = render_scope_picker(long_df, key_prefix="corr", show_target=False)

st.markdown("### Дополнительные параметры")

_corr_indicators = available_indicators(long_df)
_corr_default = [t for t in REQUIRED_TARGETS if t in _corr_indicators][:1]
selected_targets = st.multiselect(
    "Целевые показатели",
    options=_corr_indicators if _corr_indicators else list(REQUIRED_TARGETS),
    default=_corr_default if _corr_indicators else list(REQUIRED_TARGETS)[:1],
    key="corr_targets",
    help="Список сформирован из показателей загруженного файла. Начните вводить название для поиска.",
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    methods = st.multiselect(
        "Методы корреляции",
        options=["spearman", "pearson"],
        default=["spearman"],
        key="corr_methods",
    )
with col_b:
    min_shared = st.slider(
        "Минимум совместных измерений",
        min_value=5,
        max_value=100,
        value=20,
        key="corr_min_shared",
    )
with col_c:
    top_n = st.slider(
        "Топ N корреляций",
        min_value=5,
        max_value=50,
        value=15,
        key="corr_top_n",
    )

# --- Step 3: run ---
if st.button("Найти зависимости", type="primary", key="corr_run"):
    if not selected_targets:
        st.error("Выберите хотя бы один целевой показатель.")
        st.stop()
    if not methods:
        st.error("Выберите хотя бы один метод корреляции.")
        st.stop()

    try:
        with st.spinner("Рассчитываем корреляции..."):
            scope_slices = build_scope_slices(
                long_df,
                scope_name=scope_kwargs_corr["scope_name"],
                oktmo=scope_kwargs_corr["oktmo"],
                point_type=scope_kwargs_corr["point_type"],
                point_code=scope_kwargs_corr["point_code"],
            )
            if not scope_slices:
                st.warning("Выбранные фильтры не соответствуют ни одному срезу данных.")
                st.stop()

            analysis = run_correlation_analysis(
                scope_slices,
                targets=selected_targets,
                methods=tuple(methods),
                min_shared_samples=min_shared,
            )
            st.session_state["corr_analysis"] = analysis

    except Exception as exc:
        st.error(f"Ошибка: {exc}")
        with st.expander("Детали"):
            import traceback
            st.code(traceback.format_exc())
        st.stop()

if "corr_analysis" in st.session_state:
    analysis = st.session_state["corr_analysis"]
    top_n_disp = st.session_state.get("corr_top_n", 15)

    st.header("3. Результаты")

    if analysis.results.empty:
        st.warning("Подходящих пар показателей для расчёта корреляции не найдено.")
    else:
        import pandas as pd

        result_df = analysis.results.copy()
        result_df["corr_abs"] = result_df["corr"].abs()
        top_df = result_df.sort_values(["corr_abs", "n_shared"], ascending=[False, False]).head(top_n_disp)

        st.subheader(f"Топ {top_n_disp} корреляций")
        render_table(top_df.drop(columns=["corr_abs"], errors="ignore"))

        download_buttons(
            analysis.results,
            file_stem="correlation_results",
            table_name="correlation_results",
            key="dl_corr",
        )

        # Heatmap for each selected target
        import tempfile

        targets_with_results = [
            target for target in selected_targets
            if not analysis.results[analysis.results["target"] == target].empty
        ]
        for target in targets_with_results:
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    fig_path = Path(f.name)
                plot_correlation_heatmap(
                    analysis.results,
                    fig_path,
                    target=target,
                    max_features=25,
                    dpi=100,
                )
                if fig_path.exists() and fig_path.stat().st_size > 0:
                    st.subheader(f"Тепловая карта корреляций ({target})")
                    st.image(str(fig_path))
                fig_path.unlink(missing_ok=True)
            except Exception:
                pass

    if not analysis.diagnostics.empty:
        with st.expander("Диагностика"):
            render_table(analysis.diagnostics)
