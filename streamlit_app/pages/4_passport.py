"""Page 4: Quick data profiling without model training."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import io

import streamlit as st

from streamlit_app.components.data_loader import render_file_uploader
from streamlit_app.components.scope_picker import available_indicators, render_scope_picker
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.table_view import download_buttons, render_table

st.title("Паспорт данных")
st.caption("Быстрая диагностика качества и структуры данных без обучения модели.")
render_sidebar()

# --- Step 1: file upload ---
st.header("1. Загрузка файла")
temp_path, raw_df, long_df, profile = render_file_uploader(key="passport_file")

if long_df is None:
    st.info("Загрузите файл, чтобы продолжить.")
    st.stop()

# --- Step 2: scope selection ---
st.header("2. Выбор среза")
scope_kwargs = render_scope_picker(long_df, key_prefix="passport", show_target=False)

# --- Step 2b: seasonality settings ---
with st.expander("Настройки сезонного анализа (необязательно)"):
    _pp_gran_label = st.radio(
        "Гранулярность",
        options=["По сезонам", "По месяцам"],
        index=0,
        key="passport_gran",
        horizontal=True,
    )
    _passport_granularity = "season" if _pp_gran_label == "По сезонам" else "month"

    _SKIP_SEASON = "— не выполнять —"
    _season_indicators = available_indicators(long_df)
    _pp_target_choice = st.selectbox(
        "Целевой показатель для сезонного анализа",
        options=[_SKIP_SEASON] + _season_indicators,
        index=0,
        key="passport_season_target",
        help="Выберите показатель из загруженного файла или оставьте «— не выполнять —», чтобы пропустить сезонный анализ.",
    )
    _pp_target_label = "" if _pp_target_choice == _SKIP_SEASON else _pp_target_choice

# --- Step 3: build passport ---
if st.button("Построить паспорт данных", type="primary", key="passport_run"):
    from water_analysis.analysis.scopes import build_scope_slices
    from water_analysis.profiling.passport import build_profile_reports

    try:
        with st.spinner("Профилируем данные..."):
            scope_slices = build_scope_slices(
                long_df,
                scope_name=scope_kwargs["scope_name"],
                oktmo=scope_kwargs["oktmo"],
                point_type=scope_kwargs["point_type"],
                point_code=scope_kwargs["point_code"],
            )
            if not scope_slices:
                st.warning("Выбранные фильтры не соответствуют ни одному срезу данных.")
                st.stop()

            reports = build_profile_reports(scope_slices)
            st.session_state["passport_reports"] = reports
            st.session_state["passport_scope_slices"] = scope_slices

    except Exception as exc:
        st.error(f"Ошибка: {exc}")
        with st.expander("Детали"):
            import traceback
            st.code(traceback.format_exc())
        st.stop()

if "passport_reports" in st.session_state:
    reports = st.session_state["passport_reports"]
    scope_slices_state = st.session_state.get("passport_scope_slices", [])
    import matplotlib.pyplot as plt
    from water_analysis.reporting.plots import plot_cooccurrence_heatmap

    for report in reports:
        st.subheader(f"Срез: {report.scope_label}")

        summary = report.summary
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Записей", summary.get("record_count", "—"))
        with col2:
            st.metric("Фактов взятия проб", summary.get("sample_event_count", "—"))
        with col3:
            st.metric("Уникальных ОКТМО", summary.get("unique_oktmo_count", "—"))
        with col4:
            st.metric("Уникальных точек", summary.get("unique_point_count", "—"))
        with col5:
            st.metric("Показателей", summary.get("indicator_count", "—"))

        tabs = st.tabs(["Наблюдения по показателям", "Пропуски", "Типы точек", "Постоянные ряды", "Совместная встречаемость"])

        with tabs[0]:
            if not report.indicator_observations.empty:
                render_table(report.indicator_observations)
            else:
                st.info("Нет данных.")

        with tabs[1]:
            if not report.missingness.empty:
                render_table(report.missingness)
            else:
                st.info("Нет данных о пропусках.")

        with tabs[2]:
            if not report.point_type_coverage.empty:
                render_table(report.point_type_coverage)
            else:
                st.info("Нет данных о типах точек.")

        with tabs[3]:
            if not report.constant_series.empty:
                render_table(report.constant_series)
            else:
                st.info("Постоянных рядов не обнаружено.")

        with tabs[4]:
            if not report.cooccurrence_matrix.empty:
                buf = io.BytesIO()
                fig_path = None
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        fig_path = Path(f.name)
                    plot_cooccurrence_heatmap(report.cooccurrence_matrix, fig_path, max_features=25, dpi=100)
                    if fig_path.exists() and fig_path.stat().st_size > 0:
                        st.image(str(fig_path))
                    else:
                        st.dataframe(report.cooccurrence_matrix, width="stretch")
                except Exception:
                    st.dataframe(report.cooccurrence_matrix, width="stretch")
                finally:
                    if fig_path and fig_path.exists():
                        fig_path.unlink(missing_ok=True)
            else:
                st.info("Матрица совместной встречаемости пуста.")

        # download buttons (CSV + XLSX per table)
        st.divider()
        for table_name, df in [
            ("indicator_observations", report.indicator_observations),
            ("missingness", report.missingness),
            ("point_type_coverage", report.point_type_coverage),
            ("constant_series", report.constant_series),
        ]:
            if not df.empty:
                st.caption(table_name)
                download_buttons(
                    df,
                    file_stem=table_name,
                    table_name=table_name,
                    key=f"dl_passport_{report.scope_id}_{table_name}",
                )

    # --- Seasonal analysis block ---
    if _pp_target_label.strip() and scope_slices_state:
        st.header("Сезонный анализ")
        from water_analysis.analysis.seasonality import analyze_seasonality
        from water_analysis.reporting.plots import plot_seasonal_profile

        for scope_slice in scope_slices_state:
            target_name = _pp_target_label.strip()
            st.subheader(f"Сезонный анализ — срез: {scope_slice.scope_label}")
            try:
                with st.spinner("Выполняем сезонный анализ..."):
                    season_result = analyze_seasonality(
                        scope_slice,
                        target=target_name,
                        granularity=_passport_granularity,
                        min_group_size=5,
                    )

                if not season_result.group_stats.empty:
                    render_table(season_result.group_stats)

                if season_result.pattern_test.get("test") == "kruskal_wallis":
                    p = season_result.pattern_test.get("p_value", float("nan"))
                    if season_result.seasonal_pattern_detected:
                        st.success(f"Сезонный паттерн обнаружен (критерий Краскела–Уоллиса, p={p:.3g})")
                    else:
                        st.info(f"Статистически значимый паттерн не обнаружен (критерий Краскела–Уоллиса, p={p:.3g})")
                else:
                    reason = season_result.pattern_test.get("reason", "")
                    st.warning(f"Тест пропущен: {reason}")

                # Plot
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    fig_path_s = Path(f.name)
                plot_seasonal_profile(season_result, fig_path_s, dpi=100)
                if fig_path_s.exists() and fig_path_s.stat().st_size > 0:
                    st.image(str(fig_path_s))
                fig_path_s.unlink(missing_ok=True)

                if not season_result.per_season_correlations.empty:
                    with st.expander("Корреляции внутри групп"):
                        render_table(season_result.per_season_correlations)

                if season_result.diagnostics:
                    with st.expander("Диагностика"):
                        for msg in season_result.diagnostics:
                            st.caption(msg)

            except Exception as exc:
                st.error(f"Ошибка сезонного анализа: {exc}")
