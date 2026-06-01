"""Results browser component shared between page 1 and page 2."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from water_analysis.profiling.readiness import ReadinessAssessment

from streamlit_app.components.readiness_badge import render_readiness_badge
from streamlit_app.components.table_view import download_buttons_from_files, render_table


def render_report_bundle(bundle_dir: Path, readiness_assessment: ReadinessAssessment | None = None) -> None:
    """Render the full report bundle as tabbed content.

    Works for both a freshly generated bundle (page 1) and an existing bundle (page 2).
    """
    summary_path = bundle_dir / "summary" / "specialist_summary.md"
    tables_dir = bundle_dir / "tables"
    plots_dir = bundle_dir / "plots"
    log_path = bundle_dir / "metadata" / "run.log"

    if readiness_assessment is not None:
        render_readiness_badge(readiness_assessment)

    tabs = st.tabs(["Сводка", "Таблицы", "Графики", "Лог запуска"])

    with tabs[0]:
        if summary_path.exists():
            st.markdown(summary_path.read_text(encoding="utf-8"))
        else:
            st.info("Краткий отчёт не найден.")

    with tabs[1]:
        if tables_dir.exists():
            csv_files = sorted(tables_dir.glob("*.csv"))
            if csv_files:
                for csv_path in csv_files:
                    xlsx_path = csv_path.with_suffix(".xlsx")
                    with st.expander(csv_path.stem):
                        try:
                            import pandas as pd
                            df = pd.read_csv(csv_path, encoding="utf-8-sig")
                            render_table(df, max_rows=200)
                        except Exception as exc:
                            st.warning(f"Не удалось прочитать таблицу: {exc}")
                        download_buttons_from_files(
                            csv_path=csv_path,
                            xlsx_path=xlsx_path,
                            key=f"dl_{csv_path.stem}_{bundle_dir.name}",
                        )
            else:
                st.info("Таблицы не найдены.")
        else:
            st.info("Папка таблиц не найдена.")

    with tabs[2]:
        if plots_dir.exists():
            png_files = sorted(plots_dir.glob("*.png"))
            if png_files:
                for png_path in png_files:
                    if png_path.stat().st_size > 0:
                        st.subheader(png_path.stem.replace("_", " ").title())
                        st.image(str(png_path))
                    else:
                        st.caption(f"{png_path.name} — пустой файл (график не был построен)")
            else:
                st.info("Графики не найдены.")
        else:
            st.info("Папка графиков не найдена.")

    with tabs[3]:
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            st.code(log_text, language=None)
        else:
            st.info("Лог запуска не найден.")

    st.divider()
    st.caption(f"Папка с результатами: `{bundle_dir}`")
