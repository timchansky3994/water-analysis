"""Entrypoint for the water_analysis Streamlit application."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Route water_analysis logs to a NullHandler so they don't appear in the
# terminal while Streamlit is running.  Errors and exceptions are already
# surfaced inside the UI via st.exception / st.error.
_wa_logger = logging.getLogger("water_analysis")
if not _wa_logger.handlers:
    _wa_logger.addHandler(logging.NullHandler())
_wa_logger.propagate = False

import streamlit as st

from streamlit_app.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Анализ качества воды",
    page_icon="💧",
    layout="wide",
)

# Define all pages explicitly so the sidebar shows proper Russian titles.
# Using st.navigation() disables Streamlit's automatic pages/ discovery.
_page_analyze = st.Page("pages/1_report.py", title="Анализ, моделирование и отчёт", icon="📊")
_page_view = st.Page("pages/2_view_report.py", title="Просмотр отчёта", icon="📂")
_page_estimate = st.Page("pages/3_estimate.py", title="Прогноз значений", icon="🔢")
_page_passport = st.Page("pages/4_passport.py", title="Паспорт данных", icon="📋")
_page_corr = st.Page("pages/5_correlations.py", title="Корреляции", icon="📈")
_page_guide = st.Page(
    "pages/6_guide.py", title="Руководство пользователя", icon="📖", visibility="hidden"
)


def _home() -> None:
    st.title("Система анализа качества питьевой воды")

    st.markdown(
        """
Инструмент поддержки решения специалиста для анализа лабораторных данных о качестве питьевой воды.
Программа проверяет полноту данных, ищет зависимости между показателями, сравнивает статистические
и ML-модели и формирует отчёт с интерпретацией ограничений.

**Расчётные оценки не являются лабораторными измерениями** и не заменяют официальные результаты анализа.
"""
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Анализ, моделирование и отчёт")
        st.write("Загрузите файл данных, выберите сценарий и целевой показатель, получите полный отчёт с моделированием.")
        st.page_link(_page_analyze, label="Открыть", icon="📊")

    with col2:
        st.subheader("Просмотр отчёта")
        st.write("Откройте ранее сформированный отчёт из папки reports/.")
        st.page_link(_page_view, label="Открыть", icon="📂")

    with col3:
        st.subheader("Расчёт прогнозируемых значений")
        st.write("Примените сохранённую модель к новой выгрузке или введите значения показателей вручную для расчёта прогнозируемого показателя.")
        st.page_link(_page_estimate, label="Открыть", icon="🔢")

    st.divider()

    col4, col5, col6 = st.columns(3)

    with col4:
        st.subheader("Паспорт данных")
        st.write("Быстрая диагностика: пропуски, совместные измерения, постоянные ряды.")
        st.page_link(_page_passport, label="Открыть", icon="📋")

    with col5:
        st.subheader("Корреляции")
        st.write("Найдите статистические зависимости между показателями для выбранного среза.")
        st.page_link(_page_corr, label="Открыть", icon="📈")

    with col6:
        st.subheader("Руководство")
        st.write("Инструкция по установке, форматам файлов и работе с программой.")
        st.page_link(_page_guide, label="Открыть", icon="📖")

    render_sidebar()


_page_home = st.Page(_home, title="Домашняя страница", icon="🏠", default=True)

pg = st.navigation(
    [_page_home, _page_analyze, _page_view, _page_estimate, _page_passport, _page_corr, _page_guide]
)
pg.run()
