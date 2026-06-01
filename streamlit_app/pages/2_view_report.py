"""Page 2: Browse and view existing report bundles."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import streamlit as st

from streamlit_app.components.results_browser import render_report_bundle
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.services.bundle_store import delete_bundle, list_report_bundles

st.title("Просмотр существующего отчёта")
render_sidebar()

bundles = list_report_bundles()

if not bundles:
    st.info(
        "Готовые отчёты не найдены в папке `reports/`. "
        "Сначала запустите анализ на странице «Анализ и отчёт»."
    )
    st.stop()

labels = [b.display_label() for b in bundles]
selected_idx = st.selectbox(
    "Выберите отчёт",
    options=range(len(bundles)),
    format_func=lambda i: labels[i],
)
selected_bundle = bundles[selected_idx]

st.caption(f"Папка: `{selected_bundle.path}`")

render_report_bundle(selected_bundle.path)

st.divider()
st.subheader("Удаление отчёта")
confirm_delete = st.checkbox(
    "Я подтверждаю, что хочу безвозвратно удалить этот отчёт",
    key="confirm_delete",
)
if st.button("Удалить отчёт", disabled=not confirm_delete, key="delete_bundle"):
    try:
        delete_bundle(selected_bundle.path)
        st.success("Отчёт удалён. Обновите страницу.")
        st.rerun()
    except Exception as exc:
        st.error(f"Ошибка при удалении: {exc}")
