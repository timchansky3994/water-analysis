"""Shared sidebar rendered on every page of the application."""

from __future__ import annotations

import os

import streamlit as st


def render_sidebar() -> None:
    """Render the common sidebar block (guide link, about, exit button).

    Streamlit re-runs each page script independently, so the sidebar must be
    rendered explicitly from every page rather than only from the home page.
    """
    with st.sidebar:
        st.page_link("pages/6_guide.py", label="Руководство пользователя", icon="📖")
        st.divider()
        st.markdown("**О программе**")
        st.caption("Версия 1.0.0 · Для СЗНЦ Гигиены и общественного здоровья")
        st.divider()
        if st.button("Завершить приложение", key="sidebar_exit", use_container_width=True):
            os._exit(0)
