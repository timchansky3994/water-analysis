"""Page 6: User guide rendered from docs/user_guide.md."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from streamlit_app.components.sidebar import render_sidebar

_GUIDE_PATH = _PROJECT_ROOT / "docs" / "user_guide.md"

st.title("Руководство пользователя")
render_sidebar()

if _GUIDE_PATH.exists():
    st.markdown(_GUIDE_PATH.read_text(encoding="utf-8"))
else:
    st.error(f"Файл руководства не найден: `{_GUIDE_PATH}`")
    st.info("Создайте файл `docs/user_guide.md` в корне проекта.")
