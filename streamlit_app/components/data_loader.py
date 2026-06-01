"""File upload and preview component."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from water_analysis.io.source_profiles import (
    SourceProfile,
    autodetect_source_profile,
    list_available_profiles,
    load_source_profile,
)
from water_analysis.preprocessing.long_format import build_canonical_long_format, read_source_table


@st.cache_data(show_spinner=False)
def _cached_read(file_bytes: bytes, suffix: str) -> pd.DataFrame:
    """Read a raw source file from bytes (cache key = content hash via bytes)."""
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    tmp_path = Path(tmp_name)
    try:
        os.write(fd, file_bytes)
        os.close(fd)
        return read_source_table(tmp_path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def render_file_uploader(
    key: str = "file",
) -> tuple[Path | None, pd.DataFrame | None, pd.DataFrame | None, SourceProfile | None]:
    """Render file uploader + profile selector + data preview.

    Returns (temp_path, raw_df, long_df, profile) if a file is loaded, else all None.
    The spinner only appears when the file content or profile selection actually changes,
    not on every widget interaction.
    """
    uploaded = st.file_uploader(
        "Загрузите файл данных (CSV или XLSX)",
        type=["csv", "xlsx"],
        key=key,
    )
    if uploaded is None:
        st.session_state.pop(f"_dl_{key}", None)
        _cleanup_temp(key)
        return None, None, None, None

    available_profiles = list_available_profiles()
    profile_names = ["auto"] + [p.name for p in available_profiles]
    profile_labels = ["Автоопределение"] + [f"{p.name} — {p.description}" for p in available_profiles]

    selected_idx = st.selectbox(
        "Профиль формата данных",
        options=range(len(profile_names)),
        format_func=lambda i: profile_labels[i],
        index=0,
        key=f"{key}_profile",
    )
    profile_arg = profile_names[selected_idx]

    file_bytes = bytes(uploaded.getbuffer())
    file_hash = hashlib.md5(file_bytes).hexdigest()
    suffix = Path(uploaded.name).suffix

    sl_key = f"_dl_{key}"
    cached = st.session_state.get(sl_key, {})
    needs_reload = (
        cached.get("file_hash") != file_hash
        or cached.get("profile_arg") != profile_arg
    )

    if needs_reload:
        with st.spinner("Читаем файл..."):
            try:
                raw_df = _cached_read(file_bytes, suffix)
                if profile_arg == "auto":
                    detected = autodetect_source_profile(raw_df.columns)
                    profile = detected if detected is not None else load_source_profile("_default")
                else:
                    profile = load_source_profile(profile_arg)
                long_df = build_canonical_long_format(raw_df, source_profile=profile)
            except Exception as exc:
                st.error(f"Ошибка чтения файла: {exc}")
                return None, None, None, None

        if cached.get("file_hash") != file_hash:
            _cleanup_temp(key)

        st.session_state[sl_key] = {
            "file_hash": file_hash,
            "profile_arg": profile_arg,
            "raw_df": raw_df,
            "long_df": long_df,
            "profile": profile,
            "original_name": uploaded.name,
        }

    data = st.session_state[sl_key]
    raw_df = data["raw_df"]
    long_df = data["long_df"]
    profile = data["profile"]

    # Ensure temp file exists for pipeline services that need a real path.
    temp_path = _ensure_temp_file(file_bytes, suffix, key)

    st.success(f"Обнаружен профиль: **{profile.name}** — {profile.description}")

    with st.expander("Превью данных (первые 20 строк)"):
        st.dataframe(raw_df.head(20), width="stretch")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Строк в файле", len(raw_df))
    with col2:
        if "SampleDate" in long_df.columns:
            dates = pd.to_datetime(long_df["SampleDate"], errors="coerce").dropna()
            period = f"{dates.min().date()} — {dates.max().date()}" if not dates.empty else "—"
        else:
            period = "—"
        st.metric("Период наблюдений", period)
    with col3:
        n_oktmo = long_df["OKTMO"].nunique() if "OKTMO" in long_df.columns else 0
        st.metric("Уникальных ОКТМО", n_oktmo)

    return temp_path, raw_df, long_df, profile


def _ensure_temp_file(file_bytes: bytes, suffix: str, key: str) -> Path | None:
    """Return a persistent temp file path for this file, creating it if necessary."""
    tmp_key = f"_tmp_path_{key}"
    temp_path: Path | None = st.session_state.get(tmp_key)
    if temp_path is None or not temp_path.exists():
        try:
            fd, tmp_name = tempfile.mkstemp(suffix=suffix)
            os.write(fd, file_bytes)
            os.close(fd)
            temp_path = Path(tmp_name)
        except Exception:
            temp_path = None
        st.session_state[tmp_key] = temp_path
    return temp_path


def _cleanup_temp(key: str) -> None:
    """Delete the cached temp file for this key, if any."""
    tmp_key = f"_tmp_path_{key}"
    old_tmp: Path | None = st.session_state.pop(tmp_key, None)
    if old_tmp and old_tmp.exists():
        try:
            old_tmp.unlink()
        except OSError:
            pass
