"""Page 3: Apply a saved model package to estimate missing values."""

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
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.table_view import download_buttons_from_files, render_table
from streamlit_app.services.bundle_store import list_model_packages
from streamlit_app.services.pipeline import run_estimate_manual, run_estimate_missing

# ── Human-readable translations for diagnostic reason codes ─────────────────

_RUN_REASON_RU: dict[str, str] = {
    "target_mismatch": (
        "Целевой показатель в запросе не совпадает с показателем, на котором обучалась модель. "
        "Проверьте, что выбран правильный пакет модели."
    ),
    "missing_feature_columns_not_allowed": (
        "Один или несколько признаков модели полностью отсутствуют в загруженном файле. "
        "Включите настройку «Разрешить отсутствующие столбцы признаков», чтобы продолжить "
        "(пропущенные признаки будут заменены пустыми значениями)."
    ),
    "no_model_features_available": (
        "Ни один из признаков модели не найден в загруженном файле. "
        "Убедитесь, что файл содержит те же показатели, что использовались при обучении модели."
    ),
    "target_listed_as_feature": (
        "Пакет модели повреждён: целевой показатель указан как входной признак. "
        "Пересоздайте пакет через «Анализ, моделирование и отчёт»."
    ),
    "empty_scope_after_pivot": (
        "После фильтрации по сценарию данные оказались пустыми. "
        "Возможно, файл не содержит строк для выбранного ОКТМО или типа точки."
    ),
}

_ROW_REASON_RU: dict[str, str] = {
    "skipped_insufficient_features": (
        "Строки пропущены: слишком мало заполненных признаков. "
        "Увеличьте «Минимум наблюдаемых признаков» или снизьте порог, "
        "чтобы разрешить оценку при меньшем числе данных."
    ),
    "skipped_low_feature_coverage": (
        "Строки пропущены: доля заполненных признаков ниже порога «Минимальное покрытие признаков». "
        "Снизьте этот порог, чтобы разрешить оценку при большей доле пропусков."
    ),
}


def _run_reason_message(reason: str) -> str:
    """Translate a run-level reason code to Russian."""
    if reason.startswith("incompatible_scope:"):
        parts = reason.split(":")
        req = parts[1] if len(parts) > 1 else "?"
        model = parts[3] if len(parts) > 3 else "?"
        return (
            f"Сценарий «{req}» в запросе не совпадает со сценарием «{model}», "
            "на котором обучалась модель. Данные применяются к другому срезу."
        )
    if reason.startswith("scope_resolved_to_"):
        n = reason.replace("scope_resolved_to_", "").replace("_slices", "")
        if n == "0":
            return (
                "Данные не содержат строк, подходящих под сценарий модели. "
                "Проверьте фильтры ОКТМО и тип точки."
            )
        return f"Фильтрация дала {n} среза(ов) вместо одного — уточните параметры запроса."
    if "incompatible_oktmo" in reason:
        return "ОКТМО в данных не совпадает с ОКТМО, на котором обучалась модель."
    if "incompatible_point_type" in reason:
        return "Тип точки отбора в данных не совпадает с типом точки модели."
    if "incompatible_point_code" in reason:
        return "Полный код точки отбора не совпадает с кодом, использованным при обучении."
    return _RUN_REASON_RU.get(reason, f"Ошибка запуска: `{reason}`.")


def _render_inference_diagnostics(result) -> None:  # type: ignore[no-untyped-def]
    """Render human-readable diagnostic banners for an inference result."""
    import pandas as pd

    diag = result.diagnostics

    # ── No diagnostics at all ──────────────────────────────────────────────
    if diag.empty and result.skipped_rows == 0:
        st.success("Все строки успешно обработаны — диагностических сообщений нет.")
        return

    any_message_shown = False

    # ── Run-level errors (fatal — the whole run was blocked) ───────────────
    if not diag.empty and "level" in diag.columns:
        run_errors = diag[diag["level"] == "run"]
        for _, row in run_errors.iterrows():
            st.error(f"**Ошибка запуска.** {_run_reason_message(str(row.get('reason', '')))}")
            any_message_shown = True

    # ── Feature-level issues (missing columns filled with NaN) ────────────
    if not diag.empty and "level" in diag.columns:
        feat_rows = diag[(diag["level"] == "feature") & (diag["reason"] == "missing_feature_column")] if "reason" in diag.columns else pd.DataFrame()
        missing_cols = feat_rows["detail"].dropna().tolist() if "detail" in feat_rows.columns else []
        if missing_cols:
            cols_list = ", ".join(f"**{c}**" for c in missing_cols)
            st.warning(
                f"**Отсутствующие признаки ({len(missing_cols)} шт.).** "
                f"Следующие признаки модели не найдены в файле и заменены пустыми значениями: {cols_list}. "
                "Это может снизить точность оценок — чем больше пропущено признаков, тем менее надёжен результат."
            )
            any_message_shown = True

    # ── Row-level skips ───────────────────────────────────────────────────
    if not diag.empty and "level" in diag.columns and "reason" in diag.columns:
        row_diag = diag[diag["level"] == "row"]
        for reason_code, group in row_diag.groupby("reason"):
            count = len(group)
            base_msg = _ROW_REASON_RU.get(
                str(reason_code),
                f"Строки пропущены по причине `{reason_code}`.",
            )
            st.warning(f"**Пропущено строк: {count}.** {base_msg}")
            any_message_shown = True

    # ── Partial success summary ───────────────────────────────────────────
    if result.predicted_rows > 0:
        pct = int(100 * result.predicted_rows / max(result.rows_for_estimation, 1))
        if result.skipped_rows == 0:
            st.success(
                f"Все {result.predicted_rows} строк успешно оценены."
            )
        else:
            st.info(
                f"Успешно оценено **{result.predicted_rows}** из {result.rows_for_estimation} строк ({pct}%). "
                f"Пропущено: {result.skipped_rows}."
            )
        any_message_shown = True
    elif result.rows_for_estimation > 0 and result.predicted_rows == 0:
        st.error(
            f"Ни одна из {result.rows_for_estimation} строк не была оценена. "
            "Проверьте диагностику выше."
        )
        any_message_shown = True

    # ── Raw table for technical users ─────────────────────────────────────
    if not diag.empty:
        with st.expander("Подробная диагностика (техническая таблица)"):
            render_table(diag)

st.title("Расчётная оценка прогнозируемых значений")
render_sidebar()

st.warning(
    "**Расчётные значения являются аналитическими оценками модели и не заменяют "
    "лабораторные измерения.** Они не должны перезаписываться поверх исходных результатов анализа."
)

# --- Step 1: select model package ---
st.header("1. Выбор пакета модели")

packages = list_model_packages()

pre_selected_path = st.session_state.get("estimate_model_package", None)

if packages:
    labels = [p.display_label() for p in packages]
    paths_str = [str(p.path) for p in packages]

    default_idx = 0
    if pre_selected_path and pre_selected_path in paths_str:
        default_idx = paths_str.index(pre_selected_path)

    selected_idx = st.selectbox(
        "Пакет модели",
        options=range(len(packages)),
        format_func=lambda i: labels[i],
        index=default_idx,
    )
    selected_pkg = packages[selected_idx]
    package_dir = selected_pkg.path

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Целевой показатель", selected_pkg.target)
        st.metric("Модель", selected_pkg.model_name)
    with col2:
        st.metric("Сценарий", selected_pkg.scope_name)
        beats = "Да" if selected_pkg.ml_beats_baseline else "Нет"
        st.metric("ML лучше baseline", beats)

    if not selected_pkg.ml_beats_baseline:
        st.warning("Модель не превышает baseline по качеству. Применяйте оценки с осторожностью.")
else:
    st.info(
        "Готовые пакеты моделей не найдены в папке `reports/`. "
        "Сначала запустите полный анализ на странице «Анализ, моделирование и отчёт»."
    )
    if pre_selected_path:
        manual_path = st.text_input("Или введите путь к папке best_model_package", value=pre_selected_path)
        package_dir = Path(manual_path) if manual_path else None
    else:
        st.stop()
    if not package_dir or not package_dir.exists():
        st.error("Путь к пакету модели не найден.")
        st.stop()

# --- Step 2: choose input mode ---
st.header("2. Способ ввода данных")
input_mode = st.radio(
    "Как предоставить данные для оценки?",
    options=("file", "manual"),
    format_func=lambda m: (
        "Загрузить файл с пропусками" if m == "file" else "Ввести значения вручную"
    ),
    horizontal=True,
    key="estimate_input_mode",
    help=(
        "«Загрузить файл» — заполнить пропуски целевого показателя в выгрузке. "
        "«Ввести значения вручную» — указать значения показателей прямо здесь и сразу "
        "получить оценку, с экспортом в CSV/XLSX как при загрузке файла."
    ),
)

if input_mode == "file":
    # --- File mode: fill gaps in an uploaded export ---
    st.subheader("Загрузка файла с пропусками")
    temp_path, raw_df, long_df, profile = render_file_uploader(key="estimate_file")

    if long_df is None:
        st.info("Загрузите файл с новыми данными, где отсутствует целевой показатель.")
        st.stop()

    st.subheader("Дополнительные параметры")
    with st.expander("Настройки (необязательно)"):
        from water_analysis.config import load_config
        config = load_config()
        inf_cfg = config.get("inference", {})

        min_obs_feat = st.slider(
            "Минимум наблюдаемых признаков",
            min_value=1,
            max_value=10,
            value=int(inf_cfg.get("min_observed_features", 2)),
            key="est_min_feat",
        )
        min_feat_cov = st.slider(
            "Минимальное покрытие признаков",
            min_value=0.0,
            max_value=1.0,
            value=float(inf_cfg.get("min_feature_coverage", 0.5)),
            step=0.05,
            key="est_feat_cov",
        )
        allow_missing_cols = st.checkbox(
            "Разрешить отсутствующие столбцы признаков (заполнить NaN)",
            value=False,
            key="est_allow_missing",
        )
        predict_all_rows = st.checkbox(
            "Оценивать все строки (не только с пропуском целевого)",
            value=False,
            key="est_predict_all",
        )

    st.subheader("Запуск оценки")
    if st.button("Запустить оценку", type="primary", key="estimate_run"):
        if temp_path is None:
            st.error("Сначала загрузите файл данных.")
            st.stop()

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _on_progress(stage: str, fraction: float) -> None:
            progress_bar.progress(min(fraction, 1.0))
            status_text.text(stage)

        _orig_name = st.session_state.get("_dl_estimate_file", {}).get("original_name")

        try:
            with st.spinner("Выполняется расчётная оценка..."):
                result, output_dir = run_estimate_missing(
                    temp_path,
                    package_dir,
                    source_profile=profile,
                    min_observed_features=min_obs_feat,
                    min_feature_coverage=min_feat_cov,
                    allow_missing_feature_columns=allow_missing_cols,
                    predict_all=predict_all_rows,
                    on_progress=_on_progress,
                    input_display_name=_orig_name,
                )
            progress_bar.progress(1.0)
            status_text.text("Готово")
            st.session_state["last_estimate_result"] = result
            st.session_state["last_estimate_dir"] = str(output_dir)
            st.success(f"Оценка завершена. Результаты сохранены в `{output_dir}`.")

        except Exception as exc:
            st.error(f"Ошибка при выполнении оценки: {type(exc).__name__}: {exc}")
            with st.expander("Детали ошибки"):
                import traceback
                st.code(traceback.format_exc())
            st.stop()

else:
    # --- Manual mode: type indicator values and get an estimate in place ---
    import datetime as _dt

    import pandas as pd

    from water_analysis.inference.engine import manual_input_feature_names
    from water_analysis.inference.package import load_model_package

    st.subheader("Ввод значений показателей")

    try:
        _manual_pkg = load_model_package(package_dir)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не удалось загрузить пакет модели: {type(exc).__name__}: {exc}")
        st.stop()

    _features = manual_input_feature_names(_manual_pkg.model_card)
    _uses_seasonal = _manual_pkg.model_card.seasonal_feature != "none"

    st.markdown(
        f"Введите значения показателей, по которым модель оценивает **«{_manual_pkg.model_card.target}»**. "
        "Каждая строка — отдельная проба. Можно добавить несколько строк."
    )
    if _features:
        st.caption("Показатели модели: " + ", ".join(f"«{f}»" for f in _features))
    else:
        st.caption("Модель использует только сезонные признаки — укажите лишь дату отбора.")
    if _uses_seasonal:
        st.info("Модель учитывает сезонность — дата отбора важна и обязательна для каждой строки.")

    _template = pd.DataFrame([{"SampleDate": _dt.date.today().strftime("%d.%m.%Y"), **{f: None for f in _features}}])
    _column_config = {
        "SampleDate": st.column_config.TextColumn(
            "Дата отбора (ДД.ММ.ГГГГ)",
            required=True,
            validate=r"^\d{1,2}\.\d{1,2}\.\d{4}$",
            help="Дата пробы в формате ДД.ММ.ГГГГ, например 15.03.2024",
        ),
    }
    for _f in _features:
        _column_config[_f] = st.column_config.NumberColumn(_f, step=0.01)

    _edited = st.data_editor(
        _template,
        num_rows="dynamic",
        column_config=_column_config,
        hide_index=True,
        width="stretch",
        key="manual_editor",
    )

    with st.expander("Настройки (необязательно)"):
        manual_min_obs = st.slider(
            "Минимум заполненных признаков",
            min_value=0,
            max_value=max(len(_features), 1),
            value=min(1, len(_features)),
            key="manual_min_feat",
            help="Сколько показателей минимально должно быть заполнено, чтобы выдать оценку.",
        )
        manual_min_cov = st.slider(
            "Минимальное покрытие признаков",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="manual_feat_cov",
        )

    st.subheader("Запуск оценки")
    if st.button("Рассчитать", type="primary", key="manual_run_btn"):
        _samples = _edited.copy()
        _samples = _samples[_samples["SampleDate"].notna() & (_samples["SampleDate"].astype(str).str.strip() != "")]
        if _samples.empty:
            st.error("Заполните хотя бы одну строку с датой отбора.")
            st.stop()

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _on_progress_manual(stage: str, fraction: float) -> None:
            progress_bar.progress(min(fraction, 1.0))
            status_text.text(stage)

        try:
            with st.spinner("Выполняется расчётная оценка..."):
                result, output_dir = run_estimate_manual(
                    package_dir,
                    _samples,
                    min_observed_features=manual_min_obs,
                    min_feature_coverage=manual_min_cov,
                    on_progress=_on_progress_manual,
                )
            progress_bar.progress(1.0)
            status_text.text("Готово")
            st.session_state["last_estimate_result"] = result
            st.session_state["last_estimate_dir"] = str(output_dir)
            st.success(f"Оценка завершена. Результаты сохранены в `{output_dir}`.")

        except Exception as exc:
            st.error(f"Ошибка при выполнении оценки: {type(exc).__name__}: {exc}")
            with st.expander("Детали ошибки"):
                import traceback
                st.code(traceback.format_exc())
            st.stop()

# --- Results ---
if "last_estimate_result" in st.session_state:
    result = st.session_state["last_estimate_result"]
    output_dir = Path(st.session_state["last_estimate_dir"])

    st.header("3. Результаты")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Строк для оценки", result.rows_for_estimation)
    with col2:
        st.metric("Успешно оценено", result.predicted_rows)
    with col3:
        st.metric("Пропущено", result.skipped_rows)

    tabs = st.tabs(["Предсказания", "Диагностика", "Лог"])

    with tabs[0]:
        if not result.predictions.empty:
            render_table(result.predictions)
            download_buttons_from_files(
                csv_path=output_dir / "predictions.csv",
                xlsx_path=output_dir / "predictions.xlsx",
                key="dl_pred",
            )
        else:
            st.info("Нет строк для отображения.")

    with tabs[1]:
        _render_inference_diagnostics(result)

    with tabs[2]:
        summary_path = output_dir / "inference_summary.md"
        if summary_path.exists():
            st.markdown(summary_path.read_text(encoding="utf-8"))

    st.divider()
    st.caption(f"Результаты: `{output_dir}`")
