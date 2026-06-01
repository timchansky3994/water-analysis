"""Page 1: Full analysis and report generation."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly via streamlit
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import streamlit as st

from streamlit_app.components.data_loader import render_file_uploader
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.results_browser import render_report_bundle
from streamlit_app.components.scope_picker import available_indicators, render_scope_picker
from streamlit_app.services.pipeline import run_full_report

st.title("Анализ, моделирование и отчёт")
st.caption("Полный конвейер обработки данных и построения модели.")
render_sidebar()

# --- Step 1: file upload ---
st.header("1. Загрузка файла")
temp_path, raw_df, long_df, profile = render_file_uploader(key="report_file")

if long_df is None:
    st.info("Загрузите файл CSV или XLSX, чтобы продолжить.")
    st.stop()

# --- Step 2: scope and target ---
st.header("2. Параметры анализа")
scope_kwargs = render_scope_picker(long_df, key_prefix="report")

from water_analysis.modeling.registry import available_model_specs
from water_analysis.config import load_config

with st.expander("Дополнительные настройки (необязательно)"):
    config = load_config()
    model_specs = list(available_model_specs().keys())
    default_models = config.get("modeling", {}).get("default_models", model_specs)
    selected_models = st.multiselect(
        "ML-модели для сравнения",
        options=model_specs,
        default=[m for m in default_models if m in model_specs],
        key="report_models",
    )
    readiness_cfg = config.get("readiness", {})
    modeling_cfg = config.get("modeling", {})
    seasonality_cfg = config.get("seasonality", {})

    col1, col2 = st.columns(2)
    with col1:
        min_obs = st.slider(
            "Минимум наблюдений целевого показателя",
            min_value=5,
            max_value=100,
            value=int(readiness_cfg.get("min_target_observations", 30)),
            key="report_min_obs",
        )
        max_miss = st.slider(
            "Макс. доля пропусков",
            min_value=0.0,
            max_value=1.0,
            value=float(readiness_cfg.get("max_missing_ratio", 0.6)),
            step=0.05,
            key="report_max_miss",
        )
    with col2:
        min_shared = st.slider(
            "Минимум совместных измерений",
            min_value=5,
            max_value=100,
            value=int(readiness_cfg.get("min_shared_samples", 20)),
            key="report_min_shared",
        )
        heavy_cens = st.slider(
            "Порог сильного цензурирования",
            min_value=0.0,
            max_value=1.0,
            value=float(readiness_cfg.get("heavy_censoring_ratio", 0.5)),
            step=0.05,
            key="report_heavy_cens",
        )

    col3, col4 = st.columns(2)
    with col3:
        weight_holdout = st.slider(
            "Вес holdout в комбинированном скоре",
            min_value=0.0,
            max_value=1.0,
            value=float(modeling_cfg.get("combined_score_weight_holdout", 0.4)),
            step=0.05,
            key="report_weight_holdout",
            help="Остаток (1 − вес holdout) идёт на backtest.",
        )
    with col4:
        weight_backtest_display = round(1.0 - weight_holdout, 10)
        st.metric("Вес backtest (автоматически)", f"{weight_backtest_display:.2f}")

    st.divider()
    _SEASONAL_LABELS = {
        "none": "Нет (рекомендуется)",
        "season": "Сезон (зима/весна/лето/осень)",
        "month": "Месяц (циклическое кодирование sin/cos)",
    }
    seasonal_feature_label = st.radio(
        "Сезонный признак в модели",
        options=list(_SEASONAL_LABELS.values()),
        index=0,
        key="report_seasonal_feature",
        help="Добавляет сезон как признак модели. По умолчанию выключено: сезон — это не лабораторный показатель, а вычисляемая из даты величина.",
    )
    seasonal_feature = {v: k for k, v in _SEASONAL_LABELS.items()}[seasonal_feature_label]

    seasonality_granularity_label = st.radio(
        "Гранулярность сезонного анализа (только для раздела диагностики)",
        options=["По сезонам", "По месяцам"],
        index=0,
        key="report_seasonality_granularity",
    )
    seasonality_granularity = "season" if seasonality_granularity_label == "По сезонам" else "month"

    modeling_overrides = {
        "min_target_observations": min_obs,
        "max_missing_ratio": max_miss,
        "min_shared_samples": min_shared,
        "heavy_censoring_ratio": heavy_cens,
        "combined_score_weight_holdout": weight_holdout,
        "combined_score_weight_backtest": weight_backtest_display,
        "seasonal_feature": seasonal_feature,
    }

# --- Feature selection mode ---
st.header("3. Выбор предикторов")

_MODE_LABELS = {
    "auto": "Автоматический",
    "semi_auto": "Полу-автоматический",
    "manual": "Ручной",
}
_MODE_CAPTIONS = {
    "auto": "Система сама выбирает показатели по корреляции и покрытию. Настройте пороги ниже.",
    "semi_auto": "Укажите обязательные показатели. Система дополнит список автоматически до заданного числа.",
    "manual": "Только указанные вами показатели войдут в модель. Корреляция не проверяется.",
}

# Resolve preselected from correlations page
_preselected_forced = st.session_state.pop("preselected_forced_features", [])
_default_mode_idx = 1 if _preselected_forced else 0  # switch to semi_auto if coming from correlations

selection_mode_label = st.radio(
    "Режим выбора показателей для модели",
    options=list(_MODE_LABELS.values()),
    index=_default_mode_idx,
    key="report_selection_mode_label",
)
selection_mode = {v: k for k, v in _MODE_LABELS.items()}[selection_mode_label]
st.caption(_MODE_CAPTIONS[selection_mode])

forced_features: list[str] = []
feature_selection_overrides: dict = {}

if selection_mode == "auto":
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        auto_max_features = st.slider(
            "Макс. число предикторов",
            min_value=1, max_value=15,
            value=int(modeling_cfg.get("max_features", 5)),
            key="report_auto_max_features",
        )
    with col_b:
        auto_min_corr = st.slider(
            "Мин. корреляция с целевым показателем",
            min_value=0.05, max_value=0.95,
            value=float(modeling_cfg.get("min_target_correlation", 0.3)),
            step=0.05,
            key="report_auto_min_corr",
        )
    with col_c:
        auto_sig_alpha = st.slider(
            "Порог значимости (α)",
            min_value=0.01, max_value=0.20,
            value=float(modeling_cfg.get("significance_alpha", 0.05)),
            step=0.01,
            key="report_auto_sig_alpha",
        )
    feature_selection_overrides = {
        "max_features": auto_max_features,
        "min_target_correlation": auto_min_corr,
        "significance_alpha": auto_sig_alpha,
    }

elif selection_mode in ("manual", "semi_auto"):
    # Build the list of available indicators from the loaded data (same source
    # as the target picker), excluding the chosen target to avoid leakage.
    _available_indicators = available_indicators(long_df, exclude=scope_kwargs.get("target"))

    _default_forced = [f for f in _preselected_forced if f in _available_indicators]

    forced_features = st.multiselect(
        "Показатели, которые обязательно войдут в модель",
        options=_available_indicators,
        default=_default_forced,
        key="report_forced_features",
        help="Выберите показатели, которые должны быть включены в модель независимо от корреляции.",
    )

    if selection_mode == "semi_auto":
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            sa_max_features = st.slider(
                "Макс. число предикторов (включая обязательные)",
                min_value=1, max_value=15,
                value=int(modeling_cfg.get("max_features", 5)),
                key="report_sa_max_features",
            )
        with col_b:
            sa_min_corr = st.slider(
                "Мин. корреляция для авто-добавляемых",
                min_value=0.05, max_value=0.95,
                value=float(modeling_cfg.get("min_target_correlation", 0.3)),
                step=0.05,
                key="report_sa_min_corr",
            )
        with col_c:
            sa_sig_alpha = st.slider(
                "Порог значимости (α) для авто-добавляемых",
                min_value=0.01, max_value=0.20,
                value=float(modeling_cfg.get("significance_alpha", 0.05)),
                step=0.01,
                key="report_sa_sig_alpha",
            )
        feature_selection_overrides = {
            "max_features": sa_max_features,
            "min_target_correlation": sa_min_corr,
            "significance_alpha": sa_sig_alpha,
        }

# --- Step 4: run ---
st.header("4. Запуск анализа")

if st.button("Запустить анализ", type="primary", key="report_run"):
    if temp_path is None or long_df is None:
        st.error("Сначала загрузите файл данных.")
        st.stop()

    # Validate feature selection params before run
    if selection_mode in ("manual", "semi_auto") and not forced_features:
        st.error(f"Режим «{_MODE_LABELS[selection_mode]}» требует хотя бы одного обязательного показателя.")
        st.stop()

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def _on_progress(stage: str, fraction: float) -> None:
        progress_bar.progress(min(fraction, 1.0))
        status_text.text(stage)

    _selection_overrides = {
        **modeling_overrides,
        **feature_selection_overrides,
        "selection_mode": selection_mode,
        "forced_features": forced_features,
    }

    _orig_name = st.session_state.get("_dl_report_file", {}).get("original_name")

    try:
        with st.spinner("Выполняется анализ..."):
            bundle = run_full_report(
                temp_path,
                scope_name=scope_kwargs["scope_name"],
                oktmo=scope_kwargs["oktmo"],
                point_type=scope_kwargs["point_type"],
                point_code=scope_kwargs["point_code"],
                target=scope_kwargs["target"],
                source_profile=profile,
                model_names=selected_models or None,
                modeling_overrides=_selection_overrides,
                on_progress=_on_progress,
                input_display_name=_orig_name,
                seasonality_granularity=seasonality_granularity,
            )
        progress_bar.progress(1.0)
        status_text.text("Готово")
        st.session_state["last_bundle_dir"] = str(bundle.output_dir)
        st.session_state["last_readiness"] = bundle.readiness_assessment
        st.success(f"Анализ завершён. Результаты: `{bundle.output_dir}`")

    except Exception as exc:
        st.error(f"Ошибка при выполнении анализа: {type(exc).__name__}")
        with st.expander("Детали ошибки"):
            import traceback
            st.code(traceback.format_exc())
        st.stop()

# --- Results ---
if "last_bundle_dir" in st.session_state:
    st.header("5. Результаты")
    bundle_dir = Path(st.session_state["last_bundle_dir"])
    readiness = st.session_state.get("last_readiness")
    render_report_bundle(bundle_dir, readiness_assessment=readiness)

    model_package_path = bundle_dir / "models" / "best_model_package"
    if model_package_path.exists():
        st.divider()
        if st.button("Перейти к расчёту прогнозируемых значений", key="goto_estimate"):
            st.session_state["estimate_model_package"] = str(model_package_path)
            st.switch_page("pages/3_estimate.py")
