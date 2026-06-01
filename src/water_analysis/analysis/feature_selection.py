"""Leakage-safe feature selection for modeling."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal, Sequence

import pandas as pd
from scipy.stats import spearmanr

from water_analysis.preprocessing.pivot_builder import SAMPLE_POINT_INDEX

_LOGGER = logging.getLogger(__name__)

_CANDIDATE_TABLE_COLS = [
    "feature",
    "target_corr",
    "target_corr_abs",
    "target_corr_p_value",
    "n_shared",
    "selection_score",
    "included",
    "exclusion_reason",
    "is_forced",
]


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Selected and rejected predictors with selection diagnostics."""

    target: str
    selected_features: tuple[str, ...]
    candidate_table: pd.DataFrame
    dropped_multicollinear: tuple[str, ...]
    selection_mode: Literal["auto", "manual", "semi_auto"] = "auto"
    forced_features: tuple[str, ...] = ()


def indicator_columns(pivot_df: pd.DataFrame) -> list[str]:
    """Return columns corresponding to indicator values."""
    return [column for column in pivot_df.columns if column not in SAMPLE_POINT_INDEX]


def prepare_modeling_frame(long_df: pd.DataFrame) -> pd.DataFrame:
    """Build a sample-point pivot sorted chronologically for modeling."""
    from water_analysis.preprocessing.pivot_builder import build_indicator_pivot

    pivot_df = build_indicator_pivot(long_df, aggregation_level="sample_point_level")
    return pivot_df.sort_values(["SampleDate", "FullPointCode"]).reset_index(drop=True)


def _pairwise_spearman(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None]:
    """Calculate pairwise Spearman (corr, p_value) on shared non-null observations."""
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2:
        return None, None
    if valid.iloc[:, 0].nunique(dropna=True) <= 1 or valid.iloc[:, 1].nunique(dropna=True) <= 1:
        return None, None
    corr, p_value = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    if pd.isna(corr):
        return None, None
    return float(corr), float(p_value)


def _build_stats_table(train_frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Compute raw correlation stats for every indicator column except the target.

    Hard exclusions (constant_feature) are marked immediately.
    min_shared_samples, correlation, and significance filters are applied by the caller.
    """
    rows: list[dict] = []

    for feature in indicator_columns(train_frame):
        if feature == target:
            continue  # target itself is never a candidate predictor

        row: dict = {
            "feature": feature,
            "target_corr": float("nan"),
            "target_corr_abs": float("nan"),
            "target_corr_p_value": float("nan"),
            "n_shared": 0,
            "selection_score": float("nan"),
            "included": False,
            "exclusion_reason": "",
            "is_forced": False,
        }

        valid = train_frame[[target, feature]].dropna()
        row["n_shared"] = int(len(valid))

        if len(valid) < 2:
            # Too few for Spearman; will be caught by min_shared_samples filter downstream
            rows.append(row)
            continue

        if valid[feature].nunique(dropna=True) <= 1:
            row["exclusion_reason"] = "constant_feature"
            rows.append(row)
            continue

        if valid[target].nunique(dropna=True) <= 1:
            # Target is locally constant in this shared subset — cannot compute useful corr
            row["exclusion_reason"] = "constant_feature"
            rows.append(row)
            continue

        corr_val, p_val = spearmanr(valid[target], valid[feature])
        if pd.isna(corr_val):
            row["exclusion_reason"] = "constant_feature"
            rows.append(row)
            continue

        row["target_corr"] = float(corr_val)
        row["target_corr_abs"] = abs(float(corr_val))
        row["target_corr_p_value"] = float(p_val)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_CANDIDATE_TABLE_COLS)
    return pd.DataFrame(rows)


def _compute_selection_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Compute coverage-weighted selection scores for passable features in-place."""
    passable = table["exclusion_reason"] == ""
    if not passable.any():
        return table
    max_n_shared = float(table.loc[passable, "n_shared"].max())
    if max_n_shared <= 0:
        return table
    sqrt_max = math.sqrt(max_n_shared)
    table.loc[passable, "selection_score"] = (
        table.loc[passable, "target_corr_abs"]
        * table.loc[passable, "n_shared"].apply(lambda n: math.sqrt(float(n)))
        / sqrt_max
    )
    return table


def _greedy_select(
    candidates: list[str],
    already_selected: list[str],
    train_frame: pd.DataFrame,
    *,
    max_features: int,
    multicollinearity_threshold: float,
) -> tuple[list[str], dict[str, str]]:
    """Greedy selection with multicollinearity guard and budget cap.

    Returns (all_selected, exclusion_reasons_for_rejected).
    already_selected are treated as anchors and are never dropped.
    """
    selected = list(already_selected)
    exclusion_reasons: dict[str, str] = {}

    for feature in candidates:
        if feature in selected:
            continue

        multicollinear_with: str | None = None
        for approved in selected:
            corr, _ = _pairwise_spearman(train_frame[feature], train_frame[approved])
            if corr is None:
                continue
            if abs(corr) >= multicollinearity_threshold:
                multicollinear_with = approved
                break

        if multicollinear_with is not None:
            exclusion_reasons[feature] = f"multicollinear_with:{multicollinear_with}"
            continue

        if len(selected) >= max_features:
            exclusion_reasons[feature] = "budget_exhausted"
            continue

        selected.append(feature)

    return selected, exclusion_reasons


def select_predictors(
    train_frame: pd.DataFrame,
    *,
    target: str,
    mode: Literal["auto", "manual", "semi_auto"] = "auto",
    forced_features: Sequence[str] = (),
    min_shared_samples: int = 20,
    min_target_correlation: float = 0.3,
    significance_alpha: float = 0.05,
    max_features: int = 5,
    multicollinearity_threshold: float = 0.85,
) -> FeatureSelectionResult:
    """Select predictors on the training fold only (no holdout leakage).

    Three modes:
    - 'auto': fully automatic with coverage-weighted ranking
    - 'manual': use exactly the forced_features list (no correlation filtering)
    - 'semi_auto': start with forced_features, fill remaining budget with auto algorithm
    """
    if target not in train_frame.columns:
        raise ValueError(f"Target '{target}' is absent from the training frame.")

    forced_list = list(forced_features)
    all_indicators = indicator_columns(train_frame)

    if mode == "auto" and forced_list:
        raise ValueError(
            f"mode='auto' does not accept forced_features={forced_list!r}. "
            "Use mode='manual' or mode='semi_auto' to specify explicit features."
        )
    if mode in ("manual", "semi_auto") and not forced_list:
        raise ValueError(
            f"mode='{mode}' requires at least one forced_feature, but none were provided."
        )

    if forced_list:
        if target in forced_list:
            raise ValueError(
                f"Target '{target}' must not appear in forced_features — this would cause target leakage."
            )
        unknown = [f for f in forced_list if f not in all_indicators]
        if unknown:
            raise ValueError(
                f"forced_features not found in training frame's indicator columns: {unknown!r}. "
                f"Available indicators (first 20): {sorted(all_indicators)[:20]!r}"
            )

    table = _build_stats_table(train_frame, target)

    if table.empty:
        return FeatureSelectionResult(
            target=target,
            selected_features=(),
            candidate_table=table,
            dropped_multicollinear=(),
            selection_mode=mode,
            forced_features=tuple(forced_list),
        )

    dropped_multicollinear: tuple[str, ...] = ()
    selected: list[str] = []

    # ── MANUAL ──────────────────────────────────────────────────────────────
    if mode == "manual":
        tech_bad = (table["exclusion_reason"] == "") & (table["n_shared"] < min_shared_samples)
        table.loc[tech_bad, "exclusion_reason"] = "too_few_shared_samples"

        forced_set = set(forced_list)
        for feature in forced_list:
            idx = table["feature"] == feature
            if not idx.any():
                continue
            current_reason = str(table.loc[idx, "exclusion_reason"].iloc[0])
            if current_reason in ("constant_feature", "too_few_shared_samples"):
                _LOGGER.warning(
                    "Forced feature '%s' cannot be used in the model: %s.", feature, current_reason
                )
            else:
                table.loc[idx, "included"] = True
                table.loc[idx, "exclusion_reason"] = ""
                table.loc[idx, "is_forced"] = True
                selected.append(feature)

        not_forced_passable = (table["exclusion_reason"] == "") & ~table["is_forced"]
        table.loc[not_forced_passable, "exclusion_reason"] = "not_in_manual_list"

    # ── AUTO ─────────────────────────────────────────────────────────────────
    elif mode == "auto":
        tech_bad = (table["exclusion_reason"] == "") & (table["n_shared"] < min_shared_samples)
        table.loc[tech_bad, "exclusion_reason"] = "too_few_shared_samples"

        below_corr = (table["exclusion_reason"] == "") & (table["target_corr_abs"] < min_target_correlation)
        table.loc[below_corr, "exclusion_reason"] = "below_min_correlation"

        not_sig = (table["exclusion_reason"] == "") & (table["target_corr_p_value"] > significance_alpha)
        table.loc[not_sig, "exclusion_reason"] = "not_significant"

        table = _compute_selection_scores(table)

        passable_sorted = (
            table[table["exclusion_reason"] == ""]
            .sort_values("selection_score", ascending=False)["feature"]
            .tolist()
        )

        selected, excl_reasons = _greedy_select(
            passable_sorted,
            already_selected=[],
            train_frame=train_frame,
            max_features=max_features,
            multicollinearity_threshold=multicollinearity_threshold,
        )
        for feature, reason in excl_reasons.items():
            table.loc[table["feature"] == feature, "exclusion_reason"] = reason

        if selected:
            table.loc[table["feature"].isin(selected), "included"] = True

        dropped_multicollinear = tuple(
            f for f, r in excl_reasons.items() if r.startswith("multicollinear_with:")
        )

    # ── SEMI_AUTO ────────────────────────────────────────────────────────────
    else:
        # Phase 1: resolve forced features (apply only technical exclusions)
        tech_bad = (table["exclusion_reason"] == "") & (table["n_shared"] < min_shared_samples)
        table.loc[tech_bad, "exclusion_reason"] = "too_few_shared_samples"

        forced_selected: list[str] = []
        for feature in forced_list:
            idx = table["feature"] == feature
            if not idx.any():
                continue
            current_reason = str(table.loc[idx, "exclusion_reason"].iloc[0])
            if current_reason in ("constant_feature", "too_few_shared_samples"):
                _LOGGER.warning(
                    "Forced feature '%s' cannot be used in the model: %s.", feature, current_reason
                )
            else:
                forced_selected.append(feature)
                table.loc[idx, "included"] = True
                table.loc[idx, "exclusion_reason"] = ""
                table.loc[idx, "is_forced"] = True

        # Phase 2: auto-fill remaining budget
        remaining_budget = max_features - len(forced_selected)
        if remaining_budget <= 0:
            if len(forced_selected) > max_features:
                _LOGGER.warning(
                    "semi_auto: forced features count (%d) exceeds max_features=%d; "
                    "skipping automatic fill.",
                    len(forced_selected), max_features,
                )
            passable_non_forced = (table["exclusion_reason"] == "") & ~table["is_forced"]
            table.loc[passable_non_forced, "exclusion_reason"] = "budget_exhausted"
            selected = forced_selected
        else:
            passable_nf = (table["exclusion_reason"] == "") & ~table["is_forced"]

            below_corr = passable_nf & (table["target_corr_abs"] < min_target_correlation)
            table.loc[below_corr, "exclusion_reason"] = "below_min_correlation"

            passable_nf = (table["exclusion_reason"] == "") & ~table["is_forced"]
            not_sig = passable_nf & (table["target_corr_p_value"] > significance_alpha)
            table.loc[not_sig, "exclusion_reason"] = "not_significant"

            passable_nf_mask = (table["exclusion_reason"] == "") & ~table["is_forced"]
            if passable_nf_mask.any():
                max_n = float(table.loc[passable_nf_mask, "n_shared"].max())
                if max_n > 0:
                    sqrt_max = math.sqrt(max_n)
                    table.loc[passable_nf_mask, "selection_score"] = (
                        table.loc[passable_nf_mask, "target_corr_abs"]
                        * table.loc[passable_nf_mask, "n_shared"].apply(lambda n: math.sqrt(float(n)))
                        / sqrt_max
                    )

            auto_candidates = (
                table[passable_nf_mask]
                .sort_values("selection_score", ascending=False)["feature"]
                .tolist()
            )

            all_selected, excl_reasons = _greedy_select(
                auto_candidates,
                already_selected=forced_selected,
                train_frame=train_frame,
                max_features=max_features,
                multicollinearity_threshold=multicollinearity_threshold,
            )
            for feature, reason in excl_reasons.items():
                table.loc[table["feature"] == feature, "exclusion_reason"] = reason

            auto_added = [f for f in all_selected if f not in forced_selected]
            if auto_added:
                table.loc[table["feature"].isin(auto_added), "included"] = True
                table.loc[table["feature"].isin(auto_added), "exclusion_reason"] = ""

            selected = all_selected
            dropped_multicollinear = tuple(
                f for f, r in excl_reasons.items() if r.startswith("multicollinear_with:")
            )

    return FeatureSelectionResult(
        target=target,
        selected_features=tuple(selected),
        candidate_table=table.reset_index(drop=True),
        dropped_multicollinear=dropped_multicollinear,
        selection_mode=mode,
        forced_features=tuple(forced_list),
    )


@dataclass(frozen=True)
class NoPredictorDiagnosis:
    """Structured explanation of why feature selection yielded no predictors.

    Reason codes mirror the ``exclusion_reason`` values produced by
    :func:`select_predictors`; human-readable rendering belongs to the
    reporting layer, not here.
    """

    selection_mode: Literal["auto", "manual", "semi_auto"]
    forced_features: tuple[str, ...]
    # (feature, base_reason_code, n_shared) for each forced feature that was rejected.
    forced_details: tuple[tuple[str, str, int], ...]
    # (base_reason_code, count) aggregated across all candidate indicators.
    reason_counts: tuple[tuple[str, int], ...]
    total_candidates: int
    best_abs_correlation: float | None


def _base_reason_code(code: object) -> str:
    """Return the base exclusion reason (strip the ``:detail`` suffix)."""
    text = str(code) if code is not None else ""
    if not text:
        return "insufficient_data"
    return text.split(":", 1)[0]


def diagnose_no_predictors(result: FeatureSelectionResult) -> NoPredictorDiagnosis:
    """Explain why a feature selection run produced no usable predictors.

    Intended for the case where ``result.selected_features`` is empty: it reads
    the candidate table's ``exclusion_reason`` column and summarizes why every
    indicator was rejected, so the reason can be reported as explicitly as the
    data-readiness reasons.
    """
    table = result.candidate_table
    if table is None or table.empty:
        return NoPredictorDiagnosis(
            selection_mode=result.selection_mode,
            forced_features=tuple(result.forced_features),
            forced_details=(),
            reason_counts=(),
            total_candidates=0,
            best_abs_correlation=None,
        )

    forced_details: list[tuple[str, str, int]] = []
    for feature in result.forced_features:
        row = table[table["feature"] == feature]
        if row.empty:
            forced_details.append((feature, "feature_absent", 0))
            continue
        forced_details.append(
            (
                feature,
                _base_reason_code(row["exclusion_reason"].iloc[0]),
                int(row["n_shared"].iloc[0]),
            )
        )

    counts: dict[str, int] = {}
    for code in table["exclusion_reason"].tolist():
        base = _base_reason_code(code)
        counts[base] = counts.get(base, 0) + 1
    reason_counts = tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    corr_series = pd.to_numeric(table["target_corr_abs"], errors="coerce").dropna()
    best_abs_correlation = float(corr_series.max()) if not corr_series.empty else None

    return NoPredictorDiagnosis(
        selection_mode=result.selection_mode,
        forced_features=tuple(result.forced_features),
        forced_details=tuple(forced_details),
        reason_counts=reason_counts,
        total_candidates=int(len(table)),
        best_abs_correlation=best_abs_correlation,
    )
