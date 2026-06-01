"""Tests for the three feature selection modes."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from water_analysis.analysis.feature_selection import (
    FeatureSelectionResult,
    indicator_columns,
    select_predictors,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INDEX_COLS = ["SampleDate", "FullPointCode"]


def _make_frame(**kwargs: list) -> pd.DataFrame:
    """Build a pivot-like DataFrame with index columns and indicator columns."""
    n = max(len(v) for v in kwargs.values())
    data: dict = {
        "SampleDate": pd.date_range("2020-01-01", periods=n, freq="ME"),
        "FullPointCode": [f"00000000001.10110.{i % 3 + 1}" for i in range(n)],
    }
    data.update(kwargs)
    return pd.DataFrame(data)


def _corr_abs(a: list, b: list) -> float:
    return abs(float(spearmanr(a, b).statistic))


# ---------------------------------------------------------------------------
# 1. auto mode: coverage-weighted ranking
# ---------------------------------------------------------------------------

class TestAutoMode:
    def test_uses_coverage_weighted_ranking(self):
        """Feature with lower |corr| but many shared samples beats high-corr sparse feature."""
        rng = np.random.default_rng(42)
        n = 80
        target = rng.standard_normal(n)

        # high corr (0.8ish), few shared: put NaN in most rows
        high_corr_values = list(target[:20] * 0.8 + rng.standard_normal(20) * 0.1) + [float("nan")] * 60
        # moderate corr (~0.6), many shared
        moderate_corr_values = list(target * 0.6 + rng.standard_normal(n) * 0.3)

        frame = _make_frame(
            target=list(target),
            high_corr_sparse=high_corr_values,
            moderate_corr_dense=moderate_corr_values,
        )

        result = select_predictors(
            frame,
            target="target",
            mode="auto",
            min_shared_samples=15,
            min_target_correlation=0.3,
            significance_alpha=0.10,
            max_features=1,
            multicollinearity_threshold=0.95,
        )
        # With max_features=1, coverage-weighted should prefer the dense feature
        assert result.selected_features == ("moderate_corr_dense",)

    def test_filters_by_significance(self):
        """Feature with correlation above threshold but p > alpha is excluded."""
        rng = np.random.default_rng(7)
        n = 10  # small sample → high p-value
        target = list(range(n))
        # correlation ~0.5 but with tiny n the p-value > 0.05
        feature_vals = [v * 0.5 + rng.standard_normal() for v in target]

        frame = _make_frame(target=target, maybe_corr=feature_vals)

        # Check that spearmanr actually gives p > 0.05 for this sample size
        _, p = spearmanr(target, feature_vals)
        if p <= 0.05:
            pytest.skip("Random seed produced significant p-value; test not meaningful.")

        result = select_predictors(
            frame,
            target="target",
            mode="auto",
            min_shared_samples=5,
            min_target_correlation=0.3,
            significance_alpha=0.05,
            max_features=5,
        )
        assert "maybe_corr" not in result.selected_features
        ct = result.candidate_table
        reason = ct.loc[ct["feature"] == "maybe_corr", "exclusion_reason"].iloc[0]
        assert reason == "not_significant"

    def test_rejects_forced_features(self):
        """auto mode raises ValueError if forced_features is provided."""
        frame = _make_frame(target=[1, 2, 3, 4, 5], feat=[5, 4, 3, 2, 1])
        with pytest.raises(ValueError, match="mode='auto'"):
            select_predictors(frame, target="target", mode="auto", forced_features=["feat"])

    def test_candidate_table_contains_all_indicator_features(self):
        """candidate_table must have a row for every non-target indicator."""
        rng = np.random.default_rng(1)
        n = 50
        target = rng.standard_normal(n)
        frame = _make_frame(
            target=list(target),
            feat_a=list(target + rng.standard_normal(n) * 0.1),
            feat_b=list(rng.standard_normal(n)),
            feat_c=list(target * (-1)),
        )
        result = select_predictors(
            frame, target="target", mode="auto",
            min_shared_samples=10, min_target_correlation=0.3, significance_alpha=0.05,
        )
        table_features = set(result.candidate_table["feature"].tolist())
        expected = {"feat_a", "feat_b", "feat_c"}
        assert expected == table_features
        # Every row has an exclusion_reason (empty string means included)
        for _, row in result.candidate_table.iterrows():
            assert isinstance(row["exclusion_reason"], str)

    def test_multicollinear_feature_gets_reason(self):
        """Highly collinear second feature gets 'multicollinear_with:' reason."""
        rng = np.random.default_rng(2)
        n = 60
        base = rng.standard_normal(n)
        target = list(base * 0.9 + rng.standard_normal(n) * 0.05)
        feat_a = list(base + rng.standard_normal(n) * 0.02)
        feat_b = list(base + rng.standard_normal(n) * 0.02)  # nearly identical to feat_a

        frame = _make_frame(target=target, feat_a=feat_a, feat_b=feat_b)
        result = select_predictors(
            frame, target="target", mode="auto",
            min_shared_samples=10, min_target_correlation=0.3, significance_alpha=0.10,
            max_features=5, multicollinearity_threshold=0.85,
        )
        ct = result.candidate_table
        rejected = ct[ct["exclusion_reason"].str.startswith("multicollinear_with:")]
        assert len(rejected) >= 1

    def test_empty_candidate_table_when_no_candidates(self):
        """When no features pass filters, selected_features is empty."""
        frame = _make_frame(target=[1.0] * 30, low_corr=[float(i % 5) for i in range(30)])
        result = select_predictors(
            frame, target="target", mode="auto",
            min_shared_samples=20, min_target_correlation=0.9,
        )
        assert result.selected_features == ()


# ---------------------------------------------------------------------------
# 2. manual mode
# ---------------------------------------------------------------------------

class TestManualMode:
    def _base_frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(10)
        n = 40
        t = rng.standard_normal(n)
        return _make_frame(
            target=list(t),
            iron=list(t * 0.1 + rng.standard_normal(n)),  # weak corr — would be filtered in auto
            color=list(t * 0.8 + rng.standard_normal(n) * 0.1),
            ph=list(rng.standard_normal(n)),
        )

    def test_uses_exactly_the_forced_list(self):
        """manual mode includes exactly forced features regardless of correlation."""
        frame = self._base_frame()
        result = select_predictors(
            frame, target="target", mode="manual",
            forced_features=["iron"],
            min_shared_samples=10,
        )
        assert result.selected_features == ("iron",)

    def test_rejects_target_in_forced(self):
        frame = self._base_frame()
        with pytest.raises(ValueError, match="leakage"):
            select_predictors(frame, target="target", mode="manual", forced_features=["target"])

    def test_rejects_unknown_feature(self):
        frame = self._base_frame()
        with pytest.raises(ValueError, match="no_such_feature"):
            select_predictors(frame, target="target", mode="manual", forced_features=["no_such_feature"])

    def test_rejects_empty_forced(self):
        frame = self._base_frame()
        with pytest.raises(ValueError, match="requires at least one"):
            select_predictors(frame, target="target", mode="manual", forced_features=[])

    def test_skips_constant_forced_feature_with_diagnostic(self):
        """A constant forced feature is excluded but captured in candidate_table."""
        frame = self._base_frame()
        frame["constant_feat"] = 1.0  # constant
        result = select_predictors(
            frame, target="target", mode="manual",
            forced_features=["constant_feat"],
            min_shared_samples=5,
        )
        assert "constant_feat" not in result.selected_features
        ct = result.candidate_table
        row = ct[ct["feature"] == "constant_feat"]
        assert not row.empty
        assert row["exclusion_reason"].iloc[0] == "constant_feature"

    def test_non_forced_features_get_not_in_manual_list(self):
        frame = self._base_frame()
        result = select_predictors(
            frame, target="target", mode="manual",
            forced_features=["iron"],
            min_shared_samples=10,
        )
        ct = result.candidate_table
        for _, row in ct.iterrows():
            feature = row["feature"]
            if feature == "iron":
                assert row["included"] is True or row["included"] == True
            elif row["exclusion_reason"] not in ("constant_feature", "too_few_shared_samples"):
                assert row["exclusion_reason"] == "not_in_manual_list", (
                    f"Feature '{feature}' should be 'not_in_manual_list' but is '{row['exclusion_reason']}'"
                )

    def test_candidate_table_contains_all_indicator_features_manual(self):
        frame = self._base_frame()
        result = select_predictors(
            frame, target="target", mode="manual",
            forced_features=["iron"],
            min_shared_samples=10,
        )
        table_features = set(result.candidate_table["feature"].tolist())
        assert {"iron", "color", "ph"} == table_features


# ---------------------------------------------------------------------------
# 3. semi_auto mode
# ---------------------------------------------------------------------------

class TestSemiAutoMode:
    def _base_frame(self, n: int = 60) -> pd.DataFrame:
        rng = np.random.default_rng(20)
        t = rng.standard_normal(n)
        return _make_frame(
            target=list(t),
            iron=list(t * 0.5 + rng.standard_normal(n) * 0.3),
            color=list(t * 0.7 + rng.standard_normal(n) * 0.2),
            ph=list(t * 0.4 + rng.standard_normal(n) * 0.4),
            turbidity=list(t * 0.3 + rng.standard_normal(n) * 0.5),
        )

    def test_includes_forced_then_fills_budget(self):
        frame = self._base_frame()
        result = select_predictors(
            frame, target="target", mode="semi_auto",
            forced_features=["iron"],
            min_shared_samples=10,
            min_target_correlation=0.2,
            significance_alpha=0.10,
            max_features=3,
            multicollinearity_threshold=0.95,
        )
        assert "iron" in result.selected_features
        assert len(result.selected_features) <= 3
        # iron is forced, so it must be first
        assert result.selected_features[0] == "iron"

    def test_keeps_forced_even_if_multicollinear_with_auto_candidate(self):
        """Forced feature is never dropped for multicollinearity."""
        rng = np.random.default_rng(30)
        n = 60
        base = rng.standard_normal(n)
        target = list(base * 0.8 + rng.standard_normal(n) * 0.1)
        forced = list(base + rng.standard_normal(n) * 0.05)   # high corr with target
        auto_cand = list(base + rng.standard_normal(n) * 0.05)  # high corr with forced (collinear)

        frame = _make_frame(target=target, forced_feat=forced, auto_feat=auto_cand)
        result = select_predictors(
            frame, target="target", mode="semi_auto",
            forced_features=["forced_feat"],
            min_shared_samples=10,
            min_target_correlation=0.3,
            significance_alpha=0.10,
            max_features=5,
            multicollinearity_threshold=0.85,
        )
        assert "forced_feat" in result.selected_features

    def test_drops_auto_candidate_collinear_with_forced(self):
        """Auto candidate that is collinear with a forced feature is dropped."""
        rng = np.random.default_rng(31)
        n = 70
        base = rng.standard_normal(n)
        target = list(base * 0.7 + rng.standard_normal(n) * 0.2)
        forced = list(base + rng.standard_normal(n) * 0.02)    # selected as forced
        collinear_auto = list(base + rng.standard_normal(n) * 0.02)  # nearly identical to forced

        frame = _make_frame(target=target, forced_feat=forced, collinear_auto=collinear_auto)
        result = select_predictors(
            frame, target="target", mode="semi_auto",
            forced_features=["forced_feat"],
            min_shared_samples=10,
            min_target_correlation=0.3,
            significance_alpha=0.10,
            max_features=5,
            multicollinearity_threshold=0.85,
        )
        assert "collinear_auto" not in result.selected_features
        ct = result.candidate_table
        row = ct[ct["feature"] == "collinear_auto"]
        assert not row.empty
        assert row["exclusion_reason"].iloc[0].startswith("multicollinear_with:")

    def test_candidate_table_contains_all_indicator_features_semi_auto(self):
        frame = self._base_frame()
        result = select_predictors(
            frame, target="target", mode="semi_auto",
            forced_features=["iron"],
            min_shared_samples=10,
            min_target_correlation=0.2,
            significance_alpha=0.10,
        )
        table_features = set(result.candidate_table["feature"].tolist())
        assert {"iron", "color", "ph", "turbidity"} == table_features


# ---------------------------------------------------------------------------
# 4. FeatureSelectionResult fields
# ---------------------------------------------------------------------------

class TestResultFields:
    def test_auto_result_has_correct_mode_and_empty_forced(self):
        frame = _make_frame(
            target=[1.0, 2.0, 3.0, 4.0, 5.0] * 8,
            feat=[5.0, 4.0, 3.0, 2.0, 1.0] * 8,
        )
        result = select_predictors(
            frame, target="target", mode="auto",
            min_shared_samples=10, min_target_correlation=0.3,
        )
        assert result.selection_mode == "auto"
        assert result.forced_features == ()

    def test_manual_result_has_correct_mode_and_forced(self):
        frame = _make_frame(
            target=[1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            iron=[2.0, 3.0, 4.0, 5.0, 6.0] * 10,
        )
        result = select_predictors(
            frame, target="target", mode="manual",
            forced_features=["iron"],
            min_shared_samples=10,
        )
        assert result.selection_mode == "manual"
        assert result.forced_features == ("iron",)

    def test_is_forced_column_set_correctly(self):
        frame = _make_frame(
            target=[float(i) for i in range(50)],
            iron=[float(i) + 0.1 for i in range(50)],
            color=[float(50 - i) for i in range(50)],
        )
        result = select_predictors(
            frame, target="target", mode="manual",
            forced_features=["iron"],
            min_shared_samples=10,
        )
        ct = result.candidate_table
        iron_row = ct[ct["feature"] == "iron"]
        assert bool(iron_row["is_forced"].iloc[0]) is True
        color_row = ct[ct["feature"] == "color"]
        assert bool(color_row["is_forced"].iloc[0]) is False

    def test_included_column_matches_selected_features(self):
        rng = np.random.default_rng(99)
        n = 50
        t = rng.standard_normal(n)
        frame = _make_frame(
            target=list(t),
            feat_a=list(t * 0.8 + rng.standard_normal(n) * 0.1),
            feat_b=list(rng.standard_normal(n)),
        )
        result = select_predictors(
            frame, target="target", mode="auto",
            min_shared_samples=10, min_target_correlation=0.3,
        )
        ct = result.candidate_table
        included_from_table = set(ct[ct["included"] == True]["feature"].tolist())
        assert included_from_table == set(result.selected_features)


# ---------------------------------------------------------------------------
# diagnose_no_predictors: explain why selection yielded nothing
# ---------------------------------------------------------------------------

class TestDiagnoseNoPredictors:
    def test_auto_all_below_correlation(self):
        """Auto mode with only weak predictors: reasons + best correlation reported."""
        from water_analysis.analysis.feature_selection import diagnose_no_predictors

        rng = np.random.default_rng(0)
        n = 60
        target = rng.standard_normal(n)
        frame = _make_frame(
            target=list(target),
            weak_a=list(rng.standard_normal(n)),
            weak_b=list(rng.standard_normal(n)),
        )
        result = select_predictors(
            frame, target="target", mode="auto",
            min_shared_samples=10, min_target_correlation=0.9, significance_alpha=0.05,
        )
        assert result.selected_features == ()

        diagnosis = diagnose_no_predictors(result)
        assert diagnosis.selection_mode == "auto"
        assert diagnosis.total_candidates == 2
        assert diagnosis.forced_details == ()
        reasons = dict(diagnosis.reason_counts)
        assert reasons.get("below_min_correlation", 0) >= 1
        assert diagnosis.best_abs_correlation is not None

    def test_manual_forced_features_too_few_shared(self):
        """Manual mode where forced features have too few shared samples."""
        from water_analysis.analysis.feature_selection import diagnose_no_predictors

        n = 40
        target_vals = list(np.linspace(0, 1, n))
        sparse = [1.0, 2.0, 3.0] + [float("nan")] * (n - 3)
        frame = _make_frame(target=target_vals, iron=sparse)
        result = select_predictors(
            frame, target="target", mode="manual", forced_features=["iron"],
            min_shared_samples=20,
        )
        assert result.selected_features == ()

        diagnosis = diagnose_no_predictors(result)
        assert diagnosis.selection_mode == "manual"
        assert diagnosis.forced_features == ("iron",)
        feature, base_code, n_shared = diagnosis.forced_details[0]
        assert feature == "iron"
        assert base_code == "too_few_shared_samples"
        assert n_shared == 3

    def test_empty_candidate_table(self):
        """No candidate indicators at all yields a zero-candidate diagnosis."""
        from water_analysis.analysis.feature_selection import (
            FeatureSelectionResult,
            diagnose_no_predictors,
        )

        result = FeatureSelectionResult(
            target="target",
            selected_features=(),
            candidate_table=pd.DataFrame(),
            dropped_multicollinear=(),
            selection_mode="auto",
            forced_features=(),
        )
        diagnosis = diagnose_no_predictors(result)
        assert diagnosis.total_candidates == 0
        assert diagnosis.reason_counts == ()
        assert diagnosis.best_abs_correlation is None
