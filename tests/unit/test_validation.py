import pandas as pd

from water_analysis.modeling.validation import (
    chronological_holdout_split,
    compute_combined_score,
    compute_stability_ratio,
    rolling_backtest_splits,
)


def _build_unsorted_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SampleDate": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-04", "2020-01-02", "2020-01-05"]),
            "FullPointCode": ["p1", "p1", "p1", "p1", "p1"],
            "target": [3.0, 1.0, 4.0, 2.0, 5.0],
            "feature_a": [30.0, 10.0, 40.0, 20.0, 50.0],
            "season": ["winter"] * 5,
        }
    )


def test_chronological_holdout_split_sorts_before_splitting() -> None:
    split = chronological_holdout_split(_build_unsorted_frame(), test_size=0.4, min_train_size=2)

    assert split.train_frame["SampleDate"].tolist() == sorted(split.train_frame["SampleDate"].tolist())
    assert split.test_frame["SampleDate"].tolist() == sorted(split.test_frame["SampleDate"].tolist())
    assert split.train_frame["SampleDate"].max() < split.test_frame["SampleDate"].min()


def test_rolling_backtest_splits_are_chronological() -> None:
    splits = rolling_backtest_splits(_build_unsorted_frame(), initial_train_size=2, test_window_size=1, step_size=1)

    assert len(splits) == 3
    assert splits[0].train_frame["SampleDate"].max() < splits[0].test_frame["SampleDate"].min()
    assert splits[1].train_frame["SampleDate"].max() < splits[1].test_frame["SampleDate"].min()
    assert splits[0].train_frame["SampleDate"].min() == pd.Timestamp("2020-01-01")


def test_combined_score_uses_both_metrics() -> None:
    holdout = {"rmse": 0.49}
    backtest = {"rmse": 0.85}
    score, used_fallback = compute_combined_score(holdout, backtest, weight_holdout=0.4, weight_backtest=0.6)
    expected = 0.4 * 0.49 + 0.6 * 0.85
    assert score is not None
    assert abs(score - expected) < 1e-9
    assert used_fallback is False


def test_combined_score_falls_back_to_holdout_when_backtest_empty() -> None:
    holdout = {"rmse": 0.5}
    score, used_fallback = compute_combined_score(holdout, {}, weight_holdout=0.4, weight_backtest=0.6)
    assert score == 0.5
    assert used_fallback is True


def test_combined_score_falls_back_to_holdout_when_backtest_none() -> None:
    holdout = {"rmse": 0.5}
    score, used_fallback = compute_combined_score(holdout, None, weight_holdout=0.4, weight_backtest=0.6)
    assert score == 0.5
    assert used_fallback is True


def test_combined_score_returns_none_without_holdout() -> None:
    score, used_fallback = compute_combined_score({}, {"rmse": 0.8})
    assert score is None
    assert used_fallback is False


def test_stability_ratio_normal_case() -> None:
    ratio = compute_stability_ratio({"rmse": 0.49}, {"rmse": 0.85})
    assert ratio is not None
    assert abs(ratio - 0.85 / 0.49) < 1e-9


def test_stability_ratio_handles_zero_holdout() -> None:
    ratio = compute_stability_ratio({"rmse": 0.0}, {"rmse": 0.85})
    assert ratio is None


def test_stability_ratio_returns_none_when_metrics_missing() -> None:
    assert compute_stability_ratio({}, {"rmse": 0.8}) is None
    assert compute_stability_ratio({"rmse": 0.5}, None) is None
    assert compute_stability_ratio({"rmse": 0.5}, {}) is None
