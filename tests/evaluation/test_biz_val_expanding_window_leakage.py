"""biz_value test for ``evaluation.detect_expanding_window_feature_leakage``.

The win: a category's true "rate" (population frequency) directly drives the target. A frequency-count
feature computed on the FULL dataset is a low-noise estimate of that true rate (many observations). The
SAME feature recomputed honestly per expanding fold (train-only, especially in EARLY folds with few rows)
is a much noisier estimate -- but a naive full-dataset feature-selection pass would see inflated CV scores
in those early folds specifically, an artifact of information leaking backward from later (future) rows.
This mirrors the KKBox 1st place's "leaking information from the future to the past" warning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.evaluation import detect_expanding_window_feature_leakage


def _make_rate_driven_dataset(n: int, n_cats: int, seed: int):
    """Helper that make rate driven dataset."""
    rng = np.random.default_rng(seed)
    cat_rate = rng.uniform(0.5, 5.0, n_cats)
    cat_probs = cat_rate / cat_rate.sum()
    cat = rng.choice(n_cats, size=n, p=cat_probs)
    t = np.arange(n, dtype=float)
    y = cat_rate[cat] * 3.0 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"t": t, "cat": cat}), y


def _frequency_count_fit_transform(fit_df: pd.DataFrame, transform_df: pd.DataFrame) -> np.ndarray:
    """Helper that frequency count fit transform."""
    counts = fit_df["cat"].value_counts()
    return transform_df["cat"].map(counts).fillna(0).to_numpy(dtype=np.float64)


def test_biz_val_detect_expanding_window_feature_leakage_detects_planted_leak():
    """Detect expanding window feature leakage detects planted leak."""
    df, y = _make_rate_driven_dataset(n=4000, n_cats=30, seed=0)
    result = detect_expanding_window_feature_leakage(df, "t", y, _frequency_count_fit_transform, lambda: LinearRegression(), n_splits=5, scoring="r2")

    assert result["leak_detected"] is True
    assert result["inflation"] > 0.03, (
        f"expected the leaky full-dataset feature to score materially higher than the honest per-fold one, got inflation={result['inflation']:.4f}"
    )
    # The inflation should be most visible in the EARLIEST fold, where the honest per-fold count has the
    # fewest observations to estimate the true rate from.
    early_gap = result["leaky_scores"][0] - result["honest_scores"][0]
    late_gap = result["leaky_scores"][-1] - result["honest_scores"][-1]
    assert early_gap > late_gap, f"expected the leaky/honest gap to shrink as folds expand, got early={early_gap:.4f} late={late_gap:.4f}"


def test_detect_expanding_window_feature_leakage_no_leak_when_feature_is_time_invariant():
    """A feature with NO time-dependent information content (pure random noise, unrelated to category
    frequency or time) should show no inflation -- confirms the detector doesn't cry wolf on an honest
    feature."""
    rng = np.random.default_rng(1)
    n = 1000
    df = pd.DataFrame({"t": np.arange(n, dtype=float), "cat": rng.integers(0, 10, n)})
    y = rng.normal(size=n)

    def _random_fit_transform(fit_df: pd.DataFrame, transform_df: pd.DataFrame) -> np.ndarray:
        """Helper that random fit transform."""
        return np.random.default_rng(0).normal(size=len(transform_df))

    result = detect_expanding_window_feature_leakage(df, "t", y, _random_fit_transform, lambda: LinearRegression(), n_splits=4, scoring="r2")
    assert result["leak_detected"] is False


def test_detect_expanding_window_feature_leakage_output_shape():
    """Detect expanding window feature leakage output shape."""
    df, y = _make_rate_driven_dataset(n=500, n_cats=10, seed=2)
    result = detect_expanding_window_feature_leakage(df, "t", y, _frequency_count_fit_transform, lambda: LinearRegression(), n_splits=4)
    assert len(result["leaky_scores"]) == 4
    assert len(result["honest_scores"]) == 4


def test_biz_val_detect_expanding_window_feature_leakage_auto_remediate_eliminates_inflation():
    """Opt-in ``auto_remediate=True`` win: on the same planted-leak dataset that inflates the leaky CV
    score by >0.03 R2, the auto-suggested remediated feature (each row recomputed as-of its own fold
    train-cutoff) must, when fed BACK through the very same detector, show materially LESS inflation than
    the original leaky feature -- proving the suggested recomputation boundary actually fixes the leak
    rather than just labeling it."""
    df, y = _make_rate_driven_dataset(n=4000, n_cats=30, seed=0)
    baseline = detect_expanding_window_feature_leakage(df, "t", y, _frequency_count_fit_transform, lambda: LinearRegression(), n_splits=5, scoring="r2")
    result = detect_expanding_window_feature_leakage(
        df, "t", y, _frequency_count_fit_transform, lambda: LinearRegression(), n_splits=5, scoring="r2", auto_remediate=True
    )

    assert result["remediation_verified"] is True
    assert result["leaky_row_ranges"], "expected at least one flagged leaking row range on a planted-leak dataset"
    for start, end in result["leaky_row_ranges"]:
        assert 0 <= start < end <= len(df)

    remediated_feature = result["remediated_feature"]
    assert remediated_feature.shape == (len(df),)
    assert not np.isnan(remediated_feature).any()

    def _remediated_lookup_fit_transform(fit_df: pd.DataFrame, transform_df: pd.DataFrame) -> np.ndarray:
        """Helper that remediated lookup fit transform."""
        return remediated_feature[transform_df.index.to_numpy()]

    remediated_rerun = detect_expanding_window_feature_leakage(
        df, "t", y, _remediated_lookup_fit_transform, lambda: LinearRegression(), n_splits=5, scoring="r2"
    )
    assert remediated_rerun["leak_detected"] is False
    assert remediated_rerun["inflation"] < baseline["inflation"] - 0.02, (
        f"expected remediation to materially shrink inflation, got baseline={baseline['inflation']:.4f} remediated={remediated_rerun['inflation']:.4f}"
    )


def test_detect_expanding_window_feature_leakage_auto_remediate_default_off_is_bit_identical():
    """``auto_remediate`` is strictly opt-in: omitting it must reproduce the exact pre-extension output
    (same keys, same score values) -- no behavior change for existing callers."""
    df, y = _make_rate_driven_dataset(n=1000, n_cats=15, seed=3)
    default_call = detect_expanding_window_feature_leakage(df, "t", y, _frequency_count_fit_transform, lambda: LinearRegression(), n_splits=4, scoring="r2")
    explicit_off = detect_expanding_window_feature_leakage(
        df, "t", y, _frequency_count_fit_transform, lambda: LinearRegression(), n_splits=4, scoring="r2", auto_remediate=False
    )
    assert set(default_call.keys()) == {"leaky_scores", "honest_scores", "leaky_mean", "honest_mean", "inflation", "leak_detected"}
    assert default_call["leaky_scores"] == explicit_off["leaky_scores"]
    assert default_call["honest_scores"] == explicit_off["honest_scores"]
    assert default_call["inflation"] == explicit_off["inflation"]
    assert default_call["leak_detected"] == explicit_off["leak_detected"]
