"""biz_value test for ``feature_engineering.sentinel_missing_count.add_sentinel_missing_count_feature``.

The win (2nd_porto-seguro-safe-driver-prediction.md): many real datasets encode missingness as an explicit
SENTINEL value (-1, -999, ...) rather than true NaN. mlframe's existing NaN-based row-missing-count generator
(``feature_selection.filters._missingness_fe``) is blind to this -- it would report ZERO missing values on a
sentinel-encoded frame, even when the sentinel count is itself a genuinely predictive signal (e.g. rows with
more missing fields correlate with the target, a classic insurance/credit-risk pattern). This test confirms
the sentinel-count feature recovers signal a naive NaN-based (or no) missing-count feature would miss
entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.sentinel_missing_count import add_sentinel_missing_count_feature, detect_sentinel_values


def _make_sentinel_encoded_dataset(n: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    # missing_rate driven by y itself (rows with the target condition tend to have more sentinel-encoded
    # fields) -- a realistic "missingness is informative" pattern, matching the source's own finding.
    y = rng.integers(0, 2, n)
    missing_rate_per_row = 0.05 + 0.35 * y  # y=1 rows have far more sentinel fields
    mask = rng.random((n, n_features)) < missing_rate_per_row[:, None]
    X_sentinel = np.where(mask, -1.0, X)  # sentinel value -1, NOT NaN

    df = pd.DataFrame(X_sentinel, columns=[f"f{i}" for i in range(n_features)])
    return df, y


def test_biz_val_sentinel_missing_count_recovers_signal_naive_nan_count_misses():
    df, y = _make_sentinel_encoded_dataset(n=2000, n_features=15, seed=0)

    # A naive NaN-based missing-count is blind to sentinel-encoded missingness -- the column is ALWAYS 0.
    naive_nan_count = df.isna().sum(axis=1).to_numpy()
    assert (naive_nan_count == 0).all(), "sanity check: the synthetic sentinel-encoded frame has no true NaNs"

    df_with_sentinel_count = add_sentinel_missing_count_feature(df, sentinel=-1.0)
    sentinel_count_only = df_with_sentinel_count[["sentinel_missing_count"]].to_numpy()

    auc_sentinel_count_alone = cross_val_score(LogisticRegression(max_iter=500), sentinel_count_only, y, cv=5, scoring="roc_auc").mean()
    auc_naive_nan_count_alone = 0.5  # a constant all-zero feature carries zero information -> exactly chance

    assert auc_sentinel_count_alone > 0.75, f"expected the sentinel-count feature alone to carry strong signal, got AUC={auc_sentinel_count_alone:.4f}"
    assert auc_sentinel_count_alone > auc_naive_nan_count_alone + 0.2


def test_add_sentinel_missing_count_feature_exact_counts():
    df = pd.DataFrame({"a": [-1, 0, -1], "b": [-1, -1, 5], "c": [3, -1, -1]})
    out = add_sentinel_missing_count_feature(df, sentinel=-1)
    np.testing.assert_array_equal(out["sentinel_missing_count"], [2, 2, 2])


def test_add_sentinel_missing_count_feature_respects_column_subset():
    df = pd.DataFrame({"a": [-1, -1], "b": [-1, 5]})
    out = add_sentinel_missing_count_feature(df, sentinel=-1, columns=["a"])
    np.testing.assert_array_equal(out["sentinel_missing_count"], [1, 1])


def _make_per_column_sentinel_dataset(n: int, seed: int):
    """Two blocks of columns with DIFFERENT sentinel codes (-1 vs -999), both target-informative.

    A single global-sentinel call can only match one code at a time -- it either misses the other block
    entirely (undercounting missingness, weaker signal) or, if the global sentinel happens to also be a
    legitimate value in the other block, silently corrupts that block's counts. Per-column sentinels resolve
    both blocks correctly in one call.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    missing_rate = 0.05 + 0.35 * y

    block_a = rng.normal(size=(n, 8))  # sentinel -1
    mask_a = rng.random((n, 8)) < missing_rate[:, None]
    block_a = np.where(mask_a, -1.0, block_a)

    block_b = rng.normal(loc=50, size=(n, 8))  # sentinel -999, and -1 is a perfectly legitimate value here
    mask_b = rng.random((n, 8)) < missing_rate[:, None]
    block_b = np.where(mask_b, -999.0, block_b)

    cols_a = [f"a{i}" for i in range(8)]
    cols_b = [f"b{i}" for i in range(8)]
    df = pd.DataFrame(np.hstack([block_a, block_b]), columns=cols_a + cols_b)
    return df, y, cols_a, cols_b


def test_biz_val_add_sentinel_missing_count_feature_per_column_sentinels_beats_global():
    df, y, cols_a, cols_b = _make_per_column_sentinel_dataset(n=2000, seed=1)

    # A single global sentinel (-1) only ever matches block "a" -- block "b"'s -999 codes are invisible to it.
    global_only = add_sentinel_missing_count_feature(df, sentinel=-1.0)
    auc_global = cross_val_score(LogisticRegression(max_iter=500), global_only[["sentinel_missing_count"]].to_numpy(), y, cv=5, scoring="roc_auc").mean()

    per_column = add_sentinel_missing_count_feature(
        df,
        per_column_sentinels={**{c: -1.0 for c in cols_a}, **{c: -999.0 for c in cols_b}},
    )
    auc_per_column = cross_val_score(LogisticRegression(max_iter=500), per_column[["sentinel_missing_count"]].to_numpy(), y, cv=5, scoring="roc_auc").mean()

    assert auc_per_column > 0.85, f"expected per-column sentinel resolution to carry strong signal, got AUC={auc_per_column:.4f}"
    # Measured gap on this fixture is ~0.036 (AUC_per_column~0.995 vs AUC_global~0.960); threshold set below that with margin.
    assert auc_per_column > auc_global + 0.02, (
        f"per-column sentinels (AUC={auc_per_column:.4f}) should beat a single global sentinel that misses half "
        f"the sentinel-coded columns (AUC={auc_global:.4f})"
    )


def test_add_sentinel_missing_count_feature_per_column_sentinels_exact_counts():
    df = pd.DataFrame({"a": [-1, 0, -1], "b": [-999, -999, 5]})
    out = add_sentinel_missing_count_feature(df, per_column_sentinels={"a": -1, "b": -999})
    np.testing.assert_array_equal(out["sentinel_missing_count"], [2, 1, 1])


def test_add_sentinel_missing_count_feature_per_column_multi_value_sentinel():
    df = pd.DataFrame({"a": [-1, -999, 3], "b": [5, 6, 7]})
    out = add_sentinel_missing_count_feature(df, per_column_sentinels={"a": [-1, -999]})
    np.testing.assert_array_equal(out["sentinel_missing_count"], [1, 1, 0])


def test_add_sentinel_missing_count_feature_omitted_new_params_matches_original_output():
    """Opt-in guard: omitting the new params must reproduce the exact original single-global-sentinel output."""
    df = pd.DataFrame({"a": [-1, 0, -1], "b": [-1, -1, 5], "c": [3, -1, -1]})
    out_new_default = add_sentinel_missing_count_feature(df, sentinel=-1)
    out_explicit_old_signature = add_sentinel_missing_count_feature(df, -1, None, "sentinel_missing_count")
    pd.testing.assert_frame_equal(out_new_default, out_explicit_old_signature)
    np.testing.assert_array_equal(out_new_default["sentinel_missing_count"], [2, 2, 2])


def test_biz_val_add_sentinel_missing_count_feature_auto_detect_sentinels():
    df, y, cols_a, cols_b = _make_per_column_sentinel_dataset(n=2000, seed=2)

    detected = detect_sentinel_values(df, columns=cols_a + cols_b)
    assert all(detected.get(c) == -1.0 for c in cols_a), f"expected -1.0 auto-detected on block a, got {detected}"
    assert all(detected.get(c) == -999.0 for c in cols_b), f"expected -999.0 auto-detected on block b, got {detected}"

    auto = add_sentinel_missing_count_feature(df, auto_detect_sentinels=True)
    auc_auto = cross_val_score(LogisticRegression(max_iter=500), auto[["sentinel_missing_count"]].to_numpy(), y, cv=5, scoring="roc_auc").mean()

    global_only = add_sentinel_missing_count_feature(df, sentinel=-1.0)
    auc_global = cross_val_score(LogisticRegression(max_iter=500), global_only[["sentinel_missing_count"]].to_numpy(), y, cv=5, scoring="roc_auc").mean()

    assert auc_auto > 0.85, f"expected auto-detected per-column sentinels to carry strong signal, got AUC={auc_auto:.4f}"
    # Measured gap on this fixture is ~0.028 (AUC_auto~0.995 vs AUC_global~0.967); threshold set below that with margin.
    assert auc_auto > auc_global + 0.02, (
        f"auto-detection (AUC={auc_auto:.4f}) should beat a single global sentinel blind to the second block (AUC={auc_global:.4f})"
    )
