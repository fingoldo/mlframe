"""biz_value test for ``preprocessing.adversarial_rebin_categorical``.

The win: a categorical column with a handful of extremely train/test-skewed values should show a
substantially lower adversarial-validation AUC after those specific values are merged into a shared bucket,
while the well-behaved majority of categories (and the signal they carry) is left untouched -- the source
writeup's own reported effect (adversarial AUC dropped from ~0.98 to under 0.7 after this kind of rebinning).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.preprocessing.adversarial_rebin import adversarial_rebin_categorical
from mlframe.reporting.charts.drift import adversarial_auc


def _make_skewed_column(seed: int):
    rng = np.random.default_rng(seed)
    n_train, n_test = 4000, 4000

    balanced_cats = [f"cat_{i}" for i in range(20)]
    train_balanced = rng.choice(balanced_cats, size=n_train - 1200)
    test_balanced = rng.choice(balanced_cats, size=n_test - 1200)

    # a chunk of categories that occur almost exclusively in one side -- the adversarial-detectable artifact.
    train_skewed = rng.choice(["skewed_train_only_1", "skewed_train_only_2", "skewed_train_only_3"], size=1200)
    test_skewed = rng.choice(["skewed_test_only_1", "skewed_test_only_2", "skewed_test_only_3"], size=1200)

    train_vals = np.concatenate([train_balanced, train_skewed])
    test_vals = np.concatenate([test_balanced, test_skewed])
    rng.shuffle(train_vals)
    rng.shuffle(test_vals)
    return pd.Series(train_vals, name="cat_col"), pd.Series(test_vals, name="cat_col")


def test_biz_val_adversarial_rebin_reduces_adversarial_auc():
    train_series, test_series = _make_skewed_column(seed=0)

    auc_before, _fpr, _tpr, _imp, _names = adversarial_auc(
        pd.DataFrame({"cat_col": train_series}), pd.DataFrame({"cat_col": test_series}), seed=0
    )

    result = adversarial_rebin_categorical(train_series, test_series, skew_log_ratio_threshold=1.5)
    assert len(result["merged_categories"]) > 0
    assert set(result["merged_categories"]) >= {
        "skewed_train_only_1",
        "skewed_train_only_2",
        "skewed_train_only_3",
        "skewed_test_only_1",
        "skewed_test_only_2",
        "skewed_test_only_3",
    }

    auc_after, _fpr, _tpr, _imp, _names = adversarial_auc(
        pd.DataFrame({"cat_col": result["train_rebinned"]}), pd.DataFrame({"cat_col": result["test_rebinned"]}), seed=0
    )

    assert auc_after < auc_before - 0.1, (
        f"rebinning the most skewed categories should substantially reduce adversarial AUC: "
        f"before={auc_before:.4f} after={auc_after:.4f}"
    )


def test_adversarial_rebin_leaves_balanced_categories_untouched():
    train_series, test_series = _make_skewed_column(seed=1)
    result = adversarial_rebin_categorical(train_series, test_series, skew_log_ratio_threshold=1.5)
    for cat in [f"cat_{i}" for i in range(20)]:
        assert cat not in result["merged_categories"]


def test_adversarial_rebin_category_skew_report_has_expected_columns():
    train_series, test_series = _make_skewed_column(seed=2)
    result = adversarial_rebin_categorical(train_series, test_series, skew_log_ratio_threshold=1.5)
    report = result["category_skew"]
    assert {"category", "train_count", "test_count", "train_freq", "test_freq", "log_skew", "merged"} <= set(report.columns)


def test_adversarial_rebin_categorical_mode_default_is_bit_identical():
    """Guards that the new opt-in continuous mode is truly opt-in: the default call path (positional/keyword
    args identical to the pre-extension signature) must reproduce the exact original single-pass output."""
    train_series, test_series = _make_skewed_column(seed=0)
    result_default = adversarial_rebin_categorical(train_series, test_series, skew_log_ratio_threshold=1.5)
    result_explicit = adversarial_rebin_categorical(
        train_series, test_series, skew_log_ratio_threshold=1.5, mode="categorical"
    )
    pd.testing.assert_series_equal(result_default["train_rebinned"], result_explicit["train_rebinned"])
    pd.testing.assert_series_equal(result_default["test_rebinned"], result_explicit["test_rebinned"])
    assert result_default["merged_categories"] == result_explicit["merged_categories"]
    assert "bin_edges" not in result_default


def _make_drifting_numeric_column(seed: int):
    """A numeric feature whose overall quantile SHAPE is identical in train vs test, but whose upper-tail
    density shifted between the two -- undetectable as a "category" issue (every raw float is near-unique,
    so the categorical path would either merge almost nothing at min_count or explode/merge nearly everything)
    but visible once the range is quantile-binned."""
    rng = np.random.default_rng(seed)
    n_train, n_test = 4000, 4000

    train_base = rng.normal(loc=0.0, scale=1.0, size=n_train - 800)
    test_base = rng.normal(loc=0.0, scale=1.0, size=n_test - 800)

    # a tail chunk sampled from a shifted range -- present in both sides overall, but concentrated in a
    # different quantile band per side, exactly the density-shift-within-shared-bins failure mode.
    train_tail = rng.uniform(low=3.0, high=4.0, size=800)
    test_tail = rng.uniform(low=5.0, high=6.0, size=800)

    train_vals = np.concatenate([train_base, train_tail])
    test_vals = np.concatenate([test_base, test_tail])
    rng.shuffle(train_vals)
    rng.shuffle(test_vals)
    return pd.Series(train_vals, name="num_col"), pd.Series(test_vals, name="num_col")


def test_biz_val_adversarial_rebin_continuous_mode_reduces_adversarial_auc_on_drifting_numeric():
    train_series, test_series = _make_drifting_numeric_column(seed=0)

    auc_before, _fpr, _tpr, _imp, _names = adversarial_auc(
        pd.DataFrame({"num_col": train_series}), pd.DataFrame({"num_col": test_series}), seed=0
    )

    # the categorical-only path cannot help: raw floats are near-unique so merging by raw value either
    # touches nothing (large min_count aside) or requires enumerating the whole column -- it has no notion
    # of "adjacent numeric range", which is exactly what the new continuous mode adds.
    result = adversarial_rebin_categorical(
        train_series, test_series, skew_log_ratio_threshold=1.0, mode="continuous", n_quantile_bins=20
    )
    assert len(result["merged_categories"]) > 0
    assert "bin_edges" in result

    auc_after, _fpr, _tpr, _imp, _names = adversarial_auc(
        pd.DataFrame({"num_col": result["train_rebinned"]}), pd.DataFrame({"num_col": result["test_rebinned"]}), seed=0
    )

    assert auc_after < auc_before - 0.1, (
        f"continuous-mode rebinning should substantially reduce adversarial AUC on a numeric feature whose "
        f"drift the categorical-only path cannot see: before={auc_before:.4f} after={auc_after:.4f}"
    )
