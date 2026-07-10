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
