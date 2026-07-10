"""Adversarial-validation-driven categorical rebinning.

A categorical column can pass ``train_test_support_screen``'s category-SET overlap check yet still be a
strong adversarial-validation signal because individual category VALUES occur at very different frequencies
in train vs test (e.g. a version string that shifted in prevalence over time) -- a distinct failure mode from
missing/unseen categories. The remedy (from a 6th-place Microsoft-malware writeup that dropped adversarial AUC
from 0.98 to under 0.7) is to merge the most frequency-skewed values into a shared bucket, shrinking the
train/test distributional gap those specific values create without discarding the column or losing signal
from the well-behaved majority of its categories.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def adversarial_rebin_categorical(
    train_series: pd.Series,
    test_series: pd.Series,
    skew_log_ratio_threshold: float = 1.0,
    min_count: int = 1,
    other_label: str = "__REBINNED_OTHER__",
) -> dict:
    """Merge the most train/test-frequency-skewed category values of one column into a shared bucket.

    Parameters
    ----------
    train_series, test_series
        The same categorical column from the train and test frames.
    skew_log_ratio_threshold
        A category is merged into ``other_label`` when ``|log((test_freq + eps) / (train_freq + eps))|``
        exceeds this threshold -- i.e. its relative prevalence differs by more than
        ``exp(skew_log_ratio_threshold)``x between train and test.
    min_count
        Categories with fewer than this many TOTAL (train + test) occurrences are also merged (too rare to
        estimate a reliable frequency ratio from either side).
    other_label
        The bucket label all flagged categories are mapped to.

    Returns
    -------
    dict
        ``train_rebinned``/``test_rebinned`` (the transformed Series), ``merged_categories`` (list of the
        original values folded into ``other_label``), ``category_skew`` (DataFrame with per-category train/test
        frequency and skew, for inspection).
    """
    n_train = len(train_series)
    n_test = len(test_series)
    train_counts = train_series.value_counts()
    test_counts = test_series.value_counts()
    all_cats = sorted(set(train_counts.index.tolist()) | set(test_counts.index.tolist()), key=str)

    eps = 1e-6
    rows = []
    merged = []
    for cat in all_cats:
        c_train = int(train_counts.get(cat, 0))
        c_test = int(test_counts.get(cat, 0))
        freq_train = c_train / n_train if n_train else 0.0
        freq_test = c_test / n_test if n_test else 0.0
        skew = float(np.log((freq_test + eps) / (freq_train + eps)))
        should_merge = abs(skew) > skew_log_ratio_threshold or (c_train + c_test) < min_count
        rows.append(
            {
                "category": cat,
                "train_count": c_train,
                "test_count": c_test,
                "train_freq": freq_train,
                "test_freq": freq_test,
                "log_skew": skew,
                "merged": should_merge,
            }
        )
        if should_merge:
            merged.append(cat)

    # vectorized isin+where instead of a per-row Python callback (.apply) -- the same "per-row Python
    # callback on a large Series" cost class flagged elsewhere; a plain membership mask is O(n) in C, not
    # O(n) Python-dispatch calls.
    merged_index = pd.Index(merged)
    train_rebinned = train_series.where(~train_series.isin(merged_index), other_label)
    test_rebinned = test_series.where(~test_series.isin(merged_index), other_label)

    return {
        "train_rebinned": train_rebinned,
        "test_rebinned": test_rebinned,
        "merged_categories": merged,
        "category_skew": pd.DataFrame(rows),
    }


__all__ = ["adversarial_rebin_categorical"]
