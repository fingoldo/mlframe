"""biz_value test for ``feature_engineering.variance_gated_pairwise_diff.variance_gated_pairwise_diff``.

The win (4th_mechanisms-of-action-moa-prediction.md): a combinatorial pairwise-diff generator over a wide
feature block must (a) prune near-constant/near-duplicate-column diffs (which carry no signal and would
otherwise explode the candidate count uselessly), while (b) preserving genuinely informative diff pairs -- a
target that depends on the DIFFERENCE between two individually-uninformative, highly-correlated columns is
invisible in either raw column alone but recoverable from their diff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.variance_gated_pairwise_diff import variance_gated_pairwise_diff


def _make_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    # two near-duplicate columns (correlated common factor); the target depends only on their small
    # difference, which is buried under the shared factor's large variance in either raw column.
    c1 = base + rng.normal(scale=0.05, size=n)
    diff_signal = rng.normal(scale=0.3, size=n)
    c2 = base + diff_signal
    y = (diff_signal > 0).astype(int)

    # a genuinely near-constant pair (identical column up to tiny float noise) -- should be pruned.
    dup1 = rng.normal(size=n)
    dup2 = dup1 + rng.normal(scale=1e-8, size=n)

    df = pd.DataFrame({"c1": c1, "c2": c2, "dup1": dup1, "dup2": dup2})
    return df, y


def test_biz_val_variance_gated_diff_prunes_near_constant_keeps_informative():
    df, y = _make_dataset(n=2000, seed=0)

    diffs = variance_gated_pairwise_diff(df, list(df.columns), min_variance=1e-4)

    assert "dup1__diff__dup2" not in diffs.columns, "expected the near-constant duplicate-column diff to be pruned"
    assert "c1__diff__c2" in diffs.columns, "expected the genuinely-varying diff to survive"

    auc_raw = roc_auc_score(y, LogisticRegression(max_iter=500).fit(df[["c1", "c2"]], y).predict_proba(df[["c1", "c2"]])[:, 1])
    auc_diff = roc_auc_score(y, LogisticRegression(max_iter=500).fit(diffs[["c1__diff__c2"]], y).predict_proba(diffs[["c1__diff__c2"]])[:, 1])

    assert auc_diff > 0.95, f"expected the surviving diff feature to strongly recover the target, got AUC={auc_diff:.4f}"
    assert auc_diff > auc_raw, f"expected the diff feature to beat the raw columns (which bury the signal under a shared large-variance factor), got diff={auc_diff:.4f} raw={auc_raw:.4f}"


def test_variance_gated_diff_column_naming_and_shape():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 1.0, 1.0, 1.0], "c": [5.0, 3.0, 8.0, 1.0]})
    out = variance_gated_pairwise_diff(df, ["a", "b", "c"], min_variance=0.5)
    # a-b: variance of [0,1,2,3] = 1.25 > 0.5 -> kept. a-c: variance of [-4,-1,-5,3] high -> kept.
    # b-c: variance of [-4,-2,-7,0] high -> kept. All should survive at this low threshold.
    assert set(out.columns) == {"a__diff__b", "a__diff__c", "b__diff__c"}
