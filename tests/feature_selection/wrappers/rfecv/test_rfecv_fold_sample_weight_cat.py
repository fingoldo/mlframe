"""Regression: RFECV per-fold sample_weight re-slice + cat-index vs fit_features.

P0 (_rfecv_fit_fold.py): when the early-stopping val_cv re-splits X_train down to
true_train_index, the per-fold train sample_weight MUST follow the same re-slice;
otherwise the weight vector is longer than the narrowed X_train and estimator.fit
raises a length mismatch.

P1 (_rfecv_fit_fold.py): cat_feature indices must be computed against fit_features
(= must_include_resolved + current_features, the ACTUAL X_train/X_val column
order), not current_features -- otherwise must_include's prepended columns shift
every index by len(must_include) and a categorical must_include column is dropped.

Both surface together with early_stopping_val_nsplits + sample_weight +
must_include (a categorical column) + cat_features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_rfecv_fold_sample_weight_and_cat_index_with_early_stopping_and_must_include():
    lgb = pytest.importorskip("lightgbm")
    pytest.importorskip("sklearn")
    from mlframe.feature_selection.wrappers import RFECV

    rng = np.random.default_rng(0)
    n = 240
    df = pd.DataFrame(
        {
            "num0": rng.normal(size=n),
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "num3": rng.normal(size=n),
            "cat0": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        }
    )
    y = (df["num0"].to_numpy() + df["num1"].to_numpy() > 0).astype(int)
    sw = rng.uniform(0.5, 1.5, size=n)

    sel = RFECV(
        estimator=lgb.LGBMClassifier(n_estimators=25, verbose=-1),
        cv=2,
        early_stopping_val_nsplits=2,  # forces the per-fold val_cv re-split (P0 + P1 path)
        cat_features=["cat0"],
        must_include=["cat0"],  # prepended -> fit_features = [cat0, num0, ...]
    )
    # Pre-fix this raised: P0 (sample_weight length-mismatch once X_train was
    # narrowed to true_train_index) and/or P1 (cat index computed against
    # current_features instead of the must_include-prepended fit_features).
    sel.fit(df, y, sample_weight=sw)

    assert hasattr(sel, "support_") and len(sel.support_) == df.shape[1]
    kept = [c for c, k in zip(df.columns, sel.support_) if k]
    assert "cat0" in kept, f"must_include column 'cat0' must be retained; kept={kept}"
