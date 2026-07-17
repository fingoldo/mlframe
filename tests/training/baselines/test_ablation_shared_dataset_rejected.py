"""Regression pins for the REJECTED ablation binning-amortization leads.

BaselineDiagnostics ablation runs 6 LightGBM fits per target, each rebuilding a
fresh ``lgb.Dataset`` (binning pass) from the same sampled frame, differing only
by one dropped column. Two LightGBM-idiomatic amortizations were investigated to
bin once and reuse across fits -- BOTH rejected (see
``mlframe/training/_benchmarks/bench_ablation_shared_dataset.py``):

* ``reference=`` subset: INFEASIBLE -- LightGBM requires the child Dataset to
  have the same ``num_feature`` as the reference; a column-drop subset fails
  ``construct()``.
* ``ignore_column``: FEASIBLE but NOT bit-identical -- it changes split
  decisions / feature importances / predictions, so the ablation verdict would
  drift. Forbidden for a default-ON diagnostic.

These tests pin BOTH non-equivalences so a future "just reuse the binned
Dataset" optimization cannot silently change the ablation verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")


def _data(n: int = 3000, nf: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, nf)).astype(np.float64)
    y = (X[:, 0] + 0.5 * X[:, 3] - 0.3 * X[:, 5] + rng.normal(size=n) > 0).astype(int)
    cols = [f"f{i}" for i in range(nf)]
    return pd.DataFrame(X, columns=cols), y, cols


def test_reference_dataset_rejects_column_subset():
    """Lead (b): a full-feature Dataset cannot be used as ``reference`` for a
    column-drop subset -- LightGBM enforces equal ``num_feature``. Pins that the
    bin-mapper-reuse-via-reference path is infeasible for ablation drops.
    """
    Xdf, y, cols = _data()
    full = lgb.Dataset(Xdf, label=y, free_raw_data=False)
    full.construct()
    keep = [c for c in cols if c != cols[0]]
    sub = lgb.Dataset(Xdf[keep], label=y, reference=full, free_raw_data=False)
    with pytest.raises(Exception) as excinfo:
        sub.construct()
    # The failure is specifically a feature-count mismatch, not an unrelated error.
    assert "num_feature" in str(excinfo.value) or "feature_name" in str(excinfo.value)


def test_ignore_column_not_bit_identical_to_drop():
    """Lead (a): training with ``ignore_column`` on the full Dataset is NOT
    equivalent to dropping that column -- predictions diverge meaningfully, so
    the ablation delta (and dominant-feature verdict) would change. Pins the
    non-equivalence so nobody ships ignore_column as a binning shortcut.
    """
    Xdf, y, cols = _data()
    base = dict(num_leaves=31, learning_rate=0.05, verbose=-1, seed=42, force_col_wise=True, deterministic=True)
    keep = [c for c in cols if c != cols[0]]
    m = lgb.LGBMClassifier(n_estimators=40, **base)
    m.fit(Xdf[keep], y)
    p_drop = m.predict_proba(Xdf[keep])[:, 1]

    ds = lgb.Dataset(Xdf, label=y, free_raw_data=False, params={"ignore_column": "0"})
    booster = lgb.train({**base, "objective": "binary", "num_iterations": 40}, ds)
    p_ignore = booster.predict(Xdf)

    max_abs_div = float(np.max(np.abs(p_drop - p_ignore)))
    # Selection/score-altering divergence (>> 1e-9 reduction-order noise).
    assert max_abs_div > 1e-2, (
        "ignore_column unexpectedly matched column-drop; if a LightGBM version makes this bit-identical, re-open the amortization lead with a bench."
    )
