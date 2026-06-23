"""PROP1: accuracy_ratio must be row-permutation-invariant on tied scores and
satisfy its documented identity AR == 2*AUC-1 on both tied and distinct data.

Pre-fix the naive cumsum over -score argsort made the CAP area depend on the
arbitrary intra-tie row order, breaking both properties on tie-heavy inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.core import accuracy_ratio, fast_roc_auc


def _tie_heavy_data(seed: int, n: int = 200):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    # Few distinct score levels -> many ties.
    y_score = rng.integers(0, 4, size=n).astype(np.float64)
    return y_true, y_score


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13, 42])
def test_accuracy_ratio_invariant_under_row_permutation_on_ties(seed):
    y_true, y_score = _tie_heavy_data(seed)
    base = accuracy_ratio(y_true, y_score)
    vals = set()
    rng = np.random.default_rng(1000 + seed)
    for _ in range(20):
        perm = rng.permutation(len(y_true))
        vals.add(round(accuracy_ratio(y_true[perm], y_score[perm]), 12))
    assert len(vals) == 1, f"AR not permutation-invariant on ties: {sorted(vals)}"
    assert abs(base - next(iter(vals))) < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13, 42])
def test_accuracy_ratio_equals_2auc_minus_1_on_ties(seed):
    y_true, y_score = _tie_heavy_data(seed)
    ar = accuracy_ratio(y_true, y_score)
    auc = fast_roc_auc(y_true, y_score)
    assert abs(ar - (2.0 * auc - 1.0)) < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13, 42])
def test_accuracy_ratio_equals_2auc_minus_1_on_distinct(seed):
    rng = np.random.default_rng(seed)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score = rng.standard_normal(n)  # continuous -> no ties
    ar = accuracy_ratio(y_true, y_score)
    auc = fast_roc_auc(y_true, y_score)
    assert abs(ar - (2.0 * auc - 1.0)) < 1e-9
