"""A single-class target arriving through an imbalanced split must fail fast with the class count (mrmr_audit_2026-09-14 TESTGAP-9).

A constant ``y`` was already covered, but only as a literal constant. The realistic route is a split of a rare-class frame that leaves one
class in the training part. H(y) = 0 then, every MI is 0 by construction, and a rare-class code path dividing by (n_classes - 1) would pick
columns arbitrarily. The contract is the same fit-time ValueError, whatever the label dtype.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _rare_class_frame(n=2000, seed=0):
    """A frame whose target has 1% positives, and the negatives-only slice a naive split can produce."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(4)})
    y = np.zeros(n, dtype=np.int64)
    y[rng.choice(n, size=n // 100, replace=False)] = 1
    negatives = y == 0
    return X[negatives].reset_index(drop=True), y[negatives]


@pytest.mark.parametrize("label_kind", ["int", "bool", "str", "category"])
def test_single_class_training_slice_raises_naming_the_class_count(label_kind):
    """Every label dtype must hit the same fit-time guard, and its message must state the observed class count."""
    X, y = _rare_class_frame()
    labels = {
        "int": pd.Series(y),
        "bool": pd.Series(y.astype(bool)),
        "str": pd.Series(np.where(y == 1, "pos", "neg")),
        "category": pd.Series(pd.Categorical(np.where(y == 1, "pos", "neg"), categories=["neg", "pos"])),
    }[label_kind]
    assert labels.nunique() == 1, "fixture precondition: the slice must hold exactly one class"
    with pytest.raises(ValueError, match="only 1 unique value"):
        MRMR(verbose=0, fe_max_steps=0).fit(X, labels)


def test_two_class_slice_still_fits():
    """Control: the same frame with its rare positives kept fits normally, so the guard is about the class count and nothing else."""
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(4)})
    y = (X["f0"] + 0.3 * rng.normal(size=n) > 2.0).astype(int)
    assert 0 < y.sum() < n
    MRMR(verbose=0, fe_max_steps=0).fit(X, y)
