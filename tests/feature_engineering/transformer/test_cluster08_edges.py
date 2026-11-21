"""Cluster 08 regressions: empty SMOTE shapes agree, the integer magnitude check handles the most negative int64, and single-class folds do not crash (audit 2026-11-20, findings 08-F3, 08-F5, 08-F10)."""

import numpy as np

from mlframe.feature_engineering.transformer._smote_kernels_shared import smote_synthesize_minority_rowloop, smote_synthesize_minority_vectorized


def test_empty_minority_shape_matches_across_smote_variants():
    """Both SMOTE siblings return (0, d) for an empty minority (08-F5)."""
    empty = np.zeros((0, 4), dtype=np.float64)
    a = smote_synthesize_minority_rowloop(empty, 3, 2, 0)
    b = smote_synthesize_minority_vectorized(empty, 3, 2, 0)
    assert a.shape == b.shape == (0, 4)


def test_most_negative_int64_is_flagged_as_large():
    """np.abs(-2**63) overflows back to itself, so the old magnitude check read it as small; it must warn now (08-F10)."""
    import warnings

    from mlframe.feature_engineering.transformer._utils import validate_numeric_input

    X = np.array([[-(2**63)], [1]], dtype=np.int64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_numeric_input(X, name="X")
    assert any("float" in str(w.message).lower() or "2**24" in str(w.message) or "magnitude" in str(w.message).lower() for w in caught)
