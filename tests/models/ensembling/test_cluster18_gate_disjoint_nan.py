"""Cluster 18 regression: the outlier gate still excludes an outlier when members have disjoint NaN rows (audit 2026-11-20, finding 18-F6).

Member A is NaN on the first rows and member B on the last rows. The gate used to keep only rows where every member was finite, which dropped
all rows, so the median was NaN and the gate silently excluded nothing. Rows are now kept when any member is finite, and the median and MAE
use the finite cells.
"""

import numpy as np

from mlframe.models.ensembling.predict import _compute_outlier_gate


def _members(n: int = 20, disjoint_nan: bool = True):
    """Two good members near 0.5 and one outlier member 2.0 higher. With disjoint NaN, A is NaN on rows 0-5 and B on rows 14-19, so
    rows 6-13 keep all three members finite and carry the median."""
    rng = np.random.default_rng(0)
    good_a = 0.5 + 0.01 * rng.standard_normal(n)
    good_b = 0.5 + 0.01 * rng.standard_normal(n)
    outlier = 2.5 + 0.01 * rng.standard_normal(n)
    if disjoint_nan:
        good_a = good_a.copy()
        good_b = good_b.copy()
        good_a[:6] = np.nan
        good_b[14:] = np.nan
    preds = [good_a, good_b, outlier]
    return preds, np.stack(preds)


def test_outlier_is_excluded_with_disjoint_nan_rows():
    """The outlier member is skipped even though no single row is finite in every member (18-F6)."""
    preds, arr = _members(disjoint_nan=True)
    skipped = _compute_outlier_gate(preds, arr, 1.0, 0.0, 0.0, 0.0)[0]
    assert 2 in skipped


def test_good_members_are_not_excluded():
    """Control: the two good members are not excluded when the outlier is removed from the set (18-F6)."""
    preds, arr = _members(disjoint_nan=True)
    skipped = _compute_outlier_gate(preds, arr, 1.0, 0.0, 0.0, 0.0)[0]
    assert 0 not in skipped and 1 not in skipped
