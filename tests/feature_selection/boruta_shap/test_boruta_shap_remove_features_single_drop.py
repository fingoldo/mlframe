"""Regression test for the single-call ``remove_features_if_rejected`` optimisation.

The per-feature ``DataFrame.drop(..., inplace=True)`` loop was replaced by ONE
``drop(list, errors="ignore")`` call (profiled 2.4x faster on the drop step, 5.2%
of a SHAP fit). This must stay BIT-IDENTICAL to the loop:
  - the surviving columns and their ORDER are unchanged, and
  - a feature already dropped in a prior trial (the old ``except KeyError: pass``
    branch) is silently skipped via ``errors="ignore"`` -- no raise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.boruta_shap import BorutaShap


def _loop_drop(df: pd.DataFrame, feats) -> pd.DataFrame:
    """The pre-optimisation reference behaviour."""
    out = df.copy()
    for f in feats:
        try:
            out.drop(f, axis=1, inplace=True)
        except KeyError:
            pass
    return out


def test_single_drop_matches_loop_columns_and_order():
    """Single drop matches loop columns and order."""
    cols = [f"f{i}" for i in range(12)]
    base = pd.DataFrame(np.arange(5 * 12).reshape(5, 12), columns=cols)

    # Mixed batch: out-of-order names + a non-contiguous subset.
    to_remove = ["f7", "f1", "f10", "f3"]

    ref = _loop_drop(base, to_remove)

    bs = BorutaShap.__new__(BorutaShap)  # no __init__ needed; method only touches self.X / self.features_to_remove
    bs.X = base.copy()
    bs.features_to_remove = to_remove
    bs.remove_features_if_rejected()

    assert list(bs.X.columns) == list(ref.columns), f"surviving column ORDER diverged: opt={list(bs.X.columns)} ref={list(ref.columns)}"
    assert bs.X.equals(ref), "surviving values diverged from the loop reference"


def test_single_drop_ignores_already_removed():
    """A name in features_to_remove that is no longer present must NOT raise (prior except-KeyError parity)."""
    base = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    bs = BorutaShap.__new__(BorutaShap)
    bs.X = base.copy()
    bs.features_to_remove = ["b", "zzz_not_here", "a"]  # zzz_not_here was 'dropped in a prior loop'
    bs.remove_features_if_rejected()  # must not raise
    assert list(bs.X.columns) == ["c"]


def test_single_drop_empty_list_is_noop():
    """Single drop empty list is noop."""
    base = pd.DataFrame({"a": [1], "b": [2]})
    bs = BorutaShap.__new__(BorutaShap)
    bs.X = base.copy()
    bs.features_to_remove = []
    bs.remove_features_if_rejected()
    assert list(bs.X.columns) == ["a", "b"]
