"""Wave 9.1 loop-iter-27 regression: ``get_feature_names_out`` must
distinguish synthesized placeholders from real DataFrame columns via a
sentinel attribute, not a brittle name-pattern heuristic.

Pre-fix at ``mrmr.py:1243`` (introduced by iter-12 fix):
  synthesized = all(str(n).startswith("feature_") for n in saved)

This heuristic misclassified legitimate DataFrame columns the user
happened to name ``feature_<n>`` (very common after
``pd.DataFrame(arr)`` + rename). The sklearn column-drift
``ValueError`` was silently skipped:

  sel.get_feature_names_out(['totally_wrong_name_A','B','C'])
  -> ['totally_wrong_name_A']   (BUG)
  -> ValueError                  (sklearn-canonical)

Fix:
1. ``_mrmr_fit_impl.py:202`` sets
   ``self._feature_names_in_synthesized_ = True`` when X was ndarray,
   ``False`` for DataFrame/polars input.
2. ``mrmr.py:1243`` reads the sentinel instead of guessing by name
   pattern. Falls back to a strict regex ``^feature_\\d+$`` (anchored,
   count parity required) when the sentinel is missing for back-compat
   with unpickled pre-fix estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _fit_df_with_feature_pattern_names():
    """User-supplied DataFrame whose columns happen to follow the
    ``feature_<n>`` naming pattern - the pre-fix misclassification
    surface.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        rng.standard_normal((n, 3)),
        columns=["feature_0", "feature_1", "feature_2"],
    )
    y = pd.Series((X["feature_0"] > 0).astype(np.int64), name="y")
    return MRMR(verbose=0).fit(X, y), X


def test_real_feature_names_not_misclassified_as_synthesized():
    """``_feature_names_in_synthesized_`` MUST be False for any
    DataFrame fit, regardless of column names.
    """
    sel, _ = _fit_df_with_feature_pattern_names()
    assert sel._feature_names_in_synthesized_ is False


def test_drift_input_features_raises_even_with_feature_pattern_names():
    """The sklearn column-drift contract MUST fire even when the user's
    legitimate columns follow the ``feature_<n>`` pattern. Pre-fix the
    startswith heuristic silently bypassed this check.
    """
    sel, _ = _fit_df_with_feature_pattern_names()
    with pytest.raises(ValueError, match="input_features"):
        sel.get_feature_names_out(["totally_wrong_name", "B", "C"])


def test_matching_input_features_succeeds_with_feature_pattern():
    """Negative control: matching the real names must succeed."""
    sel, X = _fit_df_with_feature_pattern_names()
    out = sel.get_feature_names_out(list(X.columns))
    assert len(out) >= 1


def test_ndarray_fit_marked_synthesized():
    """ndarray fit sets the sentinel to True so the override path fires."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(1)
    n = 200
    X = rng.standard_normal((n, 3))
    y = pd.Series((X[:, 0] > 0).astype(np.int64), name="y")
    sel = MRMR(verbose=0).fit(X, y)
    assert sel._feature_names_in_synthesized_ is True


def test_ndarray_fit_user_can_override_names():
    """ndarray-fit path lets the caller supply real names via
    ``input_features``.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(2)
    n = 200
    X = rng.standard_normal((n, 3))
    y = pd.Series((X[:, 0] > 0).astype(np.int64), name="y")
    sel = MRMR(verbose=0).fit(X, y)
    out = sel.get_feature_names_out(["a", "b", "c"])
    # At least one selected name must come from the user-supplied list.
    assert any(n in {"a", "b", "c"} for n in out)


def test_legacy_unpickled_estimator_fallback_to_regex():
    """Back-compat: estimators lacking the sentinel attribute fall back
    to a strict regex check. ``feature_x`` (string suffix) must NOT
    match -> drift contract fires.
    """
    sel, _ = _fit_df_with_feature_pattern_names()
    # Simulate an unpickled pre-iter-27 estimator by removing the
    # sentinel.
    del sel._feature_names_in_synthesized_
    # Names ``feature_0``, ``feature_1``, ``feature_2`` -> all match
    # the strict regex; fallback marks as synthesized; caller's drift
    # input is then ACCEPTED (back-compat with pre-iter-27 behaviour
    # at the cost of looser drift detection for these specific names).
    out = sel.get_feature_names_out(["a", "b", "c"])
    assert len(out) >= 1
