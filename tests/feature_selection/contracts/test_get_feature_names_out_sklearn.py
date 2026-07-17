"""Wave 9.1 loop-iter-12 regression: ``MRMR.get_feature_names_out``
must honour the sklearn protocol for ``input_features``.

Pre-fix: the ``input_features`` argument was accepted in the signature
but silently ignored on every code path, so:
1. Pipeline column-drift detection was BYPASSED -- a caller passing
   mismatched names got fit-time names back with no error, hiding
   schema drift.
2. After fitting on a raw ndarray (no column names), MRMR synthesizes
   ``feature_N`` placeholders. The caller's user-supplied names were
   silently dropped, so downstream ``Pipeline.get_feature_names_out()``
   produced wrong labels.

Fix at ``mrmr.py:1206``:
- ``input_features is None`` -> use saved ``feature_names_in_``.
- ``input_features`` provided AND fit-time saw real names -> validate
  equality, raise ``ValueError`` on mismatch (sklearn contract).
- ``input_features`` provided AND fit-time was ndarray (synthesized
  ``feature_N``) -> caller's names take precedence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _fit_df(n: int = 100):
    """Fit df."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series(rng.integers(0, 2, n), name="y")
    from mlframe.feature_selection.filters.mrmr import MRMR

    return MRMR(verbose=0).fit(X, y), X, y


def test_get_feature_names_out_none_input():
    """``input_features=None`` uses fit-time names."""
    sel, _, _ = _fit_df()
    out = sel.get_feature_names_out()
    assert len(out) >= 1
    # Each name must be one of the fit-time names (or an engineered name). Engineered features carry an
    # operator signature: a composite-recipe call form ``op(...)``, an operator-suffixed ``base__op`` name,
    # or one of the structural-FE prefixes (dcd/eng aggregate, row-argmax, binned-agg, conditional-gate,
    # integer-lattice). Raw passthroughs match ``fit_names`` exactly.
    fit_names = set(sel.feature_names_in_)
    for n in out:
        assert (
            n in fit_names
            or "(" in n  # composite recipe form, e.g. min(log(c),sin(d)) / mul(...)
            or "__" in n  # operator-suffixed, e.g. il_gcd__a__b / c__relu_lt-0.5
            or "/" in n
            or "_x_" in n
            or n.startswith(("_dcd_", "_eng_", "argmax_", "binagg_", "gate_", "il_"))
        ), f"unrecognized output feature name: {n!r}"


def test_get_feature_names_out_matching_input_features():
    """Matching ``input_features`` succeeds and returns the same as None."""
    sel, X, _ = _fit_df()
    out_none = list(sel.get_feature_names_out())
    out_match = list(sel.get_feature_names_out(list(X.columns)))
    assert out_none == out_match


def test_get_feature_names_out_drift_raises():
    """sklearn column-drift contract: mismatched ``input_features`` must raise."""
    sel, _, _ = _fit_df()
    with pytest.raises(ValueError, match="input_features"):
        sel.get_feature_names_out(["xx", "yy", "zz", "ww"])


def test_get_feature_names_out_wrong_length_raises():
    """Wrong-length ``input_features`` must also raise (drift contract)."""
    sel, _, _ = _fit_df()
    with pytest.raises(ValueError):
        sel.get_feature_names_out(["a", "b"])  # too few


def test_get_feature_names_out_ndarray_fit_honours_user_names():
    """After ndarray fit, ``input_features`` overrides the synthesized
    ``feature_N`` placeholders so Pipeline can carry custom names.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(1)
    n = 100
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    sel = MRMR(verbose=0).fit(X, pd.Series(y, name="y"))
    # Default: synthesized placeholders.
    out_none = list(sel.get_feature_names_out())
    assert all(n.startswith("feature_") for n in out_none if n in {f"feature_{i}" for i in range(4)})
    # With caller-supplied names: honoured (since fit-time names were synthesized).
    out_user = list(sel.get_feature_names_out(["a", "b", "c", "d"]))
    # The base selected feature must come from the user-supplied list,
    # NOT from "feature_N".
    assert any(n in {"a", "b", "c", "d"} for n in out_user), out_user


def test_get_feature_names_out_ndarray_fit_wrong_length_raises():
    """Even on the ndarray-fit path, length mismatch is still an error."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(2)
    n = 100
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    sel = MRMR(verbose=0).fit(X, pd.Series(y, name="y"))
    with pytest.raises(ValueError):
        sel.get_feature_names_out(["a", "b"])  # too few
