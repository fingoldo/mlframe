"""Regression: hybrid_row_argmax_fe_with_recipes / hybrid_conditional_gate_fe_with_recipes must not
re-run _is_argmax_eligible on a column list they just built via that exact predicate.

Both wrappers build `elig` via a list comprehension calling `_is_argmax_eligible` once per column,
then hand `elig` straight to the underlying `cheap_*_scan` function. Pre-fix, that function's own
`cols is not None` branch re-validated every column with the SAME predicate -- a pure duplicate O(n)
scan since every element of `elig` is eligible by construction. The `_cols_prefiltered=True` argument
threaded from the wrappers skips that redundant re-check without touching the public contract for any
other caller (structure_discovery.py, tests, ad-hoc scripts) that still passes an unfiltered `cols`.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._conditional_gate_fe import (
    _is_argmax_eligible,
    hybrid_conditional_gate_fe_with_recipes,
    hybrid_row_argmax_fe_with_recipes,
)


def _build_frame(seed: int, n: int = 600, p: int = 6):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(p)})
    y = (X["x0"].to_numpy() > X["x1"].to_numpy()).astype(int)
    return X, y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_row_argmax_eligibility_scanned_once_per_column(seed):
    X, y = _build_frame(seed)
    calls = {"n": 0}
    orig = _is_argmax_eligible

    def spy(x):
        calls["n"] += 1
        return orig(x)

    with patch("mlframe.feature_selection.filters._conditional_gate_fe._is_argmax_eligible", side_effect=spy):
        hybrid_row_argmax_fe_with_recipes(X, y, seed=seed)

    assert calls["n"] == X.shape[1], f"expected exactly {X.shape[1]} eligibility checks (one per column, no re-scan), got {calls['n']}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_conditional_gate_eligibility_scanned_once_per_column(seed):
    X, y = _build_frame(seed)
    calls = {"n": 0}
    orig = _is_argmax_eligible

    def spy(x):
        calls["n"] += 1
        return orig(x)

    with patch("mlframe.feature_selection.filters._conditional_gate_fe._is_argmax_eligible", side_effect=spy):
        hybrid_conditional_gate_fe_with_recipes(X, y, seed=seed)

    assert calls["n"] == X.shape[1], f"expected exactly {X.shape[1]} eligibility checks (one per column, no re-scan), got {calls['n']}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_row_argmax_output_unchanged_by_prefiltered_skip(seed):
    X, y = _build_frame(seed)
    appended, recipes = hybrid_row_argmax_fe_with_recipes(X, y, seed=seed)
    # A second, independent call (fresh eligibility scan each time, no shared cache) must reproduce
    # identically -- proves the skip doesn't silently admit an ineligible column.
    appended2, recipes2 = hybrid_row_argmax_fe_with_recipes(X, y, seed=seed)
    assert appended == appended2
    assert [r.name for r in recipes] == [r.name for r in recipes2]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_conditional_gate_output_unchanged_by_prefiltered_skip(seed):
    X, y = _build_frame(seed)
    appended, recipes = hybrid_conditional_gate_fe_with_recipes(X, y, seed=seed)
    appended2, recipes2 = hybrid_conditional_gate_fe_with_recipes(X, y, seed=seed)
    assert appended == appended2
    assert [r.name for r in recipes] == [r.name for r in recipes2]


def test_cheap_scans_still_validate_eligibility_for_external_unfiltered_callers():
    """The public contract for every OTHER caller (structure_discovery.py, ad-hoc cols=[...]) must be
    unchanged: passing a raw, unfiltered column list without _cols_prefiltered still re-validates."""
    from mlframe.feature_selection.filters._conditional_gate_fe import (
        cheap_conditional_gate_scan,
        cheap_row_argmax_scan,
    )

    X, y = _build_frame(0, p=5)
    # Inject one column that is NOT argmax-eligible (contains a NaN -> fails the finite-numeric
    # predicate) into a raw, caller-supplied cols list; the default (_cols_prefiltered=False) path
    # must still filter it out -- unlike a _cols_prefiltered=True caller, which would let it through
    # unchecked (correct, since ONLY hybrid_row_argmax_fe_with_recipes/hybrid_conditional_gate_fe_
    # with_recipes set that flag, and only after building `elig` via this exact predicate themselves).
    X = X.copy()
    X["has_nan"] = X["x0"].to_numpy()
    X.loc[0, "has_nan"] = np.nan
    assert not _is_argmax_eligible(np.asarray(X["has_nan"]))
    hits = cheap_row_argmax_scan(X, y, list(X.columns), max_triples=10)
    used_cols = {c for h in hits for c in h.cols}
    assert "has_nan" not in used_cols

    gate_hits = cheap_conditional_gate_scan(X, y, list(X.columns))
    gate_cols = {c for h in gate_hits for c in h.cols}
    assert "has_nan" not in gate_cols
