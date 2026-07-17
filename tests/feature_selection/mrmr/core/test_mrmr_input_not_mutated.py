"""Regression: ``MRMR.fit`` MUST NOT mutate the caller's input DataFrame.

P1 data-side-effect bug (2026-06-11). ``_fit_impl`` injects temporary
``targ_*`` columns into the working pandas frame and the FE pipeline appends
engineered columns IN PLACE (``X[name] = ...``, hinge / cat-FE generators).
The ``targ_*`` injection was reversed in the ``fit`` ``finally`` block, but the
engineered FE columns were NEVER removed -- so a caller-supplied frame came
back with::

    X.columns: ['a', 'b'] -> ['a', 'b', 'a__relu_gt...', 'a__relu_lt...', ...]

permanently appended. This violates the sklearn fit-must-not-mutate-input
contract. Worse, it silently corrupts a frame REUSED across fits: the 2nd
fit's FE builds on the 1st fit's leaked columns -> a different post-FE X
content -> the ``_FIT_CACHE`` / replay signature no longer matches, so the
selection differs from a fresh-copy fit and the cache-replay path is skipped
(this is exactly what made ``test_replay_fitted_state_isolation.py`` fail on
pristine master: A's leaked FE columns changed B's content hash, so B never
hit the replay path and its ``support_`` was neither shared with nor frozen
against A's).

Fix is at the boundary (``MRMR.fit``): the wrapper copies a pandas input frame once
(ALWAYS shallow -- ``X.copy(deep=False)``, unconditionally, regardless of pandas'
Copy-on-Write setting; see perf audit finding #2, 2026-07-17) so all downstream
appends land on the internal copy and the caller's frame is never touched. A shallow
copy is safe on every pandas version because every internal mutation site only ever
ADDS a new column key (never overwrites an EXISTING column's values in place), so no
write can ever land on the original frame's shared arrays -- the previous CoW-gated
``deep=True`` fallback was defensive-but-unnecessary and cost a real O(n*p)
alloc+memcpy on every fit on any pandas installation with CoW off (the DEFAULT for
most installed pandas < 3.0, i.e. the common case, not a rare edge case).

These tests FAIL on pre-fix code (the caller's columns gain engineered names
and the second-fit support diverges from a fresh-copy fit) and PASS after.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _make_xy(n: int = 200, seed: int = 0):
    """A small (X, y) for which the DEFAULT FE pipeline actually engineers +
    appends columns -- the side-effect under test only fires when FE produces
    something. The ``y = (a > 0)`` threshold target reliably triggers the
    hinge generator, which on pre-fix code leaks ``a__relu_gt.../a__relu_lt...``
    columns into the caller's frame (this is the exact data shape that makes
    ``test_replay_fitted_state_isolation.py`` fail on pristine master).
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    y = pd.Series((a > 0).astype(np.int64), name="y")
    X = pd.DataFrame({"a": a, "b": b})
    return X, y


def _fit(X, y):
    """Fit a default-verbosity MRMR on (X, y), silencing its accuracy-suboptimal-param warnings."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MRMR(verbose=0).fit(X, y)


def test_fit_does_not_append_columns_to_caller_frame():
    """The single most direct assertion: after one fit, the caller's frame
    has the SAME columns and shape it had before -- no engineered ``__``
    columns and no leftover ``targ_*`` columns leaked in.
    """
    X, y = _make_xy()
    cols_before = list(X.columns)
    shape_before = X.shape
    _fit(X, y)
    assert list(X.columns) == cols_before, (
        f"MRMR.fit appended columns to the caller's DataFrame (before={cols_before}, after={list(X.columns)}); fit must not mutate its input."
    )
    assert X.shape == shape_before, f"MRMR.fit changed the caller frame shape {shape_before} -> {X.shape}."
    # No engineered / target columns leaked under any naming convention.
    leaked = [c for c in X.columns if ("__" in str(c)) or str(c).startswith("targ")]
    assert not leaked, f"Engineered / target columns leaked into the caller frame: {leaked}"


def test_fit_preserves_original_column_values_and_identity():
    """The raw columns the caller passed must be byte-identical after fit
    (no in-place quantisation / NaN-fill / dtype coercion on the caller's
    own arrays).
    """
    X, y = _make_xy()
    snapshot = {c: X[c].to_numpy(copy=True) for c in X.columns}
    dtypes_before = X.dtypes.to_dict()
    _fit(X, y)
    for c, vals in snapshot.items():
        assert np.array_equal(X[c].to_numpy(), vals, equal_nan=True), f"MRMR.fit mutated the values of caller column {c!r} in place."
    assert X.dtypes.to_dict() == dtypes_before, "MRMR.fit changed caller column dtypes."


def test_second_fit_on_same_frame_is_independent_of_first():
    """Fitting twice on the SAME frame object must yield the same selection
    as fitting on a FRESH copy -- i.e. the first fit left no residue that
    perturbs the second. Pre-fix the first fit's leaked FE columns changed
    the frame, so the second fit screened a different (augmented) pool.
    """
    # seed=0 reliably triggers the hinge generator AND a 2nd-step nested-FE
    # divergence on a polluted frame (pre-fix: the 2nd fit engineers a
    # ``a__relu_gt..__haar_j1k0`` term ON TOP of the 1st fit's leaked
    # ``a__relu_gt..`` column -- a name a fresh-copy fit never produces).
    X_shared, y = _make_xy(seed=0)
    X_fresh = X_shared.copy(deep=True)  # pristine reference

    m1 = _fit(X_shared, y)
    support_first = np.asarray(m1.support_, dtype=np.int64).copy()
    names_first = list(m1.get_feature_names_out())

    # Second fit on the SAME (possibly-residue-bearing) frame.
    m2 = _fit(X_shared, y)
    support_second = np.asarray(m2.support_, dtype=np.int64)
    names_second = list(m2.get_feature_names_out())

    # Reference: fit on a frame guaranteed never touched by a prior fit.
    m_fresh = _fit(X_fresh, y)
    support_fresh = np.asarray(m_fresh.support_, dtype=np.int64)
    names_fresh = list(m_fresh.get_feature_names_out())

    assert np.array_equal(support_second, support_fresh), (
        "Second fit on a reused frame produced a DIFFERENT support_ than a "
        f"fresh-copy fit (reused={support_second.tolist()}, "
        f"fresh={support_fresh.tolist()}); the first fit leaked state into the frame."
    )
    assert names_second == names_fresh, f"Second fit selected different feature names than a fresh-copy fit (reused={names_second}, fresh={names_fresh})."
    # Sanity: the first fit was itself a clean fit on a pristine frame, so its
    # selection should also match the fresh reference (guards against the test
    # silently passing because BOTH reused fits are equally corrupted).
    assert np.array_equal(support_first, support_fresh), (
        f"First fit support {support_first.tolist()} already differs from the fresh-copy reference {support_fresh.tolist()}."
    )
    assert names_first == names_fresh


def test_polars_input_not_mutated_when_available():
    """Polars frames are immutable, so the caller's frame is structurally
    safe -- assert MRMR.fit leaves a polars input unchanged (regression
    sentry: a future boundary-copy refactor must not break the polars path).
    """
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(2)
    n = 200
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    y = pd.Series((a > 0).astype(np.int64), name="y")
    Xpl = pl.DataFrame({"a": a, "b": b})
    cols_before = list(Xpl.columns)
    _fit(Xpl, y)
    assert list(Xpl.columns) == cols_before, f"MRMR.fit mutated the polars input columns {cols_before} -> {list(Xpl.columns)}."


def test_fit_does_not_mutate_input_with_copy_on_write_forced_off():
    """Explicit regression pin for perf audit finding #2 (2026-07-17): MRMR.fit's internal isolation
    copy is now ALWAYS shallow (``X.copy(deep=False)``), regardless of pandas' Copy-on-Write setting.
    Force CoW off here (pandas' own default for most installed 2.x versions, so this is the common
    case, not a hypothetical one) so this test keeps covering the exact scenario a future pandas
    default-CoW flip could otherwise silently stop exercising, and assert the caller's frame -- columns,
    values, AND dtypes -- is completely unchanged after a real FE-heavy fit."""
    cow_before = pd.get_option("mode.copy_on_write")
    pd.set_option("mode.copy_on_write", False)
    try:
        X, y = _make_xy()
        cols_before = list(X.columns)
        dtypes_before = X.dtypes.to_dict()
        snapshot = {c: X[c].to_numpy(copy=True) for c in X.columns}
        _fit(X, y)
        assert list(X.columns) == cols_before, f"MRMR.fit mutated the caller's columns under CoW-off {cols_before} -> {list(X.columns)}."
        assert X.dtypes.to_dict() == dtypes_before, "MRMR.fit changed caller column dtypes under CoW-off."
        for c, vals in snapshot.items():
            assert np.array_equal(X[c].to_numpy(), vals, equal_nan=True), f"MRMR.fit mutated caller column {c!r} in place under CoW-off."
    finally:
        pd.set_option("mode.copy_on_write", cow_before)
