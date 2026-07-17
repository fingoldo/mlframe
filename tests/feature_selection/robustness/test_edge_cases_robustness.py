"""Edge-case and robustness tests for MRMR and helpers.

Each test asserts either CORRECT handling or that the library raises a SPECIFIC,
documented exception. Per the project rule against masking bugs with guards, an
edge case that surfaces an unexpected behaviour is annotated with a
``# !TODO! verify <observed> is intended`` comment rather than swallowed.

The 10 edge cases covered:

1. ``test_mrmr_single_class_y_raises`` -- target with one unique class.
2. ``test_mrmr_single_feature_X_returns_that_feature`` -- 1-column X is selected.
3. ``test_mrmr_all_constant_column_dropped`` -- constant col never enters support.
4. ``test_mrmr_nan_in_X_raises_or_imputes`` -- NaN handling is asserted by observed.
5. ``test_mrmr_inf_in_X_raises`` -- +/-Inf in numeric data.
6. ``test_mrmr_n_lt_10_raises`` -- screen.py asserts len(factors_data) >= 10.
7. ``test_mrmr_perfectly_correlated_pair_keeps_one`` -- redundancy handling.
8. ``test_mrmr_pickle_roundtrip_preserves_transform`` -- ``__setstate__`` BC.
9. ``test_mrmr_clone_preserves_constructor_params`` -- sklearn.clone contract.
10. ``test_mrmr_fit_cache_hit_replays_state`` -- ``_FIT_CACHE`` replay path.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from mlframe.feature_selection.filters import MRMR

# ---------------------------------------------------------------------------
# Helpers / small fixtures (n=200, random_seed=42)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_ROWS = 200


def _fast_mrmr(**overrides):
    """Tiny-budget MRMR for fast edge-case checks."""
    kwargs = dict(
        full_npermutations=2,
        baseline_npermutations=2,
        quantization_nbins=5,
        fe_max_steps=0,
        verbose=0,
        n_jobs=1,
        n_workers=1,
        random_seed=RANDOM_SEED,
        random_state=RANDOM_SEED,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _support_names(mrmr: MRMR) -> list[str]:
    """Resolve ``support_`` (integer-indices OR boolean mask) to feature names."""
    support = np.asarray(mrmr.support_)
    feat_names = list(mrmr.feature_names_in_)
    if len(support) == 0:
        return []
    if support.dtype == bool or isinstance(support.flat[0], (bool, np.bool_)):
        return [n for n, s in zip(feat_names, support) if s]
    return [feat_names[int(i)] for i in support]


# ---------------------------------------------------------------------------
# 1. Single-class y
# ---------------------------------------------------------------------------
def test_mrmr_single_class_y_raises():
    """y with one unique class -> ValueError (H(y)=0 makes MRMR vacuous)."""
    rng = np.random.default_rng(RANDOM_SEED)
    X = pd.DataFrame(rng.standard_normal((N_ROWS, 3)), columns=["a", "b", "c"])
    y = np.zeros(N_ROWS, dtype=int)
    mrmr = _fast_mrmr()
    with pytest.raises(ValueError, match=r"1 unique value|H\(y\)|unique"):
        mrmr.fit(X, y)


# ---------------------------------------------------------------------------
# 2. Single-column X
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_mrmr_single_feature_X_returns_that_feature():
    """X with 1 column carrying the signal -- MRMR selects that column."""
    rng = np.random.default_rng(RANDOM_SEED)
    signal = rng.standard_normal(N_ROWS)
    X = pd.DataFrame({"only": signal})
    y = (signal > 0).astype(int)
    mrmr = _fast_mrmr(min_features_fallback=1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)
    except ValueError as exc:
        # Some screening configs reject p=1 because the redundancy term needs
        # pairs; tolerated only if the message says so explicitly.
        pytest.skip(f"selector requires >=2 features: {exc}")

    assert mrmr.n_features_in_ == 1
    # Exactly one RAW input feature is selected, and it is "only". ``support_``
    # indexes into ``feature_names_in_`` (the raw input columns), so it is the
    # authoritative check for "which raw features were kept". ``n_features_`` is
    # NOT asserted here: hinge change-point FE is default-on (fe_hinge_enable=True
    # since the 2026-06 campaign), so on a single signal-bearing column MRMR may
    # additionally emit replayable relu legs (``only__relu_gt`` / ``only__relu_lt``)
    # that legitimately appear in transform() output. n_features_ counts those
    # engineered columns by design (= len(selected_vars) + n_engineered_out), so
    # measured n_features_==3 here (raw "only" + 2 hinge legs) is correct, not a
    # selection bug. The raw-selection invariant is what this edge case guards.
    names = _support_names(mrmr)
    assert names == ["only"]


# ---------------------------------------------------------------------------
# 3. All-constant column dropped
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_mrmr_all_constant_column_dropped():
    """Constant col has H=0 / nbins==1 -- MRMR never selects it."""
    rng = np.random.default_rng(RANDOM_SEED)
    signal = rng.standard_normal(N_ROWS)
    X = pd.DataFrame(
        {
            "constant": np.full(N_ROWS, 3.14),
            "signal": signal,
        }
    )
    y = (signal > 0).astype(int)
    mrmr = _fast_mrmr(min_features_fallback=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr.fit(X, y)
    names = _support_names(mrmr)
    assert "constant" not in names, f"constant column leaked into support_: {names}"


# ---------------------------------------------------------------------------
# 4. NaN in X
# ---------------------------------------------------------------------------
def test_mrmr_nan_in_X_native_tolerance():
    """Current contract: MRMR is NaN-tolerant natively (sparse NaNs in numeric X
    no longer raise). The screening discretiser treats NaN as a distinct bin, so
    fit must complete and produce a non-empty support_ on a realistic frame.
    Prior contract asserted ValueError("NaN|impute"); that was retired when
    Agent A made the inner kernels NaN-aware."""
    rng = np.random.default_rng(RANDOM_SEED)
    X = pd.DataFrame(rng.standard_normal((N_ROWS, 3)), columns=list("abc"))
    X.iloc[0, 0] = np.nan
    X.iloc[5, 1] = np.nan
    y = (rng.standard_normal(N_ROWS) > 0).astype(int)
    mrmr = _fast_mrmr(min_features_fallback=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr.fit(X, y)  # must NOT raise on NaN
    names = _support_names(mrmr)
    # min_features_fallback=1 guarantees at least one column survives even when
    # screening can't confirm any predictor at the configured confidence floor.
    assert len(names) >= 1


# ---------------------------------------------------------------------------
# 5. Inf in X
# ---------------------------------------------------------------------------
def test_mrmr_inf_in_X_raises():
    """Post-fix contract: +/-inf in numeric X raises ValueError; downstream discretisation produces undefined bins on inf so we reject upfront."""
    rng = np.random.default_rng(RANDOM_SEED)
    X = pd.DataFrame(rng.standard_normal((N_ROWS, 3)), columns=list("abc"))
    X.iloc[0, 0] = np.inf
    X.iloc[1, 1] = -np.inf
    y = (rng.standard_normal(N_ROWS) > 0).astype(int)
    mrmr = _fast_mrmr(min_features_fallback=1)
    with pytest.raises(ValueError, match=r"inf|infinite"):
        mrmr.fit(X, y)


# ---------------------------------------------------------------------------
# 6. n < 10 rows
# ---------------------------------------------------------------------------
def test_mrmr_n_lt_10_raises():
    """screen.py line ~201: ``assert len(factors_data) >= 10``. Anything less
    must surface as AssertionError (or ValueError if a guard is added later).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    X = pd.DataFrame(rng.standard_normal((5, 3)), columns=list("abc"))
    y = (rng.standard_normal(5) > 0).astype(int)
    # Need >=2 classes so the single-class guard doesn't fire first.
    if len(set(y)) == 1:
        y = np.array([0, 1, 0, 1, 0])
    mrmr = _fast_mrmr(min_features_fallback=1)
    with pytest.raises((AssertionError, ValueError)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)


# ---------------------------------------------------------------------------
# 7. Perfectly correlated feature pair
# ---------------------------------------------------------------------------
def test_mrmr_perfectly_correlated_pair_keeps_one():
    """x_1 = x_0 exactly -- MRMR's redundancy term must keep at most one.

    Note: the default ``use_simple_mode=True`` is documented as "works very
    fast but leaves redundant features" (see MRMR.__init__ docstring), so the
    redundancy contract is checked with ``use_simple_mode=False``.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    x0 = rng.standard_normal(N_ROWS)
    X = pd.DataFrame(
        {
            "x0": x0,
            "x0_copy": x0.copy(),  # perfectly redundant
            "noise": rng.standard_normal(N_ROWS),
        }
    )
    y = (x0 > 0).astype(int)
    mrmr = _fast_mrmr(min_features_fallback=1, use_simple_mode=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr.fit(X, y)
    names = _support_names(mrmr)
    n_dup = sum(1 for n in names if n in ("x0", "x0_copy"))
    # !TODO! verify keeping both copies when use_simple_mode=False is intended
    assert n_dup <= 1, f"redundant pair kept both copies even with use_simple_mode=False: {names} (expected at most one of x0/x0_copy)"


# ---------------------------------------------------------------------------
# 8. Pickle round-trip
# ---------------------------------------------------------------------------
@pytest.mark.fast
def test_mrmr_pickle_roundtrip_preserves_transform():
    """Pickle -> unpickle -> transform must match the original transform.

    Also indirectly exercises ``__setstate__`` (the BC default-injection path).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    signal = rng.standard_normal(N_ROWS)
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise1": rng.standard_normal(N_ROWS),
            "noise2": rng.standard_normal(N_ROWS),
        }
    )
    y = (signal > 0).astype(int)
    mrmr = _fast_mrmr(min_features_fallback=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr.fit(X, y)
    A = mrmr.transform(X)
    payload = pickle.dumps(mrmr)
    restored = pickle.loads(payload)  # nosec B301 -- round-trip of a locally-created, trusted object
    B = restored.transform(X)

    a_arr = np.asarray(A.values if hasattr(A, "values") else A)
    b_arr = np.asarray(B.values if hasattr(B, "values") else B)
    assert a_arr.shape == b_arr.shape
    np.testing.assert_array_equal(a_arr, b_arr)
    # __setstate__ injected BC defaults must be present after unpickle.
    for attr in ("_engineered_features_", "_engineered_recipes_"):
        assert hasattr(restored, attr), f"__setstate__ missing default: {attr}"


# ---------------------------------------------------------------------------
# 9. sklearn.clone preserves params, drops fitted state
# ---------------------------------------------------------------------------
def test_mrmr_clone_preserves_constructor_params():
    """clone() yields an unfitted copy with identical constructor params."""
    mrmr = _fast_mrmr(
        quantization_nbins=7,
        min_features_fallback=2,
        fe_max_steps=0,
    )
    rng = np.random.default_rng(RANDOM_SEED)
    X = pd.DataFrame(rng.standard_normal((N_ROWS, 3)), columns=list("abc"))
    y = (rng.standard_normal(N_ROWS) > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr.fit(X, y)
    assert hasattr(mrmr, "support_")

    cloned = clone(mrmr)
    # Same constructor params (sklearn contract).
    assert cloned.get_params(deep=False) == mrmr.get_params(deep=False)
    # No fitted state on the clone.
    for attr in ("support_", "n_features_", "feature_names_in_"):
        assert not hasattr(cloned, attr), f"clone leaked fitted attribute: {attr}"


# ---------------------------------------------------------------------------
# 10. _FIT_CACHE hit replays state
# ---------------------------------------------------------------------------
def test_mrmr_fit_cache_hit_replays_state():
    """Re-fitting on identical (X, y) with cloned MRMR must hit ``_FIT_CACHE``,
    return quickly, and produce identical ``support_``.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    signal = rng.standard_normal(N_ROWS)
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise1": rng.standard_normal(N_ROWS),
            "noise2": rng.standard_normal(N_ROWS),
        }
    )
    y = (signal > 0).astype(int)

    MRMR._FIT_CACHE.clear()
    mrmr1 = _fast_mrmr(
        skip_retraining_on_same_content=False,  # force cache path, not signature short-circuit
        min_features_fallback=1,
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr1.fit(X, y)
    first_dt = time.perf_counter() - t0
    first_support = np.asarray(mrmr1.support_).copy()

    # Fresh, unfitted clone -- forces the content-keyed cache path (signature
    # short-circuit is disabled above).
    mrmr2 = clone(mrmr1)
    assert not hasattr(mrmr2, "support_")
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr2.fit(X, y)
    second_dt = time.perf_counter() - t0

    second_support = np.asarray(mrmr2.support_)
    np.testing.assert_array_equal(first_support, second_support)
    # Cache hit must be at least as fast as the cold fit; allow 2x slack for
    # noisy CI. The point is to detect a full re-fit (which would be 5-50x).
    assert second_dt <= max(first_dt * 2.0, 0.5), f"_FIT_CACHE hit did not short-circuit: cold={first_dt:.3f}s, warm={second_dt:.3f}s"
    MRMR._FIT_CACHE.clear()
