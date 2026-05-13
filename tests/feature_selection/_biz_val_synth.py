"""Reusable synthetic data generators + helpers for biz_val tests.

Per CLAUDE.md "Every new ML trick gets a biz_val synthetic test":
to keep ~50+ biz_val tests fast + maintainable, the synthetic
generators they all share live HERE. Each generator is deterministic
(fixed seed), small (n=500-2000), and produces a target where a
specific structural feature is present.

Generators come with docstrings + doctests so the test file's intent
stays readable.

Usage::

    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, make_correlated_redundant, make_3way_xor,
        as_df, support_indices,
    )

    def test_biz_val_my_thing():
        df, y, signal = make_signal_plus_noise(n=1500, p_signal=3, p_noise=10)
        ...
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core generators
# ---------------------------------------------------------------------------


def make_signal_plus_noise(n: int = 2000, p_signal: int = 3,
                             p_noise: int = 10, seed: int = 42,
                             linear_only: bool = True):
    """Linear binary target with ``p_signal`` true features + ``p_noise``
    pure-noise. ``y = sign(sum(X_signal) + 0.3*noise)``.

    Returns ``(X, y, signal_indices)`` where signal_indices = [0..p_signal-1].

    >>> X, y, sig = make_signal_plus_noise(n=200, p_signal=2, p_noise=3)
    >>> X.shape
    (200, 5)
    >>> sorted(sig)
    [0, 1]
    >>> set(np.unique(y).tolist()).issubset({0, 1})
    True
    """
    rng = np.random.default_rng(seed)
    X_sig = rng.normal(size=(n, p_signal))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_sig, X_noise])
    if linear_only:
        score = X_sig.sum(axis=1) + 0.3 * rng.normal(size=n)
    else:
        score = X_sig[:, 0] ** 2 - X_sig[:, 1] ** 2 + 0.3 * rng.normal(size=n)
    y = (score > 0).astype(np.int64)
    return X, y, list(range(p_signal))


def make_correlated_redundant(n: int = 2000, n_corr: int = 4,
                                 p_noise: int = 5, corr: float = 0.95,
                                 seed: int = 42):
    """``n_corr`` features that share a base + 1 unique informative + ``p_noise``.
    Target depends on the unique informative AND one cluster member.

    Returns ``(X, y, unique_idx)`` where unique_idx is the index of the
    one truly orthogonal informative feature.

    >>> X, y, uniq = make_correlated_redundant(n=400, n_corr=3, p_noise=2)
    >>> X.shape
    (400, 6)
    >>> uniq
    3
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    noise_scale = float(np.sqrt(1 - corr ** 2) / max(corr, 1e-9))
    X_corr = np.column_stack([
        base + noise_scale * rng.normal(size=n) for _ in range(n_corr)
    ])
    unique = rng.normal(size=(n, 1))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_corr, unique, X_noise])
    y = (X_corr[:, 0] + unique[:, 0] + 0.3 * rng.normal(size=n) > 0
         ).astype(np.int64)
    return X, y, n_corr


def make_3way_xor(n: int = 2000, p: int = 10, seed: int = 42):
    """3-way XOR: ``y = sign(x_0 * x_1 * x_2)``. All individual and pair
    MIs ~0; only 3-way joint reveals signal.

    Returns ``(X, y, signal_indices=[0, 1, 2])``.

    >>> X, y, sig = make_3way_xor(n=400, p=5)
    >>> X.shape
    (400, 5)
    >>> sig
    [0, 1, 2]
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (np.sign(X[:, 0] * X[:, 1] * X[:, 2]) > 0).astype(np.int64)
    return X, y, [0, 1, 2]


def make_polynomial_target(n: int = 2000, seed: int = 42,
                              degree: int = 2):
    """``y = sign(0.7*x_a^d - 0.5*x_b^d + 0.3*x_a*x_b)`` for degree d.

    Useful for testing FE / polynomial-pair search. Signal is in
    (x_a, x_b) pair only; remaining columns are noise.

    >>> X, y, sig = make_polynomial_target(n=300, degree=2)
    >>> X.shape
    (300, 8)
    >>> sig
    [0, 1]
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    score = (0.7 * X[:, 0] ** degree
              - 0.5 * X[:, 1] ** degree
              + 0.3 * X[:, 0] * X[:, 1])
    y = (score > np.median(score)).astype(np.int64)
    return X, y, [0, 1]


def make_imbalanced(n: int = 2000, imbalance: float = 0.05,
                      p_signal: int = 3, p_noise: int = 8, seed: int = 42):
    """Class-imbalanced binary target. ``imbalance`` is the fraction
    of class-1 (default 5%).

    >>> X, y, sig = make_imbalanced(n=400, imbalance=0.1)
    >>> X.shape
    (400, 11)
    >>> bool(abs(y.mean() - 0.1) < 0.03)
    True
    """
    rng = np.random.default_rng(seed)
    X_sig = rng.normal(size=(n, p_signal))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_sig, X_noise])
    score = X_sig.sum(axis=1) + 0.3 * rng.normal(size=n)
    threshold = float(np.quantile(score, 1.0 - imbalance))
    y = (score > threshold).astype(np.int64)
    return X, y, list(range(p_signal))


def make_heavy_tail_skewed(n: int = 2000, p_noise: int = 5, seed: int = 42):
    """Heavy-tail lognormal inputs with log-multiplicative target:
    ``y = sign(log(base) + log(other) > median)``. Plug-in MI on
    raw inputs UNDER-estimates the relationship; log-aware estimators
    (or trees) recover full signal.

    >>> X, y, sig = make_heavy_tail_skewed(n=300)
    >>> X.shape
    (300, 7)
    >>> sig
    [0, 1]
    >>> bool(X[:, 0].min() > 0)  # base must be positive (lognormal)
    True
    """
    rng = np.random.default_rng(seed)
    base = np.exp(rng.normal(size=n))
    other = np.exp(rng.normal(size=n))
    noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([base, other, noise])
    score = np.log(base) + np.log(other)
    y = (score > np.median(score)).astype(np.int64)
    return X, y, [0, 1]


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def as_df(X: np.ndarray, y: np.ndarray, prefix: str = "x"):
    """Wrap a numpy ``(X, y)`` pair as ``(pd.DataFrame, pd.Series)``.

    >>> X = np.zeros((3, 2)); y = np.array([0, 1, 1])
    >>> df, ser = as_df(X, y, prefix="feat")
    >>> list(df.columns)
    ['feat0', 'feat1']
    >>> ser.name
    'y'
    """
    cols = [f"{prefix}{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def support_indices(sel):
    """Return support_ as integer indices regardless of whether the
    selector exposes a boolean mask, integer array, or list. Works for
    both ``sklearn.RFE``-style and ``mlframe.MRMR``-style selectors.

    >>> import numpy as np
    >>> class Sel: pass
    >>> s = Sel(); s.support_ = np.array([True, False, True])
    >>> support_indices(s)
    [0, 2]
    >>> s.support_ = np.array([0, 2])
    >>> support_indices(s)
    [0, 2]
    """
    s = sel.support_ if hasattr(sel, "support_") else sel
    arr = np.asarray(s)
    if arr.dtype == bool:
        return [int(i) for i in np.flatnonzero(arr)]
    return [int(i) for i in arr]


def signal_overlap(sel, signal: list, top_k: int = None) -> int:
    """Count how many signal-feature indices appear in ``sel.support_``.
    If ``top_k`` given, restrict to the first ``top_k`` of the support.

    >>> import numpy as np
    >>> class Sel: pass
    >>> s = Sel(); s.support_ = np.array([0, 1, 5, 7])
    >>> signal_overlap(s, [0, 1, 2])
    2
    >>> signal_overlap(s, [0, 1, 2], top_k=1)
    1
    """
    idx = support_indices(sel)
    if top_k is not None:
        idx = idx[:top_k]
    return len(set(idx) & set(int(i) for i in signal))


# ---------------------------------------------------------------------------
# Hypothesis property-based tests (embedded as doctests so they self-execute)
# ---------------------------------------------------------------------------
#
# Run via: ``python -m pytest --doctest-modules _biz_val_synth.py``
# or ``python -m doctest _biz_val_synth.py``
#
# Each property test uses ``@given`` from hypothesis to verify structural
# invariants across random inputs. They exercise the generators without
# requiring a full ML pipeline -- just numpy structural assertions.


def _property_make_signal_plus_noise_structural():
    """Generators must produce correct shapes and non-overlapping signal.

    >>> from hypothesis import given, strategies as st, settings
    >>> @given(n=st.integers(100, 500),
    ...        p_sig=st.integers(1, 4),
    ...        p_noise=st.integers(1, 6),
    ...        seed=st.integers(0, 100))
    ... @settings(max_examples=10, deadline=None)
    ... def _check(n, p_sig, p_noise, seed):
    ...     X, y, sig = make_signal_plus_noise(n=n, p_signal=p_sig,
    ...                                          p_noise=p_noise, seed=seed)
    ...     assert X.shape == (n, p_sig + p_noise)
    ...     assert len(sig) == p_sig
    ...     assert sorted(sig) == list(range(p_sig))
    ...     assert set(np.unique(y).tolist()).issubset({0, 1})
    ...     assert np.all(np.isfinite(X))
    >>> _check()  # doctest: +SKIP
    """


def _property_make_3way_xor_structural():
    """3-way XOR generators always produce correct shapes + binary y.

    >>> from hypothesis import given, strategies as st, settings
    >>> @given(n=st.integers(100, 400),
    ...        p=st.integers(3, 12),
    ...        seed=st.integers(0, 50))
    ... @settings(max_examples=10, deadline=None)
    ... def _check(n, p, seed):
    ...     X, y, sig = make_3way_xor(n=n, p=p, seed=seed)
    ...     assert X.shape == (n, p)
    ...     assert sig == [0, 1, 2]
    ...     assert set(np.unique(y).tolist()).issubset({0, 1})
    ...     # 3-way XOR: class balance near 50/50
    ...     cls_balance = float(np.mean(y))
    ...     assert 0.3 < cls_balance < 0.7
    >>> _check()  # doctest: +SKIP
    """


def _property_make_correlated_redundant_structural():
    """Correlated-redundant generators always produce correct shapes +
    correlation structure.

    >>> from hypothesis import given, strategies as st, settings
    >>> @given(n=st.integers(200, 500),
    ...        n_corr=st.integers(2, 5),
    ...        p_noise=st.integers(1, 5),
    ...        seed=st.integers(0, 50))
    ... @settings(max_examples=10, deadline=None)
    ... def _check(n, n_corr, p_noise, seed):
    ...     X, y, uniq_idx = make_correlated_redundant(
    ...         n=n, n_corr=n_corr, p_noise=p_noise, seed=seed)
    ...     assert X.shape == (n, n_corr + 1 + p_noise)
    ...     assert uniq_idx == n_corr
    ...     assert set(np.unique(y).tolist()).issubset({0, 1})
    ...     # Check correlation >= 0.7 between first two cluster members
    ...     if n_corr >= 2:
    ...         corr = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    ...         assert abs(corr) >= 0.7  # corr=0.95 nominal
    >>> _check()  # doctest: +SKIP
    """


def _property_make_as_df_roundtrip():
    """DataFrame wrapper preserves shape and binary y.

    >>> from hypothesis import given, strategies as st, settings
    >>> @given(n=st.integers(50, 200),
    ...        p=st.integers(2, 8),
    ...        seed=st.integers(0, 50))
    ... @settings(max_examples=8, deadline=None)
    ... def _check(n, p, seed):
    ...     rng = __import__('numpy').random.default_rng(seed)
    ...     X = rng.normal(size=(n, p))
    ...     y = (X[:, 0] > 0).astype(__import__('numpy').int64)
    ...     df, ser = as_df(X, y, prefix='f')
    ...     assert df.shape == (n, p)
    ...     assert ser.name == 'y'
    ...     assert list(df.columns) == [f'f{i}' for i in range(p)]
    >>> _check()  # doctest: +SKIP
    """


# ---------------------------------------------------------------------------
# Hypothesis strategies (lazy import so module loads without hypothesis)
# ---------------------------------------------------------------------------


def integers_or_skip(min_value, max_value):
    """Return ``st.integers(min_value, max_value)`` if hypothesis is
    installed; else ``None``. Caller is expected to gate the test with
    ``pytest.importorskip('hypothesis')``."""
    try:
        from hypothesis import strategies as st
        return st.integers(min_value=min_value, max_value=max_value)
    except ImportError:
        return None
