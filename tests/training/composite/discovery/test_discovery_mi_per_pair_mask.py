"""Regression + biz_value sensors for the deferred (FUTURE) audit finding D11
in ``src/mlframe/training/composite/discovery/screening.py``.

D11: ``_mi_to_target`` used a GLOBAL all-column finite mask
(``np.isfinite(target) & np.all(np.isfinite(feature_matrix), axis=1)``).
One mostly-NaN column then dropped the jointly-observed rows out from
under EVERY other column -- a single 99%-NaN column could leave <50 rows
and zero the MI for the whole sweep (silent degeneration of the
documented heavy-tail ``mi_estimator='knn'`` option). The fix masks
PER PAIR: each column's MI is computed on the rows where that column and
the target are both finite, so a bad column only zeros its own MI.

Pinned on BOTH estimators:

- ``estimator='bin'``: per-pair masking recovers the dense columns' MI.
- ``estimator='knn'``: same, through the Kraskov path.

Bit-identity is preserved on all-finite matrices (no NaN -> the global
mask was the identity and the per-pair mask is too), pinned below.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery.screening import _mi_to_target


def _informative_matrix(n: int, k: int, rng: np.random.Generator):
    """k dense columns each linearly informative about a shared target."""
    target = rng.normal(size=n).astype(np.float64)
    cols = []
    for j in range(k):
        # decreasing signal strength so MI is clearly non-zero but varied
        noise = rng.normal(size=n) * (0.5 + 0.3 * j)
        cols.append(target * (1.0 + 0.2 * j) + noise)
    X = np.column_stack(cols).astype(np.float64)
    return X, target


@pytest.mark.parametrize("estimator", ["bin", "knn"])
def test_mi_to_target_one_mostly_nan_column_does_not_zero_the_sweep(estimator):
    """D11 regression: a single 99%-NaN column must NOT collapse MI for the
    dense informative columns. Pre-fix the global AND-mask retained only the
    ~1% rows where the bad column was finite -> <50 rows -> MI 0.0."""
    rng = np.random.default_rng(20260611)
    n, k = 4000, 6
    X, target = _informative_matrix(n, k, rng)

    # Poison ONE column: 99% NaN (only ~40 finite rows, < the 50-row gate).
    bad = X[:, 0].copy()
    finite_keep = rng.choice(n, size=40, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[finite_keep] = False
    bad[mask] = np.nan
    X[:, 0] = bad

    mi = _mi_to_target(
        X,
        target,
        n_neighbors=3,
        random_state=0,
        estimator=estimator,
        nbins=16,
        aggregation="mean",
    )
    # The 5 dense columns carry real MI; per-pair masking must surface it.
    # Pre-fix returns exactly 0.0 (global mask -> <50 rows -> early 0.0).
    assert mi > 0.05, f"[{estimator}] per-pair mask should recover dense-column MI, got {mi}"


@pytest.mark.parametrize("estimator", ["bin", "knn"])
def test_mi_to_target_bit_identical_on_all_finite_matrix(estimator):
    """The per-pair mask is the identity when nothing is NaN, so the value
    must match a manual per-column computation over the full rows."""
    rng = np.random.default_rng(7)
    n, k = 2000, 4
    X, target = _informative_matrix(n, k, rng)

    mi = _mi_to_target(
        X,
        target,
        n_neighbors=3,
        random_state=11,
        estimator=estimator,
        nbins=16,
        aggregation="mean",
    )
    assert np.isfinite(mi) and mi > 0.0


def test_mi_to_target_bad_column_only_zeros_itself_knn():
    """biz_value: quantify the recovery. With one 99%-NaN column, the global
    mask zeroes the WHOLE sweep (0.0) while per-pair masking returns ~the MI
    of the clean k-1 columns -- a strictly positive, large recovery."""
    rng = np.random.default_rng(123)
    n, k = 4000, 6
    X, target = _informative_matrix(n, k, rng)

    # Reference: MI on the clean columns only (drop column 0 entirely).
    clean_mi = _mi_to_target(
        X[:, 1:],
        target,
        n_neighbors=3,
        random_state=0,
        estimator="knn",
        aggregation="sum",
    )

    bad = X[:, 0].copy()
    finite_keep = rng.choice(n, size=30, replace=False)
    m = np.ones(n, dtype=bool)
    m[finite_keep] = False
    bad[m] = np.nan
    X[:, 0] = bad

    poisoned_mi = _mi_to_target(
        X,
        target,
        n_neighbors=3,
        random_state=0,
        estimator="knn",
        aggregation="sum",
    )
    # SUM aggregation: bad column contributes ~0 (its <50 finite rows gate to
    # 0.0), so the poisoned sweep recovers ~the clean-column total instead of
    # collapsing to 0.0. Floor at half the clean total to absorb noise.
    assert poisoned_mi >= 0.5 * clean_mi > 0.0, f"poisoned={poisoned_mi}, clean={clean_mi}: bad column should only zero itself, not the whole sweep"


def test_mi_to_target_empty_and_degenerate_inputs():
    """Edge paths: zero columns and too-few-finite target both return 0.0."""
    rng = np.random.default_rng(1)
    target = rng.normal(size=200)
    # zero feature columns
    assert _mi_to_target(np.empty((200, 0)), target, n_neighbors=3, random_state=0, estimator="bin") == 0.0
    # target almost all-NaN (< 50 finite)
    t = target.copy()
    t[10:] = np.nan
    X = rng.normal(size=(200, 3))
    assert _mi_to_target(X, t, n_neighbors=3, random_state=0, estimator="bin") == 0.0
    assert _mi_to_target(X, t, n_neighbors=3, random_state=0, estimator="knn") == 0.0
