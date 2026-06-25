"""Core-selection coverage: seed reproducibility, n_workers invariance, and column-order tie-break determinism.

Contracts:
- Same ``random_seed`` + same data + cleared cache -> identical ``support_`` and identical ``mrmr_gains_``.
- ``n_workers in {1, 2, 4}`` -> identical ``support_`` (the worker count is a parallelism knob, never a result knob).
- Permuting the INPUT COLUMN ORDER permutes the support indices consistently but selects the SAME underlying
  features by NAME (the greedy criterion is order-invariant on its score; ties break deterministically).

The existing ``tests/feature_selection/test_concurrency_determinism.py`` covers n_jobs (joblib worker count) and
seed-repeat, but NOT the distinct ``n_workers`` knob, NOT mrmr_gains_ reproducibility, and NOT column-order
tie-break invariance. Those gaps are filled here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _no_fe(**kw):
    base = dict(
        random_seed=0, verbose=0, fe_max_steps=0, interactions_max_order=1,
        dcd_enable=False, cluster_aggregate_enable=False, build_friend_graph=False,
        cat_fe_config=None, fe_hinge_enable=False, fe_modular_enable=False,
        fe_pairwise_modular_enable=False, fe_integer_lattice_enable=False,
        fe_row_argmax_enable=False, fe_conditional_gate_enable=False,
    )
    base.update(kw)
    return MRMR(**base)


def _data(n=600, seed=7):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = pd.DataFrame({
        "x0": x0, "x1": x1, "x2": x2,
        "noise0": rng.normal(size=n), "noise1": rng.normal(size=n),
    })
    y = (1.5 * x0 + x1 - 0.5 * x2 > 0).astype(int)
    return X, y


def _selected_names(m):
    return [str(m.feature_names_in_[i]) for i in np.sort(np.asarray(m.support_, dtype=np.intp))]


def test_same_seed_same_support_and_gains():
    """Same seed + cleared cache -> identical support_ AND identical mrmr_gains_."""
    X, y = _data()
    MRMR._FIT_CACHE.clear()
    a = _no_fe(random_seed=42).fit(X, y)
    sup_a = np.sort(np.asarray(a.support_, dtype=np.intp))
    gains_a = np.asarray(a.mrmr_gains_, dtype=float).copy()

    MRMR._FIT_CACHE.clear()
    b = _no_fe(random_seed=42).fit(X, y)
    sup_b = np.sort(np.asarray(b.support_, dtype=np.intp))
    gains_b = np.asarray(b.mrmr_gains_, dtype=float).copy()

    np.testing.assert_array_equal(sup_a, sup_b)
    np.testing.assert_allclose(np.sort(gains_a), np.sort(gains_b), rtol=0, atol=0)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_n_workers_invariant_support(workers):
    """n_workers (the MRMR parallelism knob, distinct from n_jobs) must NOT change support_ vs the 1-worker run."""
    X, y = _data()
    MRMR._FIT_CACHE.clear()
    base = _no_fe(n_workers=1, random_seed=3).fit(X, y)
    sup_base = np.sort(np.asarray(base.support_, dtype=np.intp))

    MRMR._FIT_CACHE.clear()
    other = _no_fe(n_workers=workers, random_seed=3).fit(X, y)
    sup_other = np.sort(np.asarray(other.support_, dtype=np.intp))

    np.testing.assert_array_equal(sup_base, sup_other)


def test_column_order_permutation_selects_same_features_by_name():
    """Permuting input columns permutes indices but selects the SAME features by NAME (deterministic tie-break)."""
    X, y = _data()
    MRMR._FIT_CACHE.clear()
    m1 = _no_fe(random_seed=5).fit(X, y)
    names1 = set(_selected_names(m1))

    perm = ["noise1", "x2", "x0", "noise0", "x1"]
    Xp = X[perm]
    MRMR._FIT_CACHE.clear()
    m2 = _no_fe(random_seed=5).fit(Xp, y)
    names2 = set(_selected_names(m2))

    assert names1 == names2, f"column reorder changed the selected feature SET: {names1} vs {names2}"


def test_repeated_fit_same_instance_is_stable():
    """Re-fitting the SAME instance on the same data yields the same support_ (cache replay or recompute)."""
    X, y = _data()
    m = _no_fe(random_seed=9)
    m.fit(X, y)
    sup1 = np.sort(np.asarray(m.support_, dtype=np.intp))
    m.fit(X, y)
    sup2 = np.sort(np.asarray(m.support_, dtype=np.intp))
    np.testing.assert_array_equal(sup1, sup2)
