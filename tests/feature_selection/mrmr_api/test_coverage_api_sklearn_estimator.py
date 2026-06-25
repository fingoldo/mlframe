"""sklearn-estimator compliance coverage for MRMR.

Gap addressed: the sibling ``test_coverage_core_selection_*`` files cover support/names
consistency and reproducibility, but NOT the BaseEstimator contract itself --
``get_params``/``set_params`` round-trip over the constructor, ``clone`` reproducing
behaviour, ``fit`` returning ``self``, and the fitted-attribute conventions
(``feature_names_in_`` / ``n_features_in_``).

KNOWN GAP (pending fix, do NOT assert the correct contract yet): MRMR.__init__
resolves ``n_jobs == -1`` to ``psutil.cpu_count`` and materialises a ``None``
``parallel_kwargs`` into a dict BEFORE ``store_params_in_object``. This violates the
sklearn rule that ``__init__`` stores ctor args unmodified, so ``get_params`` echoes
the resolved value for those two params. The lazy-resolve fix lives at the fit site in
an in-flux module, so the two tests below pin the CURRENT (resolved-in-init) behaviour
rather than failing; flip them to assert pass-through once the fit-site lazy resolve lands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone

from mlframe.feature_selection.filters import MRMR


def _data(n: int = 160, m: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, m))
    y = (X[:, 0] + 0.4 * X[:, 2] > 0).astype(np.int32)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(m)]), y


def _fast(**kw):
    base = dict(full_npermutations=5, baseline_npermutations=3, n_jobs=1, verbose=0,
                fe_fast_search=False, interactions_max_order=1)
    base.update(kw)
    return MRMR(**base)


@pytest.mark.fast
def test_get_set_params_round_trip_over_ctor():
    """Every non-resolved ctor param survives a get_params -> set_params round-trip verbatim."""
    m = MRMR(quantization_nbins=7, n_workers=3, random_seed=123, bur_lambda=0.25,
             mi_normalization="su", nan_strategy="ffill_bfill")
    params = m.get_params()
    assert params["quantization_nbins"] == 7
    assert params["n_workers"] == 3
    assert params["random_seed"] == 123
    assert params["bur_lambda"] == 0.25
    assert params["mi_normalization"] == "su"
    assert params["nan_strategy"] == "ffill_bfill"

    m2 = MRMR().set_params(**params)
    assert m2.get_params()["quantization_nbins"] == 7
    assert m2.get_params()["n_workers"] == 3
    assert m2.get_params()["mi_normalization"] == "su"


@pytest.mark.fast
def test_set_params_mutates_in_place_and_returns_self():
    m = MRMR()
    ret = m.set_params(quantization_nbins=9, random_seed=5)
    assert ret is m
    assert m.quantization_nbins == 9
    assert m.random_seed == 5


@pytest.mark.fast
def test_n_jobs_minus_one_is_resolved_in_init_current_behaviour():
    """PENDING FIX: n_jobs=-1 is resolved to a concrete core count inside __init__
    (sklearn-contract violation: __init__ should store ctor args unmodified).
    The lazy-resolve fix lives at the fit site in an in-flux module, so this pins the
    CURRENT behaviour. Flip to ``== -1`` once the fit-site lazy resolve lands."""
    m = MRMR(n_jobs=-1)
    resolved = m.get_params()["n_jobs"]
    assert isinstance(resolved, int) and resolved >= 1 and resolved != -1


@pytest.mark.fast
def test_parallel_kwargs_none_is_materialised_in_init_current_behaviour():
    """PENDING FIX (same in-flux fit-site lazy-resolve as n_jobs): a None parallel_kwargs
    is turned into a concrete dict inside __init__. Pins current behaviour; flip to
    ``is None`` once lazily resolved."""
    m = MRMR(parallel_kwargs=None)
    pk = m.get_params()["parallel_kwargs"]
    assert isinstance(pk, dict) and pk.get("backend") == "threading"


@pytest.mark.fast
def test_explicit_n_jobs_passes_through():
    """A concrete (non -1) n_jobs is NOT mutated and round-trips."""
    assert MRMR(n_jobs=2).get_params()["n_jobs"] == 2


@pytest.mark.fast
def test_fit_returns_self():
    X, y = _data(seed=1)
    MRMR._FIT_CACHE.clear()
    m = _fast(random_seed=7)
    assert m.fit(X, y) is m


@pytest.mark.fast
def test_fit_sets_feature_names_in_and_n_features_in():
    X, y = _data(seed=2)
    MRMR._FIT_CACHE.clear()
    m = _fast(random_seed=7).fit(X, y)
    assert m.n_features_in_ == X.shape[1]
    np.testing.assert_array_equal(np.asarray(m.feature_names_in_, dtype=object),
                                  np.asarray(list(X.columns), dtype=object))


@pytest.mark.fast
def test_ndarray_fit_synthesizes_feature_names():
    X, y = _data(seed=3)
    MRMR._FIT_CACHE.clear()
    m = _fast(random_seed=7).fit(X.to_numpy(), y)
    assert m.n_features_in_ == X.shape[1]
    # ndarray input -> placeholder names, and the synthesized sentinel is set.
    assert getattr(m, "_feature_names_in_synthesized_", False) is True
    assert all(str(n).startswith("f") or str(n).startswith("feature_")
               for n in m.feature_names_in_)


def test_clone_reproduces_selection_behaviour():
    """clone(MRMR(...)) produces an unfitted estimator that, fit on the same data+seed,
    yields the same support_ as the original."""
    X, y = _data(seed=4)
    MRMR._FIT_CACHE.clear()
    orig = _fast(random_seed=42)
    orig.fit(X, y)
    sup_orig = np.sort(np.asarray(orig.support_))

    fresh = clone(orig)
    # clone must be unfitted.
    assert not hasattr(fresh, "support_")
    MRMR._FIT_CACHE.clear()
    fresh.fit(X, y)
    sup_clone = np.sort(np.asarray(fresh.support_))
    np.testing.assert_array_equal(sup_orig, sup_clone)


def test_clone_preserves_all_explicit_params():
    """clone copies the full constructor configuration (get_params identity)."""
    orig = _fast(random_seed=42, quantization_nbins=8, bur_lambda=0.1, n_workers=2)
    fresh = clone(orig)
    op, fp = orig.get_params(), fresh.get_params()
    # Compare on the keys that are NOT resolved-in-init (n_jobs/parallel_kwargs excluded).
    for k in ("quantization_nbins", "bur_lambda", "n_workers", "random_seed",
              "mi_normalization", "nan_strategy"):
        assert op[k] == fp[k], k
