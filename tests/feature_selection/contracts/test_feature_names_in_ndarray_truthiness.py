"""Regression: no code path may test ``feature_names_in_`` for truthiness (``or []``, bare
``if``/``not``/``and``) once the attribute is set.

2026-07 mypy cleanup (commit 920f8bc72, "mrmr class trio 32 -> 0") normalised
``_fit_identity_shortcut``'s ``feature_names_in_`` assignment from a plain list to
``np.asarray(_names, dtype=object)`` to match the OTHER two assignment sites already in
``_mrmr_class_fit_helpers.py`` -- but two sites in the MAIN fit path
(``_fit_impl_core.py``, lines ~5118/5120) still assigned a plain list, so the attribute's
runtime type silently differed depending on which code path set it.

Fixing that main-path inconsistency (making it an ndarray everywhere, matching sklearn's
own ``BaseEstimator._check_feature_names`` convention) surfaced a LATENT bug: ~13 call
sites across the FE package used the idiom ``getattr(self, "feature_names_in_", []) or []``
to normalise a possibly-missing attribute to an empty container. ``X or []`` evaluates
``bool(X)``; for a multi-element numpy array this raises
``ValueError: The truth value of an array with more than one element is ambiguous``. This
was previously masked because the main fit path's list happened to make ``or`` safe (a
non-empty list is simply truthy, no ambiguity) -- only the identity-shortcut path (an
ndarray) could ever have tripped it, and that path is rarely exercised directly.

This test pins the real-world symptom: fitting a real MRMR (main path, no identity
shortcut) and then calling every one of the previously-broken helper functions with the
fitted instance must not raise. It fails pre-fix (dict/list/set container built from an
ndarray ``feature_names_in_`` via ``X or []``) and passes post-fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _fitted_mrmr():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(
        {f"x{i}": rng.standard_normal(n) for i in range(6)},
    )
    y = pd.Series(X["x0"] * 2 + X["x1"] - X["x2"] + rng.standard_normal(n) * 0.1, name="y")
    m = MRMR(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        quantization_nbins=10,
        random_seed=0,
    )
    m.fit(X, y)
    return m, X, y


def test_feature_names_in_is_ndarray_after_main_fit():
    """Pin the canonical type so a future change doesn't silently flip it back to list
    (which would re-hide the truthiness bug this file guards against)."""
    m, _, _ = _fitted_mrmr()
    assert isinstance(m.feature_names_in_, np.ndarray)


def test_build_usability_lists_no_crash():
    from mlframe.feature_selection.filters._usability_lists import build_usability_lists

    m, X, y = _fitted_mrmr()
    build_usability_lists(m, X, np.asarray(y, dtype=np.float64))
    assert hasattr(m, "support_nonlinear_")


def test_retain_usable_pure_forms_no_crash():
    from mlframe.feature_selection.filters._fe_pure_form_retention import retain_usable_pure_forms

    m, X, y = _fitted_mrmr()
    retain_usable_pure_forms(m, X, np.asarray(y, dtype=np.float64))


def test_retain_usable_raw_columns_no_crash():
    from mlframe.feature_selection.filters._fe_pure_form_retention import retain_usable_raw_columns

    m, X, y = _fitted_mrmr()
    retain_usable_raw_columns(m, X, np.asarray(y, dtype=np.float64))


def test_compute_fe_provenance_no_crash():
    from mlframe.feature_selection.filters._mrmr_fe_provenance import compute_fe_provenance

    m, _, _ = _fitted_mrmr()
    compute_fe_provenance(m)
