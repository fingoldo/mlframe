"""The honest supervised null for ``fayyad_irani`` with ``mdlp_fast_mode=False`` runs the batched njit chain and must select exactly what the
categorise-per-permutation path selects.

``mdlp_fast_mode=False`` is the production default, so spelling it out is the same categorisation. The njit spec used to be excluded for any
kwarg outside ``max_depth``/``max_adaptive_nbins``, which sent this configuration to one ``categorize_dataset`` call per permutation (about
20x slower). The equality is checked end to end: the same fit runs once with the spec and once with ``build_honest_spec`` forced to None.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _fixture():
    """Binary target driven by two of five continuous columns; small enough for a fast fit."""
    rng = np.random.default_rng(11)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"].to_numpy() + 0.5 * X["f1"].to_numpy() + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
    return X, y


def _fit(X, y, monkeypatch=None, disable_spec=False):
    """Fit MRMR on the fixture; return (selected features, importance vector, number of spec builds)."""
    from mlframe.feature_selection.filters import _honest_spec
    from mlframe.feature_selection.filters.mrmr import MRMR

    builds = {"n": 0}
    real_build = _honest_spec.build_honest_spec

    def _spy(*args, **kwargs):
        """Count spec builds, returning None instead when the categorise-per-permutation route is requested."""
        builds["n"] += 1
        return None if disable_spec else real_build(*args, **kwargs)

    if monkeypatch is not None:
        monkeypatch.setattr(_honest_spec, "build_honest_spec", _spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(
            nbins_strategy="fayyad_irani",
            nbins_strategy_kwargs={"mdlp_fast_mode": False},
            quantization_nbins=8,
            full_npermutations=5,
            baseline_npermutations=5,
            fe_max_steps=0,
            random_state=0,
            verbose=0,
        )
        sel.fit(X, y)
    selected = [int(i) for i in np.asarray(sel.support_).ravel()]
    importances = np.asarray(getattr(sel, "feature_importances_", []), dtype=np.float64)
    return selected, importances, builds["n"]


def test_honest_null_fast_mode_off_selects_identically_to_categorise_path(monkeypatch):
    """The batched njit null and the per-permutation categorise path select the same features with bit-identical importances."""
    X, y = _fixture()
    sel_spec, imp_spec, spec_builds = _fit(X, y, monkeypatch, disable_spec=False)
    sel_cat, imp_cat, _ = _fit(X, y, monkeypatch, disable_spec=True)
    assert spec_builds > 0, "the njit spec must be built for fayyad_irani with mdlp_fast_mode=False"
    assert sel_spec == sel_cat, f"selection diverged: spec={sel_spec} categorise={sel_cat}"
    np.testing.assert_array_equal(imp_spec, imp_cat)


@pytest.mark.parametrize("kwargs", [{"mdlp_fast_mode": True}])
def test_fast_mode_on_keeps_categorise_path(monkeypatch, kwargs):
    """``mdlp_fast_mode=True`` is outside the njit chain and must not build the spec."""
    from mlframe.feature_selection.filters import _honest_spec
    from mlframe.feature_selection.filters.mrmr import MRMR

    builds = {"n": 0}
    real_build = _honest_spec.build_honest_spec

    def _spy(*args, **kw):
        """Count spec builds."""
        builds["n"] += 1
        return real_build(*args, **kw)

    monkeypatch.setattr(_honest_spec, "build_honest_spec", _spy)
    X, y = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        MRMR(
            nbins_strategy="fayyad_irani", nbins_strategy_kwargs=kwargs, quantization_nbins=8, full_npermutations=5,
            baseline_npermutations=5, fe_max_steps=0, random_state=0, verbose=0,
        ).fit(X, y)
    assert builds["n"] == 0
